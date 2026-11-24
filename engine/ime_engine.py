"""Core Sogou-Style Pinyin IME Engine.

Combines:
1. 552k+ Rime-ICE SQLite dictionary (exact B-tree lookups < 0.5ms)
2. In-memory prefix indexes over top-40k hot vocabulary (prefix & Chinese completion < 1ms)
3. Viterbi DP sentence decoder (整句智能组句)
4. User Memory Store (动态词频记忆 + 自动新造词 + 上下文 Bigram 联想)
5. Ghost-Text Inline Completion & Post-Commit Association Engine
"""

from __future__ import annotations

import bisect
import math
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from engine.memory_store import MemoryStore
from engine.pinyin_parser import (
    Segmentation,
    VALID_INITIALS,
    VALID_SYLLABLES,
    segment_pinyin,
    token_matches_syllable,
)


@dataclass
class Candidate:
    word: str
    pinyin: str              # space-separated full pinyin of this candidate
    consumed_tokens: int     # how many tokens of the primary segmentation this covers
    consumed_chars: int      # how many raw input chars this covers
    score: float
    tag: str                 # '记忆' | '新词' | '整句' | '补全' | '简拼' | '词库' | '联想'
    priority_tier: int = 1
    base_score: float = 0.0
    memory_score: float = 0.0
    context_score: float = 0.0
    completion_hint: str = ""  # remaining pinyin or completion hint for UI
    user_freq: int = 0
    dict_freq: int = 0

    def to_dict(self) -> dict:
        return {
            "word": self.word,
            "pinyin": self.pinyin,
            "consumed_tokens": self.consumed_tokens,
            "consumed_chars": self.consumed_chars,
            "score": round(self.score, 2),
            "priority_tier": self.priority_tier,
            "tag": self.tag,
            "breakdown": {
                "base": round(self.base_score, 2),
                "memory": round(self.memory_score, 2),
                "context": round(self.context_score, 2),
            },
            "completion_hint": self.completion_hint,
            "user_freq": self.user_freq,
            "dict_freq": self.dict_freq,
        }


class PinyinIMEEngine:
    """Thread-safe Chinese Pinyin IME Engine."""

    def __init__(self, dict_db_path: Path, memory_db_path: Path):
        self.dict_db_path = Path(dict_db_path)
        self.memory = MemoryStore(memory_db_path)
        self._db_lock = threading.RLock()
        self._conn = sqlite3.connect(
            f"file:{self.dict_db_path}?mode=ro", uri=True, check_same_thread=False
        )
        self._conn.execute("PRAGMA cache_size=-32000;")  # 32MB SQLite page cache

        # In-memory hot indexes for fast prefix autocomplete & Chinese completion
        # Sorted lists of (key, freq, word, pinyin_space)
        self._compact_keys: List[str] = []
        self._compact_items: List[Tuple[str, int, str, str]] = []

        self._initial_keys: List[str] = []
        self._initial_items: List[Tuple[str, int, str, str]] = []

        self._word_keys: List[str] = []
        self._word_items: List[Tuple[str, int, str]] = []  # (word, freq, pinyin)

        self._collocations: Dict[str, List[Tuple[str, int]]] = {}
        self.dict_stats: Dict[str, int] = {}

        self._load_hot_indexes()

    def _load_hot_indexes(self) -> None:
        t0 = time.perf_counter()
        with self._db_lock:
            total_entries = self._conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
            total_chars = self._conn.execute("SELECT COUNT(*) FROM entries WHERE word_len = 1").fetchone()[0]
            total_words = total_entries - total_chars

            # Load collocations
            for prev_w, next_w, fr in self._conn.execute(
                "SELECT prev_word, next_word, freq FROM collocations ORDER BY freq DESC"
            ):
                self._collocations.setdefault(prev_w, []).append((next_w, fr))

            # Load top 45,000 multi-character words into memory for blazing-fast prefix & Chinese completion
            cur = self._conn.execute(
                """
                SELECT word, pinyin, pinyin_compact, initials, freq
                FROM entries
                WHERE word_len >= 2
                ORDER BY freq DESC
                LIMIT 45000
                """
            )
            compact_raw: List[Tuple[str, int, str, str]] = []
            initial_raw: List[Tuple[str, int, str, str]] = []
            word_raw: List[Tuple[str, int, str]] = []

            for word, py, py_c, init, fr in cur.fetchall():
                compact_raw.append((py_c, fr, word, py))
                initial_raw.append((init, fr, word, py))
                word_raw.append((word, fr, py))

            compact_raw.sort(key=lambda x: x[0])
            self._compact_items = compact_raw
            self._compact_keys = [x[0] for x in compact_raw]

            initial_raw.sort(key=lambda x: x[0])
            self._initial_items = initial_raw
            self._initial_keys = [x[0] for x in initial_raw]

            word_raw.sort(key=lambda x: x[0])
            self._word_items = word_raw
            self._word_keys = [x[0] for x in word_raw]

            self.dict_stats = {
                "total_entries": total_entries,
                "single_chars": total_chars,
                "multi_words": total_words,
                "hot_index_size": len(compact_raw),
                "collocations": sum(len(v) for v in self._collocations.values()),
                "load_ms": round((time.perf_counter() - t0) * 1000, 1),
            }

    # ---------- SQLite fast exact lookups ----------

    def _query_exact_pinyin(self, pinyin_space: str, limit: int = 30) -> List[Tuple[str, str, int]]:
        with self._db_lock:
            cur = self._conn.execute(
                "SELECT word, pinyin, freq FROM entries WHERE pinyin = ? ORDER BY freq DESC LIMIT ?",
                (pinyin_space, limit),
            )
            return cur.fetchall()

    def _query_exact_initials(self, initials: str, limit: int = 25) -> List[Tuple[str, str, int]]:
        with self._db_lock:
            cur = self._conn.execute(
                "SELECT word, pinyin, freq FROM entries WHERE initials = ? AND word_len >= 2 ORDER BY freq DESC LIMIT ?",
                (initials, limit),
            )
            return cur.fetchall()

    def _query_compact_prefix(self, prefix: str, limit: int = 16) -> List[Tuple[str, str, int]]:
        """In-memory bisect prefix scan on top-45k words by compact pinyin."""
        if not prefix:
            return []
        lo = bisect.bisect_left(self._compact_keys, prefix)
        hi_key = prefix[:-1] + chr(ord(prefix[-1]) + 1)
        hi = bisect.bisect_left(self._compact_keys, hi_key)
        if lo >= hi:
            return []
        # Cap scan window if prefix is very short (e.g. 1-2 letters)
        slice_items = self._compact_items[lo : min(hi, lo + 2500)]
        # Sort by freq descending, prefer shorter completion slightly for snappiness
        slice_items.sort(key=lambda x: x[1], reverse=True)
        out: List[Tuple[str, str, int]] = []
        seen = set()
        for py_c, fr, word, py in slice_items:
            if word in seen:
                continue
            seen.add(word)
            out.append((word, py, fr))
            if len(out) >= limit:
                break
        return out

    def _query_initials_prefix(self, prefix: str, limit: int = 16) -> List[Tuple[str, str, int]]:
        """In-memory bisect prefix scan on top-45k words by initials (Jianpin prefix)."""
        if not prefix:
            return []
        lo = bisect.bisect_left(self._initial_keys, prefix)
        hi_key = prefix[:-1] + chr(ord(prefix[-1]) + 1)
        hi = bisect.bisect_left(self._initial_keys, hi_key)
        if lo >= hi:
            return []
        slice_items = self._initial_items[lo : min(hi, lo + 2500)]
        slice_items.sort(key=lambda x: x[1], reverse=True)
        out: List[Tuple[str, str, int]] = []
        seen = set()
        for init, fr, word, py in slice_items:
            if word in seen:
                continue
            seen.add(word)
            out.append((word, py, fr))
            if len(out) >= limit:
                break
        return out

    def _query_chinese_prefix(self, cn_prefix: str, limit: int = 10) -> List[Tuple[str, str, int]]:
        """In-memory bisect prefix scan on top-45k Chinese words/phrases."""
        if not cn_prefix:
            return []
        lo = bisect.bisect_left(self._word_keys, cn_prefix)
        hi_key = cn_prefix[:-1] + chr(ord(cn_prefix[-1]) + 1)
        hi = bisect.bisect_left(self._word_keys, hi_key)
        if lo >= hi:
            return []
        slice_items = self._word_items[lo : min(hi, lo + 800)]
        matches = [x for x in slice_items if len(x[0]) > len(cn_prefix)]
        matches.sort(key=lambda x: x[1], reverse=True)
        return [(w, py, fr) for w, fr, py in matches[:limit]]

    # ---------- Scoring helpers ----------

    @staticmethod
    def _dict_base_score(freq: int, word_len: int) -> float:
        """Normalize Rime-ICE frequencies so multi-char words balance cleanly against 8105 single chars."""
        log_f = math.log1p(max(1, freq))
        if word_len == 1:
            # 8105 single-char corpus frequencies are ~10x-100x larger numerically
            return log_f * 0.72
        # Multi-character words get cohesion bonus
        return log_f + 3.6 * (word_len - 1)

    def _viterbi_sentence(
        self,
        tokens: Sequence[str],
        is_complete: Sequence[bool],
        context: str,
        enable_memory: bool,
    ) -> Optional[Candidate]:
        """Viterbi DP decoder for multi-syllable full/mixed Pinyin sentences."""
        n = len(tokens)
        if n < 2:
            return None

        # dp[i] = (score, words_list, pinyins_list, base_sum, mem_sum, ctx_sum)
        dp: List[Optional[Tuple[float, List[str], List[str], float, float, float]]] = [None] * (n + 1)
        dp[0] = (0.0, [], [], 0.0, 0.0, 0.0)

        for i in range(n):
            state = dp[i]
            if state is None:
                continue
            cur_score, cur_words, cur_pys, cur_base, cur_mem, cur_ctx = state
            prev_ctx = cur_words[-1] if cur_words else context

            # Try spans of length 1..min(6, n-i)
            max_span = min(6, n - i)
            for span in range(1, max_span + 1):
                end = i + span
                sub_tokens = tokens[i:end]
                sub_complete = is_complete[i:end]

                best_word = None
                best_py = ""
                best_span_score = -1e9
                best_parts = (0.0, 0.0, 0.0)

                candidates_for_span: List[Tuple[str, str, int, bool]] = []

                if all(sub_complete):
                    py_key = " ".join(sub_tokens)
                    if enable_memory:
                        for ue in self.memory.lookup_exact_pinyin(py_key):
                            candidates_for_span.append((ue.word, ue.pinyin, 5000 + ue.freq * 1000, True))
                    for w, py, fr in self._query_exact_pinyin(py_key, limit=6):
                        candidates_for_span.append((w, py, fr, False))
                elif span >= 2:
                    # Mixed or Jianpin span: check initials + token filter
                    init_key = "".join(t[0] for t in sub_tokens)
                    raw_matches: List[Tuple[str, str, int, bool]] = []
                    if enable_memory:
                        for ue in self.memory.lookup_initials_prefix(init_key, limit=6):
                            if len(ue.initials) == span:
                                raw_matches.append((ue.word, ue.pinyin, 5000 + ue.freq * 1000, True))
                    for w, py, fr in self._query_exact_initials(init_key, limit=12):
                        raw_matches.append((w, py, fr, False))
                    for w, py, fr, from_mem in raw_matches:
                        syls = py.split()
                        if len(syls) == span and all(
                            token_matches_syllable(sub_tokens[k], syls[k], is_last=(i + k == n - 1))
                            for k in range(span)
                        ):
                            candidates_for_span.append((w, py, fr, from_mem))
                elif span == 1 and not sub_complete[0]:
                    # Single trailing partial syllable (e.g. 'g' at end of 'zhongg' -> '国')
                    tok = sub_tokens[0]
                    # Check top syllable starting with tok
                    if tok in VALID_INITIALS or len(tok) >= 2:
                        # Find best single char whose pinyin starts with tok
                        with self._db_lock:
                            row = self._conn.execute(
                                "SELECT word, pinyin, freq FROM entries WHERE word_len = 1 AND pinyin >= ? AND pinyin < ? ORDER BY freq DESC LIMIT 1",
                                (tok, tok[:-1] + chr(ord(tok[-1]) + 1)),
                            ).fetchone()
                            if row:
                                candidates_for_span.append((row[0], row[1], row[2], False))

                for w, py, fr, _ in candidates_for_span:
                    b_sc = self._dict_base_score(fr, len(w))
                    m_sc = 0.0
                    c_sc = 0.0
                    if enable_memory:
                        m_sc, _, _ = self.memory.get_word_memory_score(w, py)
                        c_sc = self.memory.get_bigram_boost(prev_ctx, w)
                    # Check built-in collocation boost
                    if prev_ctx:
                        for tl in (4, 3, 2, 1):
                            if len(prev_ctx) >= tl:
                                for nw, cfr in self._collocations.get(prev_ctx[-tl:], ()):
                                    if nw == w or w.startswith(nw):
                                        c_sc = max(c_sc, 3.5 + math.log1p(cfr))
                                        break
                    # Strong penalty per word boundary so multi-char words always beat fragmented single chars
                    boundary_penalty = -9.5
                    step_total = b_sc + m_sc + c_sc + boundary_penalty
                    if step_total > best_span_score:
                        best_span_score = step_total
                        best_word = w
                        best_py = py
                        best_parts = (b_sc, m_sc, c_sc)

                if best_word is not None:
                    new_score = cur_score + best_span_score
                    prev_dp = dp[end]
                    if prev_dp is None or new_score > prev_dp[0]:
                        dp[end] = (
                            new_score,
                            cur_words + [best_word],
                            cur_pys + [best_py],
                            cur_base + best_parts[0],
                            cur_mem + best_parts[1],
                            cur_ctx + best_parts[2],
                        )

        final = dp[n]
        if final is None or len(final[1]) <= 1:
            # If solved in 1 single word, the direct exact/memory lookup already handles it better
            return None

        sentence = "".join(final[1])
        full_py = " ".join(final[2])
        raw_len = sum(len(t) for t in tokens)
        # Keep Viterbi sentence competitive for novel multi-word sentences, without overriding exact full-phrase matches
        return Candidate(
            word=sentence,
            pinyin=full_py,
            consumed_tokens=n,
            consumed_chars=raw_len,
            score=final[0] + 15.0,
            tag="整句",
            base_score=final[3],
            memory_score=final[4],
            context_score=final[5],
        )

    # ---------- Main Query API ----------

    def query(
        self,
        raw_pinyin: str,
        context: str = "",
        limit: int = 35,
        enable_memory: bool = True,
        enable_autocomplete: bool = True,
    ) -> dict:
        """Query candidates for the current pinyin buffer + preceding text context."""
        t0 = time.perf_counter()
        raw_clean = "".join(ch for ch in raw_pinyin.lower() if ("a" <= ch <= "z") or ch == "'")
        raw_compact = raw_clean.replace("'", "")

        if not raw_compact:
            pred = self.predict_completion(context, enable_memory=enable_memory) if enable_autocomplete else {"ghost_text": "", "ghost_source": "", "associations": []}
            return {
                "raw": raw_pinyin,
                "display_pinyin": "",
                "segmentations": [],
                "candidates": [],
                "ghost_text": pred["ghost_text"],
                "ghost_source": pred["ghost_source"],
                "associations": pred["associations"],
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            }

        segs = segment_pinyin(raw_clean, max_results=3)
        primary = segs[0]
        tokens = primary.tokens
        n_tok = len(tokens)

        cand_map: Dict[str, Candidate] = {}

        def add_candidate(cand: Candidate) -> None:
            existing = cand_map.get(cand.word)
            if existing is None or (cand.priority_tier, cand.score) > (existing.priority_tier, existing.score):
                cand_map[cand.word] = cand

        def score_and_add(
            word: str,
            py_space: str,
            dict_freq: int,
            consumed_tokens: int,
            consumed_chars: int,
            tag: str,
            extra_bonus: float = 0.0,
            completion_hint: str = "",
        ) -> None:
            b_sc = self._dict_base_score(dict_freq, len(word)) + extra_bonus
            m_sc = 0.0
            u_freq = 0
            is_custom = False
            c_sc = 0.0
            if enable_memory:
                m_sc, u_freq, is_custom = self.memory.get_word_memory_score(word, py_space)
                c_sc = self.memory.get_bigram_boost(context, word)

            # Built-in collocation boost
            if context:
                for tl in (4, 3, 2, 1):
                    if len(context) >= tl:
                        for nw, cfr in self._collocations.get(context[-tl:], ()):
                            if nw == word:
                                c_sc = max(c_sc, 5.5 + math.log1p(cfr))
                                break

            final_tag = tag
            if is_custom and m_sc > 0:
                final_tag = "新词"
            elif m_sc > 0 and tag not in ("补全",):
                final_tag = "记忆"

            mem_weighted = m_sc * 2.2
            is_memorized = enable_memory and u_freq > 0
            if consumed_tokens < n_tok:
                priority_tier = 0
            elif is_memorized and tag != "补全":
                priority_tier = 3
            elif is_memorized and tag == "补全" and not primary.all_complete:
                priority_tier = 2
            else:
                priority_tier = 1

            if priority_tier >= 2:
                total = mem_weighted * 2.5 + c_sc * 1.5 + b_sc * 0.25 + (4.0 if is_custom else 0.0)
            else:
                total = b_sc + mem_weighted + c_sc

            add_candidate(
                Candidate(
                    word=word,
                    pinyin=py_space,
                    consumed_tokens=consumed_tokens,
                    consumed_chars=consumed_chars,
                    score=total,
                    priority_tier=priority_tier,
                    tag=final_tag,
                    base_score=b_sc,
                    memory_score=mem_weighted,
                    context_score=c_sc,
                    completion_hint=completion_hint,
                    user_freq=u_freq,
                    dict_freq=dict_freq,
                )
            )

        # 1) Full-input matches across top segmentations
        for seg_idx, seg in enumerate(segs):
            seg_penalty = 0.0 if seg_idx == 0 else -1.8
            s_toks = seg.tokens
            s_len = len(s_toks)
            s_compact_len = sum(len(t) for t in s_toks)

            if seg.all_complete:
                py_key = " ".join(s_toks)
                # User memory exact match
                if enable_memory:
                    for ue in self.memory.lookup_exact_pinyin(py_key):
                        score_and_add(
                            ue.word, ue.pinyin, 25000 + ue.freq * 5000,
                            consumed_tokens=s_len, consumed_chars=s_compact_len,
                            tag="记忆", extra_bonus=14.0 + seg_penalty,
                        )
                # Dictionary exact match
                for w, py, fr in self._query_exact_pinyin(py_key, limit=25):
                    score_and_add(
                        w, py, fr,
                        consumed_tokens=s_len, consumed_chars=s_compact_len,
                        tag="词库", extra_bonus=10.0 + seg_penalty,
                    )

            # Jianpin / Mixed Pinyin exact length match (e.g. 'nhsj' -> '你好世界', 'jinttq' -> '今天天气')
            if s_len >= 2:
                init_key = seg.initials
                if enable_memory:
                    for ue in self.memory.lookup_initials_prefix(init_key, limit=12):
                        if len(ue.initials) == s_len:
                            syls = ue.pinyin.split()
                            if len(syls) == s_len and all(
                                token_matches_syllable(s_toks[k], syls[k], is_last=(k == s_len - 1))
                                for k in range(s_len)
                            ):
                                score_and_add(
                                    ue.word, ue.pinyin, 20000 + ue.freq * 4000,
                                    consumed_tokens=s_len, consumed_chars=s_compact_len,
                                    tag="记忆", extra_bonus=11.0 + seg_penalty,
                                )

                for w, py, fr in self._query_exact_initials(init_key, limit=30):
                    syls = py.split()
                    if len(syls) == s_len and all(
                        token_matches_syllable(s_toks[k], syls[k], is_last=(k == s_len - 1))
                        for k in range(s_len)
                    ):
                        tag_name = "词库" if seg.all_complete else "简拼"
                        bonus = 9.0 if seg.all_complete else 6.8
                        score_and_add(
                            w, py, fr,
                            consumed_tokens=s_len, consumed_chars=s_compact_len,
                            tag=tag_name, extra_bonus=bonus + seg_penalty,
                        )

        # 2) Viterbi DP full-sentence composition (only when majority of tokens are complete syllables)
        complete_count = sum(1 for c in primary.is_complete if c)
        if n_tok >= 2 and complete_count >= max(2, (n_tok + 1) // 2):
            viterbi_cand = self._viterbi_sentence(
                tokens, primary.is_complete, context=context, enable_memory=enable_memory
            )
            if viterbi_cand and viterbi_cand.word not in cand_map:
                add_candidate(viterbi_cand)

        # 3) Intelligent Pinyin Prefix Auto-Completion (智能前缀拼音补全)
        # e.g. 'zhongg' -> '中国 (+uo)', 'jintia' -> '今天 (+n)', 'rgz' -> '人工智能 (+neng)'
        if enable_autocomplete and len(raw_compact) >= 2:
            # 3a) User memory prefix autocomplete (compact pinyin & initials)
            if enable_memory:
                for ue in self.memory.lookup_compact_prefix(raw_compact, limit=8):
                    if ue.pinyin_compact != raw_compact:
                        rem = ue.pinyin_compact[len(raw_compact):]
                        rem_pen = len(rem) * 0.45
                        score_and_add(
                            ue.word, ue.pinyin, 15000 + ue.freq * 3000,
                            consumed_tokens=n_tok, consumed_chars=len(raw_compact),
                            tag="补全", extra_bonus=7.5 - rem_pen, completion_hint=f"+{rem}",
                        )
                if len(primary.initials) >= 2:
                    for ue in self.memory.lookup_initials_prefix(primary.initials, limit=8):
                        syls = ue.pinyin.split()
                        if len(syls) > n_tok and all(
                            token_matches_syllable(tokens[k], syls[k], is_last=(k == n_tok - 1))
                            for k in range(n_tok)
                        ):
                            rem_py = "'".join(syls[n_tok:])
                            extra_syls = len(syls) - n_tok
                            score_and_add(
                                ue.word, ue.pinyin, 14000 + ue.freq * 3000,
                                consumed_tokens=n_tok, consumed_chars=len(raw_compact),
                                tag="补全", extra_bonus=6.5 - extra_syls * 2.5, completion_hint=f"+{rem_py}",
                            )

            # 3b) Hot dictionary compact-pinyin prefix completion (e.g. 'jintia' -> '今天', 'zhongg' -> '中国')
            for w, py, fr in self._query_compact_prefix(raw_compact, limit=18):
                py_c = py.replace(" ", "")
                if py_c == raw_compact:
                    continue
                rem = py_c[len(raw_compact):]
                extra_words = max(0, len(w) - n_tok)
                # Prefer completing the current unfinished syllable (extra_words == 0) over long multi-word extensions
                base_boost = 7.5 if (not primary.all_complete and extra_words == 0) else 2.2
                # Penalize remaining length & extra characters so '中国(+uo)' >> '中国共产党(+uogongchandang)'
                len_penalty = len(rem) * 0.75 + extra_words * 3.8
                score_and_add(
                    w, py, fr,
                    consumed_tokens=n_tok, consumed_chars=len(raw_compact),
                    tag="补全", extra_bonus=base_boost - len_penalty,
                    completion_hint=f"+{rem}",
                )

            # 3c) Hot dictionary initials prefix completion (e.g. 'jtt' -> '今天天气', 'rgz' -> '人工智能')
            if len(primary.initials) >= 2:
                for w, py, fr in self._query_initials_prefix(primary.initials, limit=16):
                    syls = py.split()
                    if len(syls) <= n_tok:
                        continue
                    if all(
                        token_matches_syllable(tokens[k], syls[k], is_last=(k == n_tok - 1))
                        for k in range(n_tok)
                    ):
                        rem_py = "'".join(syls[n_tok:])
                        extra_syls = len(syls) - n_tok
                        # Keep exact Jianpin (bonus 6.8) ahead of longer completions
                        len_penalty = extra_syls * 4.2
                        score_and_add(
                            w, py, fr,
                            consumed_tokens=n_tok, consumed_chars=len(raw_compact),
                            tag="补全", extra_bonus=4.2 - len_penalty,
                            completion_hint=f"+{rem_py}",
                        )

        # 4) Partial-prefix candidates (progressive selection like Sogou: e.g. for 'mayiyinli', also show '蚂蚁', '马', etc.)
        if n_tok >= 2:
            for prefix_len in range(n_tok - 1, 0, -1):
                sub_toks = tokens[:prefix_len]
                sub_complete = primary.is_complete[:prefix_len]
                sub_chars = sum(len(t) for t in sub_toks)
                # Penalize shorter prefix relative to full-input matches so full words stay ahead
                prefix_penalty = -6.0 - (n_tok - prefix_len) * 2.4

                if all(sub_complete):
                    py_key = " ".join(sub_toks)
                    if enable_memory:
                        for ue in self.memory.lookup_exact_pinyin(py_key)[:4]:
                            score_and_add(
                                ue.word, ue.pinyin, 10000 + ue.freq * 2000,
                                consumed_tokens=prefix_len, consumed_chars=sub_chars,
                                tag="记忆", extra_bonus=prefix_penalty + 2.0,
                            )
                    lim = 8 if prefix_len == 1 else 6
                    for w, py, fr in self._query_exact_pinyin(py_key, limit=lim):
                        score_and_add(
                            w, py, fr,
                            consumed_tokens=prefix_len, consumed_chars=sub_chars,
                            tag="词库", extra_bonus=prefix_penalty,
                        )
                elif prefix_len >= 2:
                    init_key = "".join(t[0] for t in sub_toks)
                    for w, py, fr in self._query_exact_initials(init_key, limit=4):
                        syls = py.split()
                        if len(syls) == prefix_len and all(
                            token_matches_syllable(sub_toks[k], syls[k]) for k in range(prefix_len)
                        ):
                            score_and_add(
                                w, py, fr,
                                consumed_tokens=prefix_len, consumed_chars=sub_chars,
                                tag="简拼", extra_bonus=prefix_penalty - 1.0,
                            )

        # 5) Single-letter / partial single-syllable fallback (e.g. user just typed 'z' or 'sh' or 'zhon')
        if n_tok == 1 and not primary.is_complete[0]:
            tok = tokens[0]
            # Best single chars starting with this prefix
            hi = tok[:-1] + chr(ord(tok[-1]) + 1)
            with self._db_lock:
                rows = self._conn.execute(
                    "SELECT word, pinyin, freq FROM entries WHERE word_len = 1 AND pinyin >= ? AND pinyin < ? ORDER BY freq DESC LIMIT 15",
                    (tok, hi),
                ).fetchall()
            for w, py, fr in rows:
                rem = py[len(tok):]
                score_and_add(
                    w, py, fr,
                    consumed_tokens=1, consumed_chars=len(tok),
                    tag="补全", extra_bonus=2.0, completion_hint=(f"+{rem}" if rem else ""),
                )

        # Rank all candidates (strictly by priority_tier first, then score)
        ranked = sorted(cand_map.values(), key=lambda c: (c.priority_tier, c.score), reverse=True)[:limit]

        # Build ghost-text preview for the top candidate if it's a prefix completion OR context continuation
        ghost_text = ""
        ghost_source = ""
        if enable_autocomplete and ranked:
            top = ranked[0]
            if top.tag == "补全" and top.completion_hint:
                ghost_text = top.word
                ghost_source = f"拼音补全 ({top.pinyin.replace(' ', chr(39))})"
            elif top.consumed_tokens == n_tok:
                # Predict what comes AFTER the top candidate!
                next_pred = self.predict_completion(context + top.word, enable_memory=enable_memory)
                if next_pred["ghost_text"]:
                    ghost_text = next_pred["ghost_text"]
                    ghost_source = next_pred["ghost_source"]

        return {
            "raw": raw_pinyin,
            "display_pinyin": primary.display,
            "tokens": list(tokens),
            "segmentations": [s.display for s in segs],
            "candidates": [c.to_dict() for c in ranked],
            "ghost_text": ghost_text,
            "ghost_source": ghost_source,
            "associations": [],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    # ---------- Contextual Prediction & Ghost Text ----------

    def predict_completion(self, context: str, limit: int = 8, enable_memory: bool = True) -> dict:
        """Predict next-word associations & inline ghost-text completion from preceding Chinese text."""
        if not context:
            return {"ghost_text": "", "ghost_source": "", "associations": []}

        # Keep only trailing non-punctuation Chinese/alphanumeric context
        tail_ctx = context.rstrip()
        if not tail_ctx:
            return {"ghost_text": "", "ghost_source": "", "associations": []}
        # Stop at sentence-ending punctuation
        if tail_ctx[-1] in "，。！？；：、,.!?;:\n\r\t ":
            return {"ghost_text": "", "ghost_source": "", "associations": []}

        scored: Dict[str, Tuple[float, str, str]] = {}  # text -> (score, source, tag)

        def put(text: str, score: float, source: str, tag: str) -> None:
            if not text or len(text) > 12:
                return
            old = scored.get(text)
            if old is None or score > old[0]:
                scored[text] = (score, source, tag)

        # 1) Learned User Memory (strictly highest priority tier)
        if enable_memory:
            for comp_text, sc, freq, src in self.memory.get_context_candidates(tail_ctx, limit=10):
                put(comp_text, sc + 1000.0, src, "记忆")

        # 2) Curated Contextual Collocations (e.g. "今天天" -> "气真好", "人工" -> "智能")
        for tl in (5, 4, 3, 2, 1):
            if len(tail_ctx) < tl:
                continue
            sub = tail_ctx[-tl:]
            for nxt, fr in self._collocations.get(sub, ()):
                sc = (12.0 + math.log1p(fr) * 2.2) * (1.0 + tl * 0.35)
                put(nxt, sc, f"智能联想({sub}→)", "联想")

        # 3) Hot Dictionary Chinese-Prefix Sentence/Word Completion
        # e.g. user typed "自然语言" -> completes "处理"; typed "今天天" -> completes "气真好"
        for tl in (5, 4, 3, 2):
            if len(tail_ctx) < tl:
                continue
            sub = tail_ctx[-tl:]
            for full_word, py, fr in self._query_chinese_prefix(sub, limit=8):
                suffix = full_word[tl:]
                if not suffix:
                    continue
                sc = (8.0 + math.log1p(fr)) * (0.9 + tl * 0.5)
                put(suffix, sc, f"词库补全({sub}+)", "补全")

        ranked = sorted(scored.items(), key=lambda kv: kv[1][0], reverse=True)[:limit]
        associations = [
            {
                "word": text,
                "score": round(meta[0], 2),
                "source": meta[1],
                "tag": meta[2],
            }
            for text, meta in ranked
        ]

        ghost_text = associations[0]["word"] if associations else ""
        ghost_source = associations[0]["source"] if associations else ""
        return {
            "ghost_text": ghost_text,
            "ghost_source": ghost_source,
            "associations": associations,
        }

    # ---------- Commit & Learning API ----------

    def commit(
        self,
        word: str,
        pinyin: str,
        prev_context: str = "",
        composed_phrase: Optional[Tuple[str, str]] = None,
        enable_memory: bool = True,
        enable_autocomplete: bool = True,
    ) -> dict:
        """Record user commit, update memory & bigrams, and return next associations."""
        learned_events: List[str] = []

        if enable_memory and word:
            # 1) Record selected word unigram frequency
            if pinyin:
                # Check if word exists in base dictionary
                in_dict = self._word_exists_in_dict(word)
                self.memory.record_selection(word, pinyin, is_custom=(not in_dict))
                learned_events.append(f"词频记忆 +1: {word}")

            # 2) Automatic new-phrase learning (when user composed a phrase via multiple picks)
            if composed_phrase:
                c_word, c_py = composed_phrase
                if len(c_word) >= 2 and c_word != word:
                    in_dict = self._word_exists_in_dict(c_word)
                    self.memory.record_selection(c_word, c_py, is_custom=(not in_dict))
                    learned_events.append(f"自动新造词: {c_word} ({c_py})")

            # 3) Contextual Bigram learning from immediately preceding Chinese phrase
            prev_clean = self._extract_prev_token(prev_context)
            if prev_clean and prev_clean != word:
                self.memory.record_bigram(prev_clean, word, pinyin)
                learned_events.append(f"上下文记忆: {prev_clean} → {word}")

        new_context = prev_context + word
        pred = (
            self.predict_completion(new_context, enable_memory=enable_memory)
            if enable_autocomplete
            else {"ghost_text": "", "ghost_source": "", "associations": []}
        )

        return {
            "committed": word,
            "learned_events": learned_events,
            "ghost_text": pred["ghost_text"],
            "ghost_source": pred["ghost_source"],
            "associations": pred["associations"],
            "memory_stats": self.memory.get_stats(),
        }

    def _word_exists_in_dict(self, word: str) -> bool:
        with self._db_lock:
            row = self._conn.execute(
                "SELECT 1 FROM entries WHERE word = ? LIMIT 1", (word,)
            ).fetchone()
            return row is not None

    @staticmethod
    def _extract_prev_token(prev_context: str) -> str:
        """Extract the trailing 2-4 Chinese chars before the new word for bigram learning."""
        if not prev_context:
            return ""
        s = prev_context.rstrip()
        if not s:
            return ""
        # Collect contiguous trailing Chinese characters
        buf: List[str] = []
        for ch in reversed(s):
            if "\u4e00" <= ch <= "\u9fff":
                buf.append(ch)
                if len(buf) >= 4:
                    break
            else:
                break
        if not buf:
            return ""
        tail = "".join(reversed(buf))
        # Prefer last 2 chars as canonical prev_word unless whole tail is 2-3 chars
        return tail[-2:] if len(tail) >= 3 else tail

    def get_stats(self) -> dict:
        return {
            "dictionary": self.dict_stats,
            "memory": self.memory.get_stats(),
        }

    def close(self) -> None:
        self.memory.close()
        with self._db_lock:
            self._conn.close()
