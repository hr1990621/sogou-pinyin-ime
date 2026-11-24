"""User Memory Store (SQLite-backed with in-memory index).

Provides Sogou-style persistent learning:
1. Dynamic word frequency + recency boost (选词记忆、常用词置顶)
2. Automatic new-phrase learning (自动新造词学习)
3. Contextual bigram transition memory (上下文词组联想记忆)
4. Full CRUD management & JSON export/import
"""

from __future__ import annotations

import math
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class UserWordEntry:
    word: str
    pinyin: str            # space-separated, e.g. "ma yi yin li"
    pinyin_compact: str    # e.g. "mayiyinli"
    initials: str          # e.g. "myyl"
    freq: int
    is_custom: bool
    last_used: float
    created_at: float

    def to_dict(self) -> dict:
        return {
            "word": self.word,
            "pinyin": self.pinyin,
            "initials": self.initials,
            "freq": self.freq,
            "is_custom": self.is_custom,
            "last_used": self.last_used,
            "created_at": self.created_at,
        }


@dataclass
class UserBigramEntry:
    prev_word: str
    next_word: str
    next_pinyin: str
    freq: int
    last_used: float

    def to_dict(self) -> dict:
        return {
            "prev_word": self.prev_word,
            "next_word": self.next_word,
            "next_pinyin": self.next_pinyin,
            "freq": self.freq,
            "last_used": self.last_used,
        }


def _normalize_pinyin(pinyin: str) -> Tuple[str, str, str]:
    """Return (space_separated, compact, initials)."""
    parts = [p for p in pinyin.lower().replace("'", " ").split() if p]
    space_py = " ".join(parts)
    compact_py = "".join(parts)
    initials = "".join(p[0] for p in parts if p)
    return space_py, compact_py, initials


class MemoryStore:
    """Thread-safe SQLite + in-memory cache for user typing memory."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_schema()

        # In-memory caches for sub-millisecond query performance
        self.words_by_key: Dict[Tuple[str, str], UserWordEntry] = {}
        self.words_by_text: Dict[str, UserWordEntry] = {}
        self.words_by_pinyin: Dict[str, List[UserWordEntry]] = {}
        self.words_by_initials: Dict[str, List[UserWordEntry]] = {}
        self.bigrams_by_prev: Dict[str, Dict[str, UserBigramEntry]] = {}
        self._load_cache()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_words (
                    word TEXT NOT NULL,
                    pinyin TEXT NOT NULL,
                    pinyin_compact TEXT NOT NULL,
                    initials TEXT NOT NULL,
                    freq INTEGER NOT NULL DEFAULT 1,
                    is_custom INTEGER NOT NULL DEFAULT 0,
                    last_used REAL NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (word, pinyin)
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_bigrams (
                    prev_word TEXT NOT NULL,
                    next_word TEXT NOT NULL,
                    next_pinyin TEXT NOT NULL DEFAULT '',
                    freq INTEGER NOT NULL DEFAULT 1,
                    last_used REAL NOT NULL,
                    PRIMARY KEY (prev_word, next_word)
                );
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_uw_pinyin ON user_words(pinyin);"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_uw_initials ON user_words(initials);"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ub_prev ON user_bigrams(prev_word);"
            )

    def _load_cache(self) -> None:
        with self._lock:
            self.words_by_key.clear()
            self.words_by_text.clear()
            self.words_by_pinyin.clear()
            self.words_by_initials.clear()
            self.bigrams_by_prev.clear()

            cur = self._conn.execute(
                "SELECT word, pinyin, pinyin_compact, initials, freq, is_custom, last_used, created_at FROM user_words"
            )
            for row in cur.fetchall():
                entry = UserWordEntry(
                    word=row[0],
                    pinyin=row[1],
                    pinyin_compact=row[2],
                    initials=row[3],
                    freq=row[4],
                    is_custom=bool(row[5]),
                    last_used=row[6],
                    created_at=row[7],
                )
                self._index_word(entry)

            cur = self._conn.execute(
                "SELECT prev_word, next_word, next_pinyin, freq, last_used FROM user_bigrams"
            )
            for row in cur.fetchall():
                bg = UserBigramEntry(
                    prev_word=row[0],
                    next_word=row[1],
                    next_pinyin=row[2],
                    freq=row[3],
                    last_used=row[4],
                )
                self.bigrams_by_prev.setdefault(bg.prev_word, {})[bg.next_word] = bg

    def _index_word(self, entry: UserWordEntry) -> None:
        key = (entry.word, entry.pinyin)
        self.words_by_key[key] = entry
        existing_text = self.words_by_text.get(entry.word)
        if existing_text is None or entry.freq >= existing_text.freq:
            self.words_by_text[entry.word] = entry

        py_list = self.words_by_pinyin.setdefault(entry.pinyin, [])
        if entry not in py_list:
            py_list.append(entry)

        init_list = self.words_by_initials.setdefault(entry.initials, [])
        if entry not in init_list:
            init_list.append(entry)

    def record_selection(self, word: str, pinyin: str, is_custom: bool = False) -> Optional[UserWordEntry]:
        """Record a user word selection (or newly composed phrase)."""
        word = word.strip()
        if not word:
            return None
        space_py, compact_py, initials = _normalize_pinyin(pinyin)
        if not space_py:
            return None

        now = time.time()
        with self._lock:
            key = (word, space_py)
            existing = self.words_by_key.get(key)
            if existing:
                existing.freq += 1
                existing.last_used = now
                if is_custom:
                    existing.is_custom = True
                entry = existing
            else:
                entry = UserWordEntry(
                    word=word,
                    pinyin=space_py,
                    pinyin_compact=compact_py,
                    initials=initials,
                    freq=1,
                    is_custom=is_custom,
                    last_used=now,
                    created_at=now,
                )
                self._index_word(entry)

            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO user_words (word, pinyin, pinyin_compact, initials, freq, is_custom, last_used, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(word, pinyin) DO UPDATE SET
                        freq = freq + 1,
                        is_custom = MAX(is_custom, excluded.is_custom),
                        last_used = excluded.last_used
                    """,
                    (
                        entry.word,
                        entry.pinyin,
                        entry.pinyin_compact,
                        entry.initials,
                        entry.freq,
                        1 if entry.is_custom else 0,
                        entry.last_used,
                        entry.created_at,
                    ),
                )
            return entry

    def record_bigram(self, prev_word: str, next_word: str, next_pinyin: str = "") -> Optional[UserBigramEntry]:
        """Learn a contextual transition prev_word -> next_word."""
        prev_word = prev_word.strip()
        next_word = next_word.strip()
        if not prev_word or not next_word:
            return None
        # Limit key lengths to avoid storing entire paragraphs
        if len(prev_word) > 12 or len(next_word) > 16:
            return None

        space_py, _, _ = _normalize_pinyin(next_pinyin)
        now = time.time()
        with self._lock:
            bucket = self.bigrams_by_prev.setdefault(prev_word, {})
            existing = bucket.get(next_word)
            if existing:
                existing.freq += 1
                existing.last_used = now
                if space_py:
                    existing.next_pinyin = space_py
                entry = existing
            else:
                entry = UserBigramEntry(
                    prev_word=prev_word,
                    next_word=next_word,
                    next_pinyin=space_py,
                    freq=1,
                    last_used=now,
                )
                bucket[next_word] = entry

            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO user_bigrams (prev_word, next_word, next_pinyin, freq, last_used)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(prev_word, next_word) DO UPDATE SET
                        freq = freq + 1,
                        next_pinyin = CASE WHEN excluded.next_pinyin != '' THEN excluded.next_pinyin ELSE user_bigrams.next_pinyin END,
                        last_used = excluded.last_used
                    """,
                    (entry.prev_word, entry.next_word, entry.next_pinyin, entry.freq, entry.last_used),
                )
            return entry

    def compute_recency_weight(self, freq: int, last_used: float) -> float:
        """Compute memory boost combining frequency and exponential recency decay."""
        age_hours = max(0.0, (time.time() - last_used) / 3600.0)
        # Half-life ~ 14 days (336 hours), floor at 0.35 so old frequent words remain remembered
        recency = 0.35 + 0.65 * math.exp(-age_hours / 336.0)
        return (4.5 + 3.2 * math.log1p(freq)) * recency

    def get_word_memory_score(self, word: str, pinyin: Optional[str] = None) -> Tuple[float, int, bool]:
        """Return (bonus_score, user_freq, is_custom) for a candidate word."""
        with self._lock:
            entry = None
            if pinyin:
                space_py, _, _ = _normalize_pinyin(pinyin)
                entry = self.words_by_key.get((word, space_py))
            if entry is None:
                entry = self.words_by_text.get(word)
            if entry is None:
                return 0.0, 0, False
            bonus = self.compute_recency_weight(entry.freq, entry.last_used)
            if entry.is_custom:
                bonus += 2.5
            return bonus, entry.freq, entry.is_custom

    def lookup_exact_pinyin(self, pinyin_space: str) -> List[UserWordEntry]:
        with self._lock:
            return list(self.words_by_pinyin.get(pinyin_space, []))

    def lookup_initials_prefix(self, initials_prefix: str, limit: int = 20) -> List[UserWordEntry]:
        if not initials_prefix:
            return []
        with self._lock:
            matches: List[UserWordEntry] = []
            for init, entries in self.words_by_initials.items():
                if init.startswith(initials_prefix):
                    matches.extend(entries)
            matches.sort(
                key=lambda e: self.compute_recency_weight(e.freq, e.last_used),
                reverse=True,
            )
            return matches[:limit]

    def lookup_compact_prefix(self, raw_prefix: str, limit: int = 20) -> List[UserWordEntry]:
        if not raw_prefix:
            return []
        with self._lock:
            matches = [
                e
                for e in self.words_by_key.values()
                if e.pinyin_compact.startswith(raw_prefix)
            ]
            matches.sort(
                key=lambda e: self.compute_recency_weight(e.freq, e.last_used),
                reverse=True,
            )
            return matches[:limit]

    def get_context_candidates(self, prev_context: str, limit: int = 10) -> List[Tuple[str, float, int, str]]:
        """Predict next words or suffix completions from learned user memory.

        Returns list of (completion_text, score, freq, source_label).
        """
        if not prev_context:
            return []
        ctx = prev_context.strip()
        if not ctx:
            return []

        results: Dict[str, Tuple[float, int, str]] = {}
        with self._lock:
            # 1) Learned Bigram transitions matching tail of context (longest match gets higher boost)
            for tail_len in (6, 5, 4, 3, 2, 1):
                if len(ctx) < tail_len:
                    continue
                tail = ctx[-tail_len:]
                bucket = self.bigrams_by_prev.get(tail)
                if not bucket:
                    continue
                length_bonus = 1.0 + tail_len * 0.45
                for next_word, bg in bucket.items():
                    sc = (self.compute_recency_weight(bg.freq, bg.last_used) + 6.0) * length_bonus
                    old = results.get(next_word)
                    if old is None or sc > old[0]:
                        results[next_word] = (sc, bg.freq, f"记忆联想({tail}→)")

            # 2) Learned custom/frequent multi-char phrases whose prefix matches tail of context
            # e.g. user learned "今天天气真好", context ends with "今天天" -> predict "气真好"
            for entry in self.words_by_text.values():
                w = entry.word
                if len(w) < 2:
                    continue
                max_overlap = min(len(w) - 1, len(ctx), 8)
                for k in range(max_overlap, 0, -1):
                    if ctx.endswith(w[:k]):
                        suffix = w[k:]
                        # Prefer longer overlap (e.g. 3 chars > 2 chars > 1 char)
                        if k >= 2 or entry.freq >= 2 or entry.is_custom:
                            sc = (self.compute_recency_weight(entry.freq, entry.last_used) + 4.0) * (0.9 + k * 0.6)
                            old = results.get(suffix)
                            if old is None or sc > old[0]:
                                results[suffix] = (sc, entry.freq, f"记忆补全({w[:k]}+)")
                        break

        ranked = [
            (text, vals[0], vals[1], vals[2])
            for text, vals in results.items()
        ]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:limit]

    def get_bigram_boost(self, prev_context: str, candidate_word: str) -> float:
        """Return extra score if candidate_word frequently follows prev_context."""
        if not prev_context or not candidate_word:
            return 0.0
        ctx = prev_context.strip()
        best = 0.0
        with self._lock:
            for tail_len in (4, 3, 2, 1):
                if len(ctx) < tail_len:
                    continue
                tail = ctx[-tail_len:]
                bg = self.bigrams_by_prev.get(tail, {}).get(candidate_word)
                if bg:
                    sc = self.compute_recency_weight(bg.freq, bg.last_used) * (1.0 + tail_len * 0.25)
                    if sc > best:
                        best = sc
        return best

    def list_words(self, limit: int = 200) -> List[dict]:
        with self._lock:
            items = sorted(
                self.words_by_key.values(),
                key=lambda e: (e.last_used, e.freq),
                reverse=True,
            )[:limit]
            return [e.to_dict() for e in items]

    def list_bigrams(self, limit: int = 100) -> List[dict]:
        with self._lock:
            items: List[UserBigramEntry] = []
            for bucket in self.bigrams_by_prev.values():
                items.extend(bucket.values())
            items.sort(key=lambda e: (e.last_used, e.freq), reverse=True)
            return [e.to_dict() for e in items[:limit]]

    def delete_word(self, word: str, pinyin: Optional[str] = None) -> int:
        with self._lock:
            with self._conn:
                if pinyin:
                    space_py, _, _ = _normalize_pinyin(pinyin)
                    cur = self._conn.execute(
                        "DELETE FROM user_words WHERE word = ? AND pinyin = ?",
                        (word, space_py),
                    )
                else:
                    cur = self._conn.execute(
                        "DELETE FROM user_words WHERE word = ?",
                        (word,),
                    )
                deleted = cur.rowcount
            self._load_cache()
            return deleted

    def delete_bigram(self, prev_word: str, next_word: str) -> int:
        with self._lock:
            with self._conn:
                cur = self._conn.execute(
                    "DELETE FROM user_bigrams WHERE prev_word = ? AND next_word = ?",
                    (prev_word, next_word),
                )
                deleted = cur.rowcount
            self._load_cache()
            return deleted

    def clear_all(self) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute("DELETE FROM user_words")
                self._conn.execute("DELETE FROM user_bigrams")
            self._load_cache()

    def get_stats(self) -> dict:
        with self._lock:
            total_words = len(self.words_by_key)
            custom_words = sum(1 for e in self.words_by_key.values() if e.is_custom)
            total_bigrams = sum(len(b) for b in self.bigrams_by_prev.values())
            total_uses = sum(e.freq for e in self.words_by_key.values())
            return {
                "total_words": total_words,
                "custom_words": custom_words,
                "total_bigrams": total_bigrams,
                "total_uses": total_uses,
            }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
