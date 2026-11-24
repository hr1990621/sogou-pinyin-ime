"""Mandarin Pinyin syllable parser & segmenter.

Supports:
- Full Pinyin (全拼): e.g. 'jintiantianqi' -> ['jin', 'tian', 'tian', 'qi']
- Initials / Jianpin (简拼): e.g. 'jttq' -> ['j', 't', 't', 'q']
- Mixed Pinyin (混拼): e.g. 'jinttq' -> ['jin', 't', 't', 'q']
- Trailing incomplete syllable prefix: e.g. 'zhonggu' -> ['zhong', 'gu'] or 'shuruf' -> ['shu', 'ru', 'f']
- Explicit apostrophe separator: e.g. "xi'an" -> ['xi', 'an']
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

# Complete standard Mandarin Pinyin syllable table (including nv/lv/nve/lve/m/n/ng/hm/hng/er)
VALID_SYLLABLES: Set[str] = {
    # Vowel-initial (zero initial)
    "a", "ai", "an", "ang", "ao",
    "e", "ei", "en", "eng", "er",
    "o", "ou",
    # b
    "ba", "bai", "ban", "bang", "bao", "bei", "ben", "beng",
    "bi", "bian", "biao", "bie", "bin", "bing", "bo", "bu",
    # p
    "pa", "pai", "pan", "pang", "pao", "pei", "pen", "peng",
    "pi", "pian", "piao", "pie", "pin", "ping", "po", "pou", "pu",
    # m
    "ma", "mai", "man", "mang", "mao", "me", "mei", "men", "meng",
    "mi", "mian", "miao", "mie", "min", "ming", "miu", "mo", "mou", "mu",
    # f
    "fa", "fan", "fang", "fei", "fen", "feng", "fiao", "fo", "fou", "fu",
    # d
    "da", "dai", "dan", "dang", "dao", "de", "dei", "den", "deng",
    "di", "dia", "dian", "diao", "die", "ding", "diu",
    "dong", "dou", "du", "duan", "dui", "dun", "duo",
    # t
    "ta", "tai", "tan", "tang", "tao", "te", "tei", "teng",
    "ti", "tian", "tiao", "tie", "ting",
    "tong", "tou", "tu", "tuan", "tui", "tun", "tuo",
    # n
    "na", "nai", "nan", "nang", "nao", "ne", "nei", "nen", "neng",
    "ni", "nian", "niang", "niao", "nie", "nin", "ning", "niu",
    "nong", "nou", "nu", "nuan", "nun", "nuo", "nv", "nve",
    # l
    "la", "lai", "lan", "lang", "lao", "le", "lei", "leng",
    "li", "lia", "lian", "liang", "liao", "lie", "lin", "ling", "liu",
    "lo", "long", "lou", "lu", "luan", "lun", "luo", "lv", "lve",
    # g
    "ga", "gai", "gan", "gang", "gao", "ge", "gei", "gen", "geng",
    "gong", "gou", "gu", "gua", "guai", "guan", "guang", "gui", "gun", "guo",
    # k
    "ka", "kai", "kan", "kang", "kao", "ke", "kei", "ken", "keng",
    "kong", "kou", "ku", "kua", "kuai", "kuan", "kuang", "kui", "kun", "kuo",
    # h
    "ha", "hai", "han", "hang", "hao", "he", "hei", "hen", "heng",
    "hong", "hou", "hu", "hua", "huai", "huan", "huang", "hui", "hun", "huo", "hm", "hng",
    # j
    "ji", "jia", "jian", "jiang", "jiao", "jie", "jin", "jing", "jiong", "jiu",
    "ju", "juan", "jue", "jun",
    # q
    "qi", "qia", "qian", "qiang", "qiao", "qie", "qin", "qing", "qiong", "qiu",
    "qu", "quan", "que", "qun",
    # x
    "xi", "xia", "xian", "xiang", "xiao", "xie", "xin", "xing", "xiong", "xiu",
    "xu", "xuan", "xue", "xun",
    # zh
    "zha", "zhai", "zhan", "zhang", "zhao", "zhe", "zhei", "zhen", "zheng",
    "zhi", "zhong", "zhou", "zhu", "zhua", "zhuai", "zhuan", "zhuang", "zhui", "zhun", "zhuo",
    # ch
    "cha", "chai", "chan", "chang", "chao", "che", "chen", "cheng",
    "chi", "chong", "chou", "chu", "chua", "chuai", "chuan", "chuang", "chui", "chun", "chuo",
    # sh
    "sha", "shai", "shan", "shang", "shao", "she", "shei", "shen", "sheng",
    "shi", "shou", "shu", "shua", "shuai", "shuan", "shuang", "shui", "shun", "shuo",
    # r
    "ran", "rang", "rao", "re", "ren", "reng", "ri",
    "rong", "rou", "ru", "rua", "ruan", "rui", "run", "ruo",
    # z
    "za", "zai", "zan", "zang", "zao", "ze", "zei", "zen", "zeng",
    "zi", "zong", "zou", "zu", "zuan", "zui", "zun", "zuo",
    # c
    "ca", "cai", "can", "cang", "cao", "ce", "cen", "ceng",
    "ci", "cong", "cou", "cu", "cuan", "cui", "cun", "cuo",
    # s
    "sa", "sai", "san", "sang", "sao", "se", "sen", "seng",
    "si", "song", "sou", "su", "suan", "sui", "sun", "suo",
    # y
    "ya", "yan", "yang", "yao", "ye", "yi", "yin", "ying", "yo", "yong", "you",
    "yu", "yuan", "yue", "yun",
    # w
    "wa", "wai", "wan", "wang", "wei", "wen", "weng", "wo", "wu",
}

VALID_INITIALS: Set[str] = {
    "b", "p", "m", "f", "d", "t", "n", "l",
    "g", "k", "h", "j", "q", "x",
    "zh", "ch", "sh", "r", "z", "c", "s",
    "y", "w",
}

# Precompute all valid non-empty prefixes of valid syllables (for incomplete syllable detection)
SYLLABLE_PREFIXES: Set[str] = set()
PREFIX_TO_SYLLABLES: Dict[str, List[str]] = {}
for _syl in sorted(VALID_SYLLABLES):
    for _i in range(1, len(_syl) + 1):
        _pref = _syl[:_i]
        SYLLABLE_PREFIXES.add(_pref)
        PREFIX_TO_SYLLABLES.setdefault(_pref, []).append(_syl)


def get_initial(syllable: str) -> str:
    """Return the first-letter initial of a syllable (used for Jianpin indexing)."""
    return syllable[0] if syllable else ""


def get_strict_initial(syllable: str) -> str:
    """Return full initial (zh/ch/sh or single char)."""
    if len(syllable) >= 2 and syllable[:2] in ("zh", "ch", "sh"):
        return syllable[:2]
    return syllable[0] if syllable else ""


@dataclass(frozen=True)
class Segmentation:
    """A candidate segmentation of raw pinyin input."""
    tokens: Tuple[str, ...]        # e.g. ('jin', 'tian', 'tia')
    is_complete: Tuple[bool, ...]  # whether each token is a complete valid syllable
    score: float                   # higher is better

    @property
    def display(self) -> str:
        return "'".join(self.tokens)

    @property
    def initials(self) -> str:
        return "".join(t[0] for t in self.tokens if t)

    @property
    def all_complete(self) -> bool:
        return all(self.is_complete)


def _segment_chunk(chunk: str, max_results: int = 4) -> List[Segmentation]:
    """Segment a contiguous lowercase alpha string without apostrophes.

    Uses beam DP to find top-k segmentations preferring:
    1. Complete valid syllables over partials/initials
    2. Longer complete syllables (e.g. 'xian' vs 'xi'+'an', while still keeping 'xi'+'an' in beam)
    3. Allowing trailing partial prefix (for autocomplete like 'zhonggu' -> ['zhong', 'gu'])
    4. Allowing single-letter/zh/ch/sh initials for Jianpin (e.g. 'nhsj' -> ['n','h','s','j'])
    """
    n = len(chunk)
    if n == 0:
        return [Segmentation((), (), 0.0)]

    # dp[i] holds list of (score, tokens_list, complete_list)
    dp: List[List[Tuple[float, List[str], List[bool]]]] = [[] for _ in range(n + 1)]
    dp[0] = [(0.0, [], [])]

    beam_width = max(max_results * 3, 10)

    for i in range(n):
        if not dp[i]:
            continue
        # Prune states at position i
        states = sorted(dp[i], key=lambda x: x[0], reverse=True)[:beam_width]
        max_len = min(6, n - i)
        matched_any = False

        for length in range(max_len, 0, -1):
            sub = chunk[i : i + length]
            end = i + length

            if sub in VALID_SYLLABLES:
                # Per-character reward + length cohesion bonus minus per-token boundary cost
                # Ensures longer complete syllables always beat fragmented shorter syllables/initials
                step_score = length * 10.0 + (length - 1) * 2.5 - 2.0
                for sc, toks, comps in states:
                    dp[end].append((sc + step_score, toks + [sub], comps + [True]))
                matched_any = True
            elif end == n and len(sub) >= 2 and sub in SYLLABLE_PREFIXES:
                # Trailing incomplete syllable (e.g. 'jintia' -> 'jin' + 'tia', 'zhonggu' -> 'zhong' + 'gu')
                step_score = length * 7.5 - 1.0
                for sc, toks, comps in states:
                    dp[end].append((sc + step_score, toks + [sub], comps + [False]))
                matched_any = True
            elif sub in VALID_INITIALS:
                # Initial token for Jianpin / mixed input (e.g. 'rgzn', 'zhongg')
                step_score = length * 2.0 - 3.0
                for sc, toks, comps in states:
                    dp[end].append((sc + step_score, toks + [sub], comps + [False]))
                matched_any = True
            elif end == n and sub in SYLLABLE_PREFIXES:
                step_score = length * 4.0 - 2.0
                for sc, toks, comps in states:
                    dp[end].append((sc + step_score, toks + [sub], comps + [False]))
                matched_any = True

        # Fallback: consume 1 unknown character (e.g. 'v' or typo) so DP never dies
        if not matched_any:
            sub = chunk[i : i + 1]
            for sc, toks, comps in states:
                dp[i + 1].append((sc - 12.0, toks + [sub], comps + [False]))

    final_states = dp[n]
    if not final_states:
        return [Segmentation((chunk,), (False,), -10.0)]

    # Deduplicate by token tuple and sort by score descending
    best_by_tokens: Dict[Tuple[str, ...], Tuple[float, Tuple[bool, ...]]] = {}
    for sc, toks, comps in final_states:
        key = tuple(toks)
        # Prefer fewer tokens slightly when scores tie
        adj_sc = sc - len(toks) * 0.35
        if key not in best_by_tokens or adj_sc > best_by_tokens[key][0]:
            best_by_tokens[key] = (adj_sc, tuple(comps))

    results = [
        Segmentation(tokens=k, is_complete=v[1], score=v[0])
        for k, v in best_by_tokens.items()
    ]
    results.sort(key=lambda s: s.score, reverse=True)
    return results[:max_results]


def segment_pinyin(raw: str, max_results: int = 4) -> List[Segmentation]:
    """Segment raw pinyin string (may contain apostrophe `'` separators)."""
    cleaned = raw.lower().strip()
    if not cleaned:
        return []

    parts = [p for p in cleaned.split("'") if p]
    if not parts:
        return []

    if len(parts) == 1:
        return _segment_chunk(parts[0], max_results=max_results)

    # Combine segmentations across explicit apostrophe boundaries
    combined: List[Segmentation] = [Segmentation((), (), 0.0)]
    for part in parts:
        sub_segs = _segment_chunk(part, max_results=2)
        next_combined: List[Segmentation] = []
        for base in combined:
            for sub in sub_segs:
                next_combined.append(
                    Segmentation(
                        tokens=base.tokens + sub.tokens,
                        is_complete=base.is_complete + sub.is_complete,
                        score=base.score + sub.score,
                    )
                )
        next_combined.sort(key=lambda s: s.score, reverse=True)
        combined = next_combined[:max_results]
    return combined


def token_matches_syllable(token: str, syllable: str, is_last: bool = False) -> bool:
    """Check if a parsed pinyin token matches a dictionary syllable."""
    if token == syllable:
        return True
    # Initial match (e.g. 'z' matches 'zhong', 'zh' matches 'zhong')
    if token in VALID_INITIALS and syllable.startswith(token):
        return True
    # Prefix match on trailing incomplete syllable (e.g. 'tia' matches 'tian')
    if is_last and syllable.startswith(token):
        return True
    return False
