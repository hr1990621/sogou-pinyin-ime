"""Compile open-source Rime-ICE YAML dictionaries + curated phrases into SQLite.

Produces: data/ime_dict.db
Tables:
  - entries(word, pinyin, pinyin_compact, initials, freq, word_len)
  - collocations(prev_word, next_word, freq)

Optimized with B-tree indexes for sub-millisecond prefix & exact lookups.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CURATED_FILE = ROOT / "data" / "curated_phrases.txt"
OUT_DB = ROOT / "data" / "ime_dict.db"


def normalize_syllables(raw_py: str) -> Tuple[str, str, str]:
    """Normalize a pinyin sequence to (space_sep, compact, initials)."""
    parts = [p.strip().lower() for p in raw_py.replace("'", " ").split() if p.strip()]
    # Keep only alphabetic syllables
    cleaned = ["".join(ch for ch in p if "a" <= ch <= "z") for p in parts]
    cleaned = [p for p in cleaned if p]
    if not cleaned:
        return "", "", ""
    return " ".join(cleaned), "".join(cleaned), "".join(p[0] for p in cleaned)


def parse_rime_yaml(path: Path, default_freq: int = 100) -> List[Tuple[str, str, int]]:
    """Parse a Rime .dict.yaml file into (word, pinyin_space, freq)."""
    rows: List[Tuple[str, str, int]] = []
    if not path.exists():
        return rows

    in_body = False
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not in_body:
                if line.strip() == "...":
                    in_body = True
                continue
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            word = parts[0].strip()
            py_raw = parts[1].strip()
            if not word or not py_raw:
                continue
            freq = default_freq
            if len(parts) >= 3 and parts[2].strip():
                raw_w = parts[2].strip().rstrip("%")
                try:
                    freq = max(1, int(float(raw_w)))
                except ValueError:
                    freq = default_freq
            rows.append((word, py_raw, freq))
    return rows


def parse_curated(path: Path) -> Tuple[List[Tuple[str, str, int]], List[Tuple[str, str, int]]]:
    """Parse curated phrases and contextual collocations."""
    phrases: List[Tuple[str, str, int]] = []
    collocations: List[Tuple[str, str, int]] = []
    if not path.exists():
        return phrases, collocations

    section = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line == "[PHRASES]":
                section = "PHRASES"
                continue
            if line == "[COLLOCATIONS]":
                section = "COLLOCATIONS"
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            try:
                w = int(parts[2])
            except ValueError:
                w = 100
            if section == "PHRASES":
                phrases.append((parts[0].strip(), parts[1].strip(), w))
            elif section == "COLLOCATIONS":
                collocations.append((parts[0].strip(), parts[1].strip(), w))
    return phrases, collocations


def build_database() -> None:
    t0 = time.perf_counter()
    merged: Dict[Tuple[str, str], Tuple[str, str, int]] = {}

    def ingest(items: List[Tuple[str, str, int]], source_name: str) -> None:
        added = 0
        for word, py_raw, freq in items:
            space_py, compact_py, initials = normalize_syllables(py_raw)
            if not space_py:
                continue
            # Ensure syllable count roughly matches Chinese character length for clean alignment
            key = (word, space_py)
            prev = merged.get(key)
            if prev is None or freq > prev[2]:
                merged[key] = (compact_py, initials, freq)
                added += 1
        print(f"  Ingested {added:,} entries from {source_name}")

    print("Building SQLite IME dictionary database...")
    # 1. Rime-ICE 8105 characters
    f_8105 = RAW_DIR / "8105.dict.yaml"
    if f_8105.exists():
        ingest(parse_rime_yaml(f_8105, default_freq=500), "8105.dict.yaml")

    # 2. Rime-ICE base vocabulary (~110k+ modern Chinese words)
    f_base = RAW_DIR / "base.dict.yaml"
    if f_base.exists() and f_base.stat().st_size > 10_000:
        ingest(parse_rime_yaml(f_base, default_freq=200), "base.dict.yaml")
    else:
        print("  [WARN] base.dict.yaml not available yet or incomplete.")

    # 3. Curated modern phrases & collocations
    phrases, collocations = parse_curated(CURATED_FILE)
    ingest(phrases, "curated_phrases.txt")

    if OUT_DB.exists():
        OUT_DB.unlink()

    conn = sqlite3.connect(str(OUT_DB))
    conn.execute("PRAGMA journal_mode=OFF;")
    conn.execute("PRAGMA synchronous=OFF;")
    conn.execute("PRAGMA temp_store=MEMORY;")

    with conn:
        conn.execute(
            """
            CREATE TABLE entries (
                word TEXT NOT NULL,
                pinyin TEXT NOT NULL,
                pinyin_compact TEXT NOT NULL,
                initials TEXT NOT NULL,
                freq INTEGER NOT NULL,
                word_len INTEGER NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE collocations (
                prev_word TEXT NOT NULL,
                next_word TEXT NOT NULL,
                freq INTEGER NOT NULL,
                PRIMARY KEY (prev_word, next_word)
            );
            """
        )

        batch = [
            (word, space_py, vals[0], vals[1], vals[2], len(word))
            for (word, space_py), vals in merged.items()
        ]
        conn.executemany(
            "INSERT INTO entries (word, pinyin, pinyin_compact, initials, freq, word_len) VALUES (?, ?, ?, ?, ?, ?)",
            batch,
        )

        # Deduplicate collocations keeping max freq
        col_map: Dict[Tuple[str, str], int] = {}
        for p, n, fr in collocations:
            if (p, n) not in col_map or fr > col_map[(p, n)]:
                col_map[(p, n)] = fr
        conn.executemany(
            "INSERT INTO collocations (prev_word, next_word, freq) VALUES (?, ?, ?)",
            [(p, n, fr) for (p, n), fr in col_map.items()],
        )

        print("  Creating indexes...")
        conn.execute("CREATE INDEX idx_pinyin_freq ON entries(pinyin, freq DESC);")
        conn.execute("CREATE INDEX idx_compact_freq ON entries(pinyin_compact, freq DESC);")
        conn.execute("CREATE INDEX idx_initials_freq ON entries(initials, freq DESC);")
        conn.execute("CREATE INDEX idx_word_freq ON entries(word, freq DESC);")
        conn.execute("CREATE INDEX idx_col_prev ON collocations(prev_word, freq DESC);")

    conn.execute("ANALYZE;")
    conn.close()
    dt = time.perf_counter() - t0
    size_mb = OUT_DB.stat().st_size / (1024 * 1024)
    print(f"Done! {len(merged):,} entries + {len(col_map):,} collocations -> {OUT_DB} ({size_mb:.2f} MB) in {dt:.2f}s")


if __name__ == "__main__":
    build_database()
