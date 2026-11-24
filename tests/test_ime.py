"""Unit tests for Sogou-Style Pinyin IME Engine (segmentation, autocomplete, memory)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.ime_engine import PinyinIMEEngine
from engine.pinyin_parser import segment_pinyin


class TestPinyinIME(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.engine = PinyinIMEEngine(
            dict_db_path=ROOT / "data" / "ime_dict.db",
            memory_db_path=Path(self.tmpdir.name) / "test_memory.db",
        )

    def tearDown(self) -> None:
        self.engine.close()
        self.tmpdir.cleanup()

    def test_pinyin_segmentation(self) -> None:
        self.assertEqual(segment_pinyin("nihao")[0].display, "ni'hao")
        self.assertEqual(segment_pinyin("jintiantianqizhenhao")[0].display, "jin'tian'tian'qi'zhen'hao")
        self.assertEqual(segment_pinyin("xi'an")[0].display, "xi'an")
        self.assertEqual(segment_pinyin("rgzn")[0].display, "r'g'z'n")

    def test_full_pinyin_and_viterbi_sentence(self) -> None:
        r1 = self.engine.query("nihao")
        self.assertEqual(r1["candidates"][0]["word"], "你好")

        r2 = self.engine.query("woxiangqubeijinglvyou")
        self.assertEqual(r2["candidates"][0]["word"], "我想去北京旅游")
        self.assertEqual(r2["candidates"][0]["tag"], "整句")

    def test_jianpin_and_prefix_autocomplete(self) -> None:
        r1 = self.engine.query("rgzn")
        words = [c["word"] for c in r1["candidates"][:5]]
        self.assertEqual(words[0], "人工智能")
        self.assertIn("人工智能大模型", words)

        r2 = self.engine.query("jttq")
        words2 = [c["word"] for c in r2["candidates"][:5]]
        self.assertEqual(words2[0], "今天天气")
        self.assertIn("今天天气真好", words2)

        r3 = self.engine.query("zhongg")
        self.assertEqual(r3["candidates"][0]["word"], "中国")

    def test_context_ghost_completion(self) -> None:
        p1 = self.engine.predict_completion("今天天")
        self.assertEqual(p1["ghost_text"], "气真好")

        p2 = self.engine.predict_completion("人工")
        self.assertEqual(p2["ghost_text"], "智能")

        p3 = self.engine.predict_completion("自然语言")
        self.assertEqual(p3["ghost_text"], "处理")

    def test_memory_learning_new_phrase_and_bigram(self) -> None:
        # 1. Auto-learn new composed phrase "蚂蚁引力"
        self.engine.commit(
            word="引力",
            pinyin="yin li",
            prev_context="蚂蚁",
            composed_phrase=("蚂蚁引力", "ma yi yin li"),
        )
        r_full = self.engine.query("mayiyinli")
        self.assertEqual(r_full["candidates"][0]["word"], "蚂蚁引力")
        self.assertEqual(r_full["candidates"][0]["tag"], "新词")

        r_jp = self.engine.query("myyl")
        self.assertEqual(r_jp["candidates"][0]["word"], "蚂蚁引力")

        # 2. Bigram context memory learning: "明天" -> "写代码"
        self.engine.commit(word="写代码", pinyin="xie dai ma", prev_context="明天")
        pred = self.engine.predict_completion("明天")
        self.assertEqual(pred["ghost_text"], "写代码")

    def test_memory_persistence_across_restart(self) -> None:
        mem_path = Path(self.tmpdir.name) / "persist.db"
        e1 = PinyinIMEEngine(ROOT / "data" / "ime_dict.db", mem_path)
        e1.memory.record_selection("搜影输入法", "sou ying shu ru fa", is_custom=True)
        e1.close()

        e2 = PinyinIMEEngine(ROOT / "data" / "ime_dict.db", mem_path)
        try:
            res = e2.query("sysrf")
            self.assertEqual(res["candidates"][0]["word"], "搜影输入法")
        finally:
            e2.close()


if __name__ == "__main__":
    unittest.main()
