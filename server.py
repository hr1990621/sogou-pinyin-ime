"""Local HTTP API + Static Web Server for Sogou-Style Pinyin IME.

Zero external dependencies (Python 3.13 stdlib only).
Run: python server.py [--host 127.0.0.1] [--port 8765] [--no-browser]
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from engine.ime_engine import PinyinIMEEngine

WEB_DIR = ROOT / "web"
DICT_DB = ROOT / "data" / "ime_dict.db"
MEMORY_DB = ROOT / "data" / "user_memory.db"


class IMEAppContext:
    def __init__(self) -> None:
        if not DICT_DB.exists():
            print("Dictionary DB not found, building from raw dicts...")
            from scripts.build_dict import build_database
            build_database()
        self.engine = PinyinIMEEngine(DICT_DB, MEMORY_DB)


def make_handler(ctx: IMEAppContext):
    class IMEHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            # Keep terminal output clean; log only errors or API startup
            return

        def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8"))

        def _serve_static(self, rel_path: str) -> None:
            if rel_path in ("", "/"):
                rel_path = "index.html"
            rel_path = rel_path.lstrip("/")
            file_path = (WEB_DIR / rel_path).resolve()
            # Prevent path traversal
            if not str(file_path).startswith(str(WEB_DIR.resolve())) or not file_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "File not found")
                return
            mime, _ = mimetypes.guess_type(str(file_path))
            if mime is None:
                mime = "application/octet-stream"
            if mime.startswith("text/") or mime in ("application/javascript", "application/json"):
                mime += "; charset=utf-8"
            data = file_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/api/stats":
                self._send_json(ctx.engine.get_stats())
                return
            if path == "/api/memory":
                self._send_json(
                    {
                        "words": ctx.engine.memory.list_words(limit=300),
                        "bigrams": ctx.engine.memory.list_bigrams(limit=150),
                        "stats": ctx.engine.memory.get_stats(),
                    }
                )
                return
            if path == "/api/memory/export":
                payload = {
                    "words": ctx.engine.memory.list_words(limit=5000),
                    "bigrams": ctx.engine.memory.list_bigrams(limit=5000),
                }
                data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="sogou_ime_memory.json"')
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

            self._serve_static(path)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            try:
                body = self._read_json()
            except Exception as exc:
                self._send_json({"error": f"Invalid JSON: {exc}"}, status=HTTPStatus.BAD_REQUEST)
                return

            if path == "/api/query":
                result = ctx.engine.query(
                    raw_pinyin=str(body.get("raw", "")),
                    context=str(body.get("context", "")),
                    limit=int(body.get("limit", 35)),
                    enable_memory=bool(body.get("enable_memory", True)),
                    enable_autocomplete=bool(body.get("enable_autocomplete", True)),
                )
                self._send_json(result)
                return

            if path == "/api/predict":
                result = ctx.engine.predict_completion(
                    context=str(body.get("context", "")),
                    limit=int(body.get("limit", 8)),
                    enable_memory=bool(body.get("enable_memory", True)),
                )
                self._send_json(result)
                return

            if path == "/api/commit":
                cp = body.get("composed_phrase")
                composed = None
                if isinstance(cp, list) and len(cp) == 2 and cp[0] and cp[1]:
                    composed = (str(cp[0]), str(cp[1]))
                result = ctx.engine.commit(
                    word=str(body.get("word", "")),
                    pinyin=str(body.get("pinyin", "")),
                    prev_context=str(body.get("prev_context", "")),
                    composed_phrase=composed,
                    enable_memory=bool(body.get("enable_memory", True)),
                    enable_autocomplete=bool(body.get("enable_autocomplete", True)),
                )
                self._send_json(result)
                return

            if path == "/api/memory/word":
                word = str(body.get("word", "")).strip()
                pinyin = str(body.get("pinyin", "")).strip()
                if not word or not pinyin:
                    self._send_json({"error": "word and pinyin are required"}, status=HTTPStatus.BAD_REQUEST)
                    return
                entry = ctx.engine.memory.record_selection(word, pinyin, is_custom=True)
                self._send_json({"ok": True, "entry": entry.to_dict() if entry else None})
                return

            if path == "/api/memory/clear":
                ctx.engine.memory.clear_all()
                self._send_json({"ok": True, "stats": ctx.engine.memory.get_stats()})
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Unknown API endpoint")

        def do_DELETE(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            try:
                body = self._read_json()
            except Exception as exc:
                self._send_json({"error": f"Invalid JSON: {exc}"}, status=HTTPStatus.BAD_REQUEST)
                return

            if path == "/api/memory/word":
                deleted = ctx.engine.memory.delete_word(
                    str(body.get("word", "")),
                    body.get("pinyin"),
                )
                self._send_json({"ok": True, "deleted": deleted, "stats": ctx.engine.memory.get_stats()})
                return

            if path == "/api/memory/bigram":
                deleted = ctx.engine.memory.delete_bigram(
                    str(body.get("prev_word", "")),
                    str(body.get("next_word", "")),
                )
                self._send_json({"ok": True, "deleted": deleted, "stats": ctx.engine.memory.get_stats()})
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Unknown API endpoint")

    return IMEHandler


def main() -> None:
    parser = argparse.ArgumentParser(description="Sogou-Style Pinyin IME Web Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    ctx = IMEAppContext()
    stats = ctx.engine.get_stats()
    d = stats["dictionary"]
    m = stats["memory"]
    print(
        f"[搜影拼音 IME] 词库已加载: {d['total_entries']:,} 词条 "
        f"(热点前缀索引 {d['hot_index_size']:,}, 加载耗时 {d['load_ms']}ms) | "
        f"用户记忆: {m['total_words']} 词 / {m['total_bigrams']} 联想对"
    )

    server = ThreadingHTTPServer((args.host, args.port), make_handler(ctx))
    url = f"http://{args.host}:{args.port}"
    print(f"[搜影拼音 IME] 服务已启动: {url}")

    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在关闭服务...")
    finally:
        server.server_close()
        ctx.engine.close()


if __name__ == "__main__":
    main()
