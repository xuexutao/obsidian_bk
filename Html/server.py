#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Obsidian 知识库本地读写与静态资源服务 (Port 60051)
支持：
1. 静态文件与 vault/ 软链接服务（全接口支持 CORS 跨域与 OPTIONS 预检，兼容 file:// 直开、本地端口或远程预览代理）
2. GET  /api/read?path=<rel_path>  直接读取最新原始 .md 文档
3. POST /api/save                  将修改后的 .md 内容原子写入 /Users/eilonxtxue/Documents/obsidian/obsidian_bk 对应文件
"""
import os
import json
import time
from urllib.parse import urlparse, parse_qs, unquote
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

WORKSPACE_ROOT = "/Users/eilonxtxue/With/20261020/3qaq"
VAULT_ROOT = os.environ.get("VAULT_ROOT", "/Users/eilonxtxue/Documents/obsidian/obsidian_bk")
PORT = int(os.environ.get("PORT", "60051"))

import sys
import subprocess
sys.path.insert(0, WORKSPACE_ROOT)
import build_manifest as _bm

_manifest_cache = {"t": 0.0, "data": None}


def get_live_manifest():
    now = time.time()
    if _manifest_cache["data"] is not None and now - _manifest_cache["t"] < 2.0:
        return _manifest_cache["data"]
    tree, md_files, asset_map, _docs = _bm.build_tree_and_lists(VAULT_ROOT)
    data = {
        "vaultPath": VAULT_ROOT,
        "generatedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
        "totalCount": len(md_files),
        "noteCount": sum(1 for x in md_files if not x["isAssetDoc"]),
        "mathTotal": sum(x["mathCount"] for x in md_files),
        "tree": tree,
        "files": md_files,
        "assetMap": asset_map,
    }
    _manifest_cache["t"] = now
    _manifest_cache["data"] = data
    return data


def resolve_safe_vault_path(rel_path: str):
    if not rel_path:
        return None
    rel_path = rel_path.lstrip("/").replace("\\", "/")
    full_path = os.path.realpath(os.path.join(VAULT_ROOT, rel_path))
    vault_real = os.path.realpath(VAULT_ROOT)
    if not (full_path == vault_real or full_path.startswith(vault_real + os.sep)):
        return None
    if not full_path.lower().endswith((".md", ".markdown")):
        return None
    return full_path


class VaultHTTPRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WORKSPACE_ROOT, **kwargs)

    def end_headers(self):
        # 允许任意来源（包括 file://、预览域名、localhost）跨域调用读写接口
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept, Origin, X-Requested-With")
        self.send_header("Access-Control-Private-Network", "true")
        if self.path.startswith("/api/") or self.path.endswith((".md", ".js", ".css", ".html")):
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def _send_json(self, status_code: int, payload: dict):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/api/manifest", "/index.html/api/manifest"):
            try:
                return self._send_json(200, {"ok": True, "manifest": get_live_manifest()})
            except Exception as e:
                return self._send_json(500, {"ok": False, "error": str(e)})
        if parsed.path in ("/api/read", "/index.html/api/read"):
            qs = parse_qs(parsed.query)
            rel_path = unquote(qs.get("path", [""])[0])
            full_path = resolve_safe_vault_path(rel_path)
            if not full_path or not os.path.isfile(full_path):
                return self._send_json(404, {"ok": False, "error": f"文件不存在或路径非法: {rel_path}"})
            try:
                with open(full_path, "r", encoding="utf-8") as fp:
                    content = fp.read()
                st = os.stat(full_path)
                return self._send_json(200, {
                    "ok": True,
                    "path": rel_path,
                    "content": content,
                    "size": st.st.st_size if hasattr(st, 'st') else st.st_size,
                    "mtime": int(st.st_mtime)
                })
            except Exception as e:
                return self._send_json(500, {"ok": False, "error": str(e)})
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/api/save", "/index.html/api/save"):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length).decode("utf-8")
                body = json.loads(raw_body)
                rel_path = body.get("path", "")
                content = body.get("content", None)
                if content is None or not isinstance(content, str):
                    return self._send_json(400, {"ok": False, "error": "缺少有效的文档内容 content"})
                full_path = resolve_safe_vault_path(rel_path)
                if not full_path or not os.path.isfile(full_path):
                    return self._send_json(404, {"ok": False, "error": f"目标原始 md 文件不存在: {rel_path}"})

                with open(full_path, "w", encoding="utf-8") as fp:
                    fp.write(content)
                st = os.stat(full_path)
                return self._send_json(200, {
                    "ok": True,
                    "path": rel_path,
                    "fullPath": full_path,
                    "size": st.st_size,
                    "mtime": int(st.st_mtime),
                    "savedAt": time.strftime("%H:%M:%S")
                })
            except Exception as e:
                return self._send_json(500, {"ok": False, "error": str(e)})

        if parsed.path in ("/api/reveal", "/index.html/api/reveal"):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                rel_path = body.get("path", "")
                if rel_path:
                    full_path = resolve_safe_vault_path(rel_path)
                    if not full_path or not os.path.isfile(full_path):
                        return self._send_json(404, {"ok": False, "error": f"文件不存在或路径非法: {rel_path}"})
                    cmd = ["open", "-R", full_path]
                else:
                    full_path = os.path.realpath(VAULT_ROOT)
                    cmd = ["open", full_path]
                if sys.platform != "darwin":
                    return self._send_json(501, {"ok": False, "error": "当前服务不是运行在 macOS 上，无法打开 Finder"})
                subprocess.Popen(cmd)
                return self._send_json(200, {"ok": True, "opened": full_path})
            except Exception as e:
                return self._send_json(500, {"ok": False, "error": str(e)})

        return self._send_json(404, {"ok": False, "error": "未知接口"})


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), VaultHTTPRequestHandler)
    print(f"Obsidian Vault Server listening on http://127.0.0.1:{PORT}", flush=True)
    server.serve_forever()
