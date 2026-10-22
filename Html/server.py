"""Obsidian 知识库本地读写、图片资源处理与静态文件服务 (Port 60051)。"""

import base64
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Optional, Union
from urllib.parse import parse_qs, unquote, urlparse

WORKSPACE_ROOT = Path(__file__).resolve().parent
VAULT_ROOT = Path(os.environ.get("VAULT_ROOT", str(WORKSPACE_ROOT.parent))).resolve()
PORT = int(os.environ.get("PORT", "60051"))

sys.path.insert(0, str(WORKSPACE_ROOT))
import build_manifest as _bm  # noqa: E402

logger = logging.getLogger(__name__)


# -------------------------------------------
# 配置常量
# -------------------------------------------
MANIFEST_CACHE_TTL_SECONDS = 2.0                  # 目录索引缓存有效期（秒）
MAX_IMAGE_UPLOAD_BYTES = 50 * 1024 * 1024         # 单张图片上传允许的最大字节数（50MB）
ALLOWED_MARKDOWN_SUFFIXES = (".md", ".markdown")  # 允许读写的 Markdown 后缀
ALLOWED_IMAGE_SUFFIXES = (                        # 允许读写的图片文件后缀
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".svg",
)

_manifest_cache: dict[str, Any] = {"t": 0.0, "data": None}


# -------------------------------------------
# 索引与路径校验
# -------------------------------------------
def invalidate_manifest_cache() -> None:
    """清空内存中的目录与资源索引缓存。"""
    _manifest_cache["t"] = 0.0
    _manifest_cache["data"] = None


def get_live_manifest() -> dict[str, Any]:
    """获取最新的知识库目录树与图片资源映射表。"""
    now = time.time()
    cached_data = _manifest_cache["data"]
    if cached_data is not None and now - _manifest_cache["t"] < MANIFEST_CACHE_TTL_SECONDS:
        return cached_data

    tree, md_files, asset_map, _docs = _bm.build_tree_and_lists(str(VAULT_ROOT))
    data = {
        "vaultPath": str(VAULT_ROOT),
        "generatedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
        "totalCount": len(md_files),
        "noteCount": sum(1 for item in md_files if not item["isAssetDoc"]),
        "mathTotal": sum(item["mathCount"] for item in md_files),
        "tree": tree,
        "files": md_files,
        "assetMap": asset_map,
    }
    _manifest_cache["t"] = now
    _manifest_cache["data"] = data
    return data


def _is_within_vault(target_path: Path) -> bool:
    try:
        target_path.relative_to(VAULT_ROOT)
        return True
    except ValueError:
        return False


def resolve_safe_vault_path(rel_path: str) -> Optional[Path]:
    """解析并校验知识库内的 Markdown 文件绝对路径。"""
    if not rel_path:
        return None
    normalized = rel_path.lstrip("/").replace("\\", "/")
    full_path = (VAULT_ROOT / normalized).resolve()
    if not _is_within_vault(full_path):
        return None
    if full_path.suffix.lower() not in ALLOWED_MARKDOWN_SUFFIXES:
        return None
    return full_path


def resolve_safe_image_path(rel_path: str) -> Optional[Path]:
    """解析并校验知识库内的图片文件绝对路径。"""
    if not rel_path:
        return None
    normalized = unquote(rel_path).lstrip("/").replace("\\", "/")
    full_path = (VAULT_ROOT / normalized).resolve()
    if not _is_within_vault(full_path):
        return None
    if full_path.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
        return None
    return full_path


# -------------------------------------------
# 图片文件写入与备份
# -------------------------------------------
def decode_base64_image(data_url_or_b64: str) -> tuple[bytes, str]:
    """从 DataURL 或纯 Base64 字符串解码二进制图片内容并推断扩展名。"""
    raw = data_url_or_b64.strip()
    ext = ".png"
    if raw.startswith("data:"):
        header, _, encoded = raw.partition(",")
        mime = header.split(";")[0].replace("data:", "").strip().lower()
        mime_to_ext = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/webp": ".webp",
            "image/gif": ".gif",
            "image/bmp": ".bmp",
            "image/svg+xml": ".svg",
        }
        ext = mime_to_ext.get(mime, ".png")
    else:
        encoded = raw

    binary = base64.b64decode(encoded)
    if len(binary) > MAX_IMAGE_UPLOAD_BYTES:
        raise ValueError(f"图片体积超过限制 ({MAX_IMAGE_UPLOAD_BYTES // (1024 * 1024)}MB)")
    return binary, ext


def atomic_write_bytes(target_path: Path, data: bytes) -> None:
    """通过临时文件原子替换方式写入二进制文件。"""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target_path.parent), prefix=".tmp_img_", suffix=target_path.suffix)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fp:
            fp.write(data)
        tmp_path.replace(target_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def generate_attachment_filename(ext: str) -> str:
    """生成与 Obsidian Custom Attachment Location 插件格式一致的时间戳文件名。"""
    now_ms = int(time.time() * 1000)
    ms_part = f"{now_ms % 1000:03d}"
    time_part = time.strftime("%Y%m%d%H%M%S", time.localtime(now_ms / 1000))
    clean_ext = ext if ext.startswith(".") else f".{ext}"
    return f"file-{time_part}{ms_part}{clean_ext}"


def build_new_image_target(doc_rel_path: str, preferred_ext: str, custom_name: Optional[str] = None) -> tuple[Path, str, str]:
    """根据当前笔记路径计算新图片的保存绝对路径、vault 相对路径及 Markdown 引用链接。"""
    doc_path = resolve_safe_vault_path(doc_rel_path)
    if not doc_path:
        raise ValueError(f"无效的笔记路径: {doc_rel_path}")

    note_stem = doc_path.stem
    note_dir = doc_path.parent
    assets_dir = note_dir / "assets" / note_stem

    if custom_name:
        safe_name = Path(custom_name).name
        if Path(safe_name).suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
            safe_name = f"{safe_name}{preferred_ext}"
    else:
        safe_name = generate_attachment_filename(preferred_ext)

    target_path = (assets_dir / safe_name).resolve()
    if not _is_within_vault(target_path):
        raise ValueError("目标图片路径越界")

    # 避免覆盖已有同名附件
    counter = 1
    stem = target_path.stem
    suffix = target_path.suffix
    while target_path.exists():
        target_path = (assets_dir / f"{stem}-{counter}{suffix}").resolve()
        counter += 1

    vault_rel = target_path.relative_to(VAULT_ROOT).as_posix()
    encoded_note_stem = note_stem.replace(" ", "%20")
    encoded_file_name = target_path.name.replace(" ", "%20")
    markdown_rel_link = f"assets/{encoded_note_stem}/{encoded_file_name}"
    return target_path, vault_rel, markdown_rel_link


# -------------------------------------------
# HTTP 服务处理
# -------------------------------------------
class VaultHTTPRequestHandler(SimpleHTTPRequestHandler):
    """支持 Markdown 与图片原地读写的本地 HTTP 请求处理器。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(WORKSPACE_ROOT), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept, Origin, X-Requested-With")
        self.send_header("Access-Control-Private-Network", "true")
        req_path = self.path.split("?", 1)[0]
        if req_path.startswith("/api/") or req_path.endswith((".md", ".js", ".css", ".html", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg")):
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.end_headers()

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw_body = self.rfile.read(length).decode("utf-8")
        return json.loads(raw_body) if raw_body else {}

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/api/manifest", "/index.html/api/manifest"):
            try:
                self._send_json(200, {"ok": True, "manifest": get_live_manifest()})
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": str(exc)})
            return

        if parsed.path in ("/api/read", "/index.html/api/read"):
            qs = parse_qs(parsed.query)
            rel_path = unquote(qs.get("path", [""])[0])
            full_path = resolve_safe_vault_path(rel_path)
            if not full_path or not full_path.is_file():
                self._send_json(404, {"ok": False, "error": f"文件不存在或路径非法: {rel_path}"})
                return
            try:
                content = full_path.read_text(encoding="utf-8")
                st = full_path.stat()
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "path": rel_path,
                        "content": content,
                        "size": st.st_size,
                        "mtime": int(st.st_mtime),
                    },
                )
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": str(exc)})
            return

        if parsed.path in ("/api/image/read", "/index.html/api/image/read"):
            qs = parse_qs(parsed.query)
            rel_path = unquote(qs.get("path", [""])[0])
            full_path = resolve_safe_image_path(rel_path)
            if not full_path or not full_path.is_file():
                self._send_json(404, {"ok": False, "error": f"图片不存在或路径非法: {rel_path}"})
                return
            try:
                raw_bytes = full_path.read_bytes()
                ext = full_path.suffix.lower()
                mime_map = {
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".webp": "image/webp",
                    ".gif": "image/gif",
                    ".bmp": "image/bmp",
                    ".svg": "image/svg+xml",
                }
                mime = mime_map.get(ext, "image/png")
                b64_str = base64.b64encode(raw_bytes).decode("ascii")
                self._send_json(
                    200,
                    {
                        "ok": True,
                        "path": full_path.relative_to(VAULT_ROOT).as_posix(),
                        "dataUrl": f"data:{mime};base64,{b64_str}",
                        "size": len(raw_bytes),
                    },
                )
            except Exception as exc:
                self._send_json(500, {"ok": False, "error": str(exc)})
            return

        super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/api/save", "/index.html/api/save"):
            self._handle_save_markdown()
            return
        if parsed.path in ("/api/image/overwrite", "/index.html/api/image/overwrite"):
            self._handle_overwrite_image()
            return
        if parsed.path in ("/api/image/upload", "/index.html/api/image/upload"):
            self._handle_upload_image()
            return
        if parsed.path in ("/api/reveal", "/index.html/api/reveal"):
            self._handle_reveal_finder()
            return
        self._send_json(404, {"ok": False, "error": "未知接口"})

    def _handle_save_markdown(self) -> None:
        try:
            body = self._read_json_body()
            rel_path = body.get("path", "")
            content = body.get("content", None)
            if content is None or not isinstance(content, str):
                self._send_json(400, {"ok": False, "error": "缺少有效的文档内容 content"})
                return
            full_path = resolve_safe_vault_path(rel_path)
            if not full_path or not full_path.is_file():
                self._send_json(404, {"ok": False, "error": f"目标原始 md 文件不存在: {rel_path}"})
                return

            atomic_write_bytes(full_path, content.encode("utf-8"))
            st = full_path.stat()
            self._send_json(
                200,
                {
                    "ok": True,
                    "path": rel_path,
                    "fullPath": str(full_path),
                    "size": st.st_size,
                    "mtime": int(st.st_mtime),
                    "savedAt": time.strftime("%H:%M:%S"),
                },
            )
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": str(exc)})

    def _handle_overwrite_image(self) -> None:
        """原地覆盖知识库中已有的图片文件（用于原地裁剪或替换图片），返回原图备份 Base64 以便撤销。"""
        try:
            body = self._read_json_body()
            rel_path = body.get("path", "")
            image_data = body.get("imageData", "")
            if not rel_path or not image_data:
                self._send_json(400, {"ok": False, "error": "缺少图片路径 path 或 imageData"})
                return

            full_path = resolve_safe_image_path(rel_path)
            if not full_path or not full_path.is_file():
                self._send_json(404, {"ok": False, "error": f"目标图片文件不存在: {rel_path}"})
                return

            old_bytes = full_path.read_bytes()
            old_ext = full_path.suffix.lower()
            mime_map = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".gif": "image/gif",
                ".bmp": "image/bmp",
                ".svg": "image/svg+xml",
            }
            old_mime = mime_map.get(old_ext, "image/png")
            previous_data_url = f"data:{old_mime};base64,{base64.b64encode(old_bytes).decode('ascii')}"

            new_bytes, _ = decode_base64_image(image_data)
            atomic_write_bytes(full_path, new_bytes)
            st = full_path.stat()
            invalidate_manifest_cache()

            self._send_json(
                200,
                {
                    "ok": True,
                    "path": full_path.relative_to(VAULT_ROOT).as_posix(),
                    "fullPath": str(full_path),
                    "size": st.st_size,
                    "mtime": int(st.st_mtime),
                    "savedAt": time.strftime("%H:%M:%S"),
                    "previousDataUrl": previous_data_url,
                },
            )
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": str(exc)})

    def _handle_upload_image(self) -> None:
        """将新图片写入当前笔记对应的 assets/<noteFileName>/ 文件夹，并返回相对引用路径。"""
        try:
            body = self._read_json_body()
            doc_path = body.get("docPath", "")
            image_data = body.get("imageData", "")
            custom_name = body.get("fileName", None)
            if not doc_path or not image_data:
                self._send_json(400, {"ok": False, "error": "缺少笔记路径 docPath 或图片内容 imageData"})
                return

            new_bytes, preferred_ext = decode_base64_image(image_data)
            target_path, vault_rel, markdown_link = build_new_image_target(doc_path, preferred_ext, custom_name)
            atomic_write_bytes(target_path, new_bytes)
            st = target_path.stat()
            invalidate_manifest_cache()

            self._send_json(
                200,
                {
                    "ok": True,
                    "vaultPath": vault_rel,
                    "fullPath": str(target_path),
                    "markdownLink": markdown_link,
                    "fileName": target_path.name,
                    "size": st.st_size,
                    "mtime": int(st.st_mtime),
                    "savedAt": time.strftime("%H:%M:%S"),
                },
            )
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": str(exc)})

    def _handle_reveal_finder(self) -> None:
        try:
            body = self._read_json_body()
            rel_path = body.get("path", "")
            target: Optional[Union[Path, str]] = None
            if rel_path:
                target = resolve_safe_vault_path(rel_path) or resolve_safe_image_path(rel_path)
                if not target or not target.exists():
                    self._send_json(404, {"ok": False, "error": f"文件不存在或路径非法: {rel_path}"})
                    return
                cmd = ["open", "-R", str(target)]
            else:
                target = VAULT_ROOT
                cmd = ["open", str(target)]
            if sys.platform != "darwin":
                self._send_json(501, {"ok": False, "error": "当前服务不是运行在 macOS 上，无法打开 Finder"})
                return
            subprocess.Popen(cmd)
            self._send_json(200, {"ok": True, "opened": str(target)})
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": str(exc)})


# -------------------------------------------
# 命令行入口
# -------------------------------------------
def main() -> int:
    """启动本地知识库读写与静态服务。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-6s | %(name)-12s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        server = ThreadingHTTPServer(("0.0.0.0", PORT), VaultHTTPRequestHandler)
        logger.info("Obsidian Vault Server listening on http://127.0.0.1:%s", PORT)
        server.serve_forever()
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception:
        logger.exception("Server exited with unexpected error")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
