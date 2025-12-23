# package/utils/helpers.py
from __future__ import annotations
import os
import re


def normalize_url(url: str) -> str:
    """
    Chuẩn hoá URL:
    - bỏ khoảng trắng hai đầu
    - nếu không có http/https thì mặc định thêm http://
    """
    url = (url or "").strip()
    if not url:
        return ""
    if not re.match(r"^https?://", url, re.IGNORECASE):
        return "http://" + url
    return url


def to_bool(val) -> bool:
    """
    Chuyển mọi kiểu về bool theo style CSV:
    TRUE/true/1/yes/y  => True
    FALSE/false/0/no/n/"" => False
    """
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in ("true", "1", "yes", "y")


def shorten_error(error_msg: str) -> str:
    """
    Rút gọn lỗi để ghi vào CSV (không tràn dòng).
    Các pattern thường gặp đã map sang message ngắn.
    """
    if not error_msg:
        return "Unknown error"

    msg = error_msg.lower()

    if "timeout" in msg or "timed out" in msg:
        return "Timeout"
    if "dns" in msg and ("not found" in msg or "nxdomain" in msg or "name or service not known" in msg):
        return "DNS not found"
    if "connection refused" in msg or "connectionreseterror" in msg or "connection aborted" in msg:
        return "Connection refused"
    if "403" in msg and "forbidden" in msg:
        return "HTTP_403"
    if "404" in msg:
        return "HTTP_404"
    if "500" in msg:
        return "HTTP_500"
    if "ssl" in msg:
        return "SSL error"
    if "offline" in msg:
        return "Browser offline"
    if "proxy" in msg:
        return "Proxy error"

    # fallback chung
    return "Browser error"
