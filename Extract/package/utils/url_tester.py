import socket, ssl, random
from urllib.parse import urlparse
import requests
from requests.exceptions import Timeout, ConnectionError

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
]

def _headers():
    return {"User-Agent": random.choice(USER_AGENTS)}

def check_dns(domain):
    try:
        socket.gethostbyname(domain)
        return True
    except:
        return False


def test_url(url: str, timeout=3, proxy=None):
    """
    Trả về:
    - alive
    - redirect
    - dns_error
    - timeout
    - dead
    - http_xxx
    """

    if not url.startswith("http"):
        url = "http://" + url

    parsed = urlparse(url)
    if not parsed.hostname or not check_dns(parsed.hostname):
        return {"status": "dns_error", "final_url": url}

    try:
        # Nếu có proxy, sử dụng proxy trong request
        if proxy:
            proxies = {"http": proxy, "https": proxy}
            r = requests.get(url, timeout=timeout, allow_redirects=True, headers=_headers(), proxies=proxies)
        else:
            r = requests.get(url, timeout=timeout, allow_redirects=True, headers=_headers())

        code = r.status_code

        # -------- OK --------
        if 200 <= code < 300:
            return {"status": "alive", "final_url": r.url}

        # -------- Redirect --------
        if 300 <= code < 400:
            return {"status": "redirect", "final_url": r.url}

        # -------- HTTP error --------
        return {"status": f"http_{code}", "final_url": r.url}

    except Timeout:
        return {"status": "timeout", "final_url": url}
    except ConnectionError:
        return {"status": "dead", "final_url": url}
    except ssl.SSLError:
        return {"status": "ssl_error", "final_url": url}
    except Exception:
        return {"status": "dead", "final_url": url}


def classify_short_status(s: str):
    if s == "alive":
        return "OK"
    if s == "redirect":
        return "Redirect"
    if s == "timeout":
        return "Timeout"
    if s == "dns_error":
        return "DNS error"
    if s == "ssl_error":
        return "SSL error"
    if s.startswith("http_"):
        return s.upper()
    return "Dead"
