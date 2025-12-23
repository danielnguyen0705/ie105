from pathlib import Path
import requests
from bs4 import BeautifulSoup
import random
import re

# Lấy đường dẫn file proxies.txt trong thư mục utils
PROXY_FILE = Path(__file__).parent / "proxies.txt"

def _is_valid_proxy(proxy_str: str) -> bool:
    """Kiểm tra xem chuỗi có phải là proxy hợp lệ (IP:port) không"""
    # Pattern: xxx.xxx.xxx.xxx:port
    pattern = r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}:\d{1,5}$'
    return bool(re.match(pattern, proxy_str))

# Lấy danh sách proxy từ Free Proxy List
def get_proxy_list_from_website(url: str):
    proxies = []
    try:
        response = requests.get(url, timeout=3)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Tìm tất cả proxy trong trang web (dựa trên cấu trúc HTML của trang)
        rows = soup.find_all("tr")
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 2:
                ip = cols[0].get_text(strip=True)
                port = cols[1].get_text(strip=True)
                proxy = f"{ip}:{port}"
                # Kiểm tra xem proxy có hợp lệ không (phải là IP:port)
                if _is_valid_proxy(proxy):
                    proxies.append(proxy)
        
        print(f"Found {len(proxies)} valid proxies from {url}")
        return proxies
    except Exception as e:
        print(f"Error fetching proxies from {url}: {e}")
        return []

# Lưu proxy vào file
def save_proxies_to_file(proxies, filename=None):
    if filename is None:
        filename = PROXY_FILE
    try:
        with open(filename, "w") as file:
            for proxy in proxies:
                file.write(proxy + "\n")
        print(f"Proxies saved to {filename}")
    except Exception as e:
        print(f"Error saving proxies: {e}")

# Đọc proxy từ file
def load_proxies_from_file(filename=None):
    if filename is None:
        filename = PROXY_FILE
    try:
        with open(filename, "r") as file:
            proxies = file.readlines()
        return [proxy.strip() for proxy in proxies]
    except Exception as e:
        print(f"Error loading proxies from file: {e}")
        return []

# Lựa chọn ngẫu nhiên n proxy từ danh sách
def get_random_proxies(proxy_list, count=10):
    return random.sample(proxy_list, min(count, len(proxy_list)))

# Tự động lấy proxy từ các trang web và lưu vào file
def fetch_and_save_proxies():
    # URL của các trang cung cấp proxy miễn phí
    proxy_sources = [
        'https://www.free-proxy-list.net/',  # Free proxy list (website)
        'https://www.sslproxies.org/',  # SSL Proxy list (website)
    ]
    
    all_proxies = []
    for source in proxy_sources:
        proxies = get_proxy_list_from_website(source)
        all_proxies.extend(proxies)

    # Lưu tất cả proxy vào file
    save_proxies_to_file(all_proxies)

    return all_proxies
