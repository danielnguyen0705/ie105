import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

# --- CẤU HÌNH ---
START_PAGE = 1
END_PAGE = 414
OUTPUT_FILE = 'hackdb.csv'
BASE_URL = "https://hack-db.org"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    # Update lại Cookie mới từ trình duyệt nếu chạy bị lỗi 403/Login
    'Cookie': 'PHPSESSID=8cbihcgsoup2or9n0o1vb1j090' 
}

all_data = []

print(f"--- Bắt đầu lấy dữ liệu từ {START_PAGE} đến {END_PAGE} ---")

for page in range(START_PAGE, END_PAGE + 1):
    url = f"{BASE_URL}/archive/{page}"
    
    try:
        print(f"Đang tải trang {page}...", end=" ")
        response = requests.get(url, headers=HEADERS, timeout=20)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # --- SỬA ĐỔI 1: Tìm bảng bằng class thay vì ID ---
            # Trong HTML bạn gửi, table có class là 'table table-sm ...'
            table = soup.find('table', class_='table')
            
            if table:
                rows = table.find_all('tr')
                count = 0
                
                # Bỏ qua row header
                for row in rows[1:]:
                    cols = row.find_all('td')
                    
                    # --- SỬA ĐỔI 2: Cấu trúc cột mới dựa trên HTML ---
                    # 0: ID | 1: User | 2: Country | 3: Web URL | 4: IP | 5: Date | 6: View
                    if len(cols) >= 7:
                        
                        # Xử lý lấy Link Mirror (nằm ở cột cuối cùng - View)
                        mirror_link = ""
                        try:
                            view_tag = cols[6].find('a')
                            if view_tag and 'href' in view_tag.attrs:
                                # Link trong HTML là dạng relative (mirror/14147) nên cần nối thêm domain
                                mirror_path = view_tag['href']
                                if mirror_path.startswith("http"):
                                    mirror_link = mirror_path
                                else:
                                    mirror_link = f"{BASE_URL}/{mirror_path.lstrip('/')}"
                        except:
                            pass

                        # Lấy URL bị hack (Cột 3)
                        hacked_url = cols[3].get_text(strip=True)
                        
                        # Lấy IP (Cột 4)
                        ip_address = cols[4].get_text(strip=True)

                        data_row = {
                            'ID': cols[0].get_text(strip=True),
                            'Notifier': cols[1].get_text(strip=True),
                            'Country': cols[2].get_text(strip=True),
                            'Hacked_URL': hacked_url,
                            'IP': ip_address,
                            'Date': cols[5].get_text(strip=True),
                            'Mirror_Link': mirror_link,
                            'Page': page
                        }
                        all_data.append(data_row)
                        count += 1
                
                print(f"-> OK! ({count} dòng)")
            else:
                print("-> THẤT BẠI (Không tìm thấy bảng dữ liệu).")
                # Có thể in ra HTML để debug nếu cần: print(response.text[:500])
        else:
            print(f"-> Lỗi server: {response.status_code}")
            
    except Exception as e:
        print(f"-> Lỗi kết nối: {e}")

    # Delay để tránh bị chặn IP
    sleep_time = random.uniform(3, 6)
    time.sleep(sleep_time)

# Xuất file CSV
if all_data:
    df = pd.DataFrame(all_data)
    # Sắp xếp lại thứ tự cột cho đẹp
    cols_order = ['ID', 'Date', 'Notifier', 'Country', 'IP', 'Hacked_URL', 'Mirror_Link', 'Page']
    df = df[cols_order]
    
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n✅ Hoàn tất! Đã lấy {len(df)} dòng. File lưu tại: {OUTPUT_FILE}")
else:
    print("\n❌ Không lấy được dữ liệu. Hãy kiểm tra lại Cookie hoặc cấu trúc web!")