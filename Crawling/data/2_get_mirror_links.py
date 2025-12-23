import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import random
import os

# --- CẤU HÌNH ---
INPUT_FILE = 'zone_h_2.csv'    
OUTPUT_FILE = 'zone_h_2_1.csv' 

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Cookie': 'PHPSESSID=dt73fr00tq85vd2l92kupsf1l4; ZHE=0823cc383d4455bb4384c2856a7dc5cb;'
}

# Đọc dữ liệu
if os.path.exists(OUTPUT_FILE):
    print(f"🔄 Tìm thấy file đang chạy dở: {OUTPUT_FILE}. Tiếp tục chạy...")
    df = pd.read_csv(OUTPUT_FILE)
else:
    print(f"📂 Đọc file gốc: {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    # Tạo cột đánh dấu đã check chưa
    if 'Full_Url_Checked' not in df.columns:
        df['Full_Url_Checked'] = False

total_rows = len(df)
count_success = 0

print(f"Bắt đầu xử lý {total_rows} dòng dữ liệu...")

for index, row in df.iterrows():
    # Điều kiện: Domain bị cắt (...) VÀ có link Mirror VÀ chưa check xong
    domain_val = str(row['Domain'])
    mirror_link = str(row['Mirror_Link'])
    is_checked = row['Full_Url_Checked']

    if '...' in domain_val and mirror_link != 'nan' and not is_checked:
        
        print(f"[{index}/{total_rows}] Đang lấy mirror: {mirror_link} ...", end=" ")
        
        try:
            # Gửi request
            resp = requests.get(mirror_link, headers=HEADERS, timeout=30)
            
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, 'html.parser')
                
                # Tìm thẻ <li class="defaces">
                target_li = soup.find('li', class_='defaces')
                
                if target_li:
                    # Lấy text: "Domain: https://..."
                    raw_text = target_li.get_text(strip=True)
                    # Xóa chữ "Domain:"
                    full_url = raw_text.replace('Domain:', '').strip()
                    
                    # Cập nhật vào DataFrame
                    df.at[index, 'Domain'] = full_url 
                    df.at[index, 'Full_Url_Checked'] = True
                    count_success += 1
                    print(f"✅ OK: {full_url}")
                else:
                    print("⚠️ Không tìm thấy class 'defaces' trong HTML.")
                    # Đánh dấu đã check để không lặp lại, dù lỗi
                    df.at[index, 'Full_Url_Checked'] = True 
            
            elif resp.status_code in [403, 503]:
                print("\nBỊ CHẶN (CAPTCHA/WAF)! Dừng script ngay.")
                break 
            else:
                print(f"Lỗi HTTP {resp.status_code}")
                
        except Exception as e:
            print(f"Lỗi Exception: {e}")

        # --- LƯU LIÊN TỤC (Checkpoint) ---
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

        # --- Nghỉ ---
        sleep_time = random.uniform(10, 20) 
        time.sleep(sleep_time)

    else:
        pass

print(f"\nHoàn tất phiên làm việc. Đã cập nhật {count_success} dòng.")
print(f"File final: {OUTPUT_FILE}")