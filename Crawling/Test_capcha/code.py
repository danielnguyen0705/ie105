import time
import random
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import ddddocr

# --- CẤU HÌNH ---
START_PAGE = 38
END_PAGE = 38
OUTPUT_FILE = 'zone_h_full3.csv'

# Khởi tạo ddddocr
ocr = ddddocr.DdddOcr()

def init_driver():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless") 
    options.add_argument("--disable-blink-features=AutomationControlled") 
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(options=options)
    return driver

def solve_captcha_if_present(driver):
    """Hàm kiểm tra và giải captcha nếu xuất hiện"""
    try:
        # Kiểm tra captcha không
        # Dấu hiệu: Không tìm thấy bảng dữ liệu VÀ tìm thấy thẻ img captcha
        # Note: ta cần Inspect Element trên Zone-H khi bị dính captcha để lấy ID chính xác.
        
        # Giả sử thẻ img captcha có src chứa "captcha" hoặc nằm trong form
        captcha_input = driver.find_elements(By.NAME, "captcha")
        
        if len(captcha_input) > 0:
            print("\n⚠️ PHÁT HIỆN CAPTCHA! Đang tiến hành xử lý...")
            
            # 1. Tìm ảnh captcha
            # Thường là thẻ img nằm gần input captcha
            img_element = driver.find_element(By.XPATH, "//img[contains(@src, 'captcha')]")
            
            # 2. Chụp ảnh captcha
            img_bytes = img_element.screenshot_as_png
            
            # 3. Giải mã
            res = ocr.classification(img_bytes)
            print(f"-> Đã đọc được: {res}")
            
            # 4. Điền và submit
            captcha_input[0].clear()
            captcha_input[0].send_keys(res)
            
            # Tìm nút submit
            submit_btn = driver.find_element(By.XPATH, "//input[@type='submit']")
            submit_btn.click()
            
            print("-> Đã gửi CAPTCHA. Đang chờ reload...")
            time.sleep(5) # Chờ load lại trang
            return True # Đã xử lý captcha
            
    except Exception as e:
        # Không phải captcha hoặc lỗi tìm element, bỏ qua
        pass
    return False

# --- CHƯƠNG TRÌNH CHÍNH ---
driver = init_driver()
all_data = []

print(f"--- Bắt đầu lấy dữ liệu (Chế độ Selenium + Auto Captcha) ---")

try:
    for page in range(START_PAGE, END_PAGE + 1):
        url = f"https://www.zone-h.org/archive/published=0/page={page}"
        print(f"Đang tải trang {page}...", end=" ")
        
        driver.get(url)
        
        # Kiểm tra có bị dính captcha không
        if solve_captcha_if_present(driver):
            pass

        time.sleep(2) 
        
        # Lấy source HTML bằng BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        table = soup.find('table', {'id': 'ldeface'})
        
        if table:
            rows = table.find_all('tr')
            count = 0
            for row in rows[1:]:
                cols = row.find_all('td')
                if len(cols) >= 9:
                    mirror_link = ""
                    try:
                        mirror_tag = cols[9].find('a')
                        if mirror_tag:
                            mirror_link = "https://www.zone-h.org" + mirror_tag['href']
                    except: pass

                    data_row = {
                        'Date': cols[0].get_text(strip=True),
                        'Notifier': cols[1].get_text(strip=True),
                        'Flag_H': cols[2].get_text(strip=True), 
                        'Flag_M': cols[3].get_text(strip=True), 
                        'Flag_R': cols[4].get_text(strip=True), 
                        'Flag_L': cols[5].get_text(strip=True),
                        'Domain': cols[7].get_text(strip=True), 
                        'OS': cols[8].get_text(strip=True),
                        'Mirror_Link': mirror_link, 
                        'Page': page
                    }
                    all_data.append(data_row)
                    count += 1
            print(f"-> OK! ({count} dòng)")
        else:
            print("-> KHÔNG THẤY DỮ LIỆU (Có thể CAPTCHA giải sai hoặc IP bị chặn).")
            
        # Random sleep 
        sleep_time = random.uniform(3, 7)
        time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\nĐã dừng thủ công!")
finally:
    # Lưu file
    if all_data:
        df = pd.DataFrame(all_data)
        cols_order = ['Date', 'Notifier', 'Domain', 'OS', 'Flag_H', 'Flag_M', 'Flag_R', 'Flag_L', 'Mirror_Link', 'Page']
        # Lọc cột nếu tồn tại
        df = df[[c for c in cols_order if c in df.columns]]
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"\n✅ Đã lưu {len(all_data)} dòng vào: {OUTPUT_FILE}")
    
    driver.quit()