import time
import random
import os
import pandas as pd
import numpy as np
import ddddocr
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import cv2

# ==============================================================================
# 1. CẤU HÌNH 
# ==============================================================================
INPUT_FILE = 'zone_h_full1.csv'    
OUTPUT_FILE = 'final_2.csv'
# ==============================================================================
# TIỀN XỬ LÝ HÌNH ẢNH CAPTCHA
# ==============================================================================

def process_image_for_better_accuracy(img_bytes):

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    #Chuyển sang ảnh xám (Grayscale)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Nhị phân hóa (Thresholding)
    # Biến tất cả điểm ảnh mờ thành trắng, chữ đậm thành đen (hoặc ngược lại)
    # Threshold = 140, có thể tùy chỉnh
    _, img_binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY) 

    # Khử nhiễu Denoise
    # Nếu captcha có nhiều chấm li ti thì dùng 
    img_binary = cv2.fastNlMeansDenoising(img_binary, None, 10, 7, 21)

    #Chuyển ngược lại thành bytes => ddddocr
    is_success, buffer = cv2.imencode(".png", img_binary)
    return buffer.tobytes()

# ==============================================================================
# 2. KHỞI TẠO CÔNG CỤ (Selenium + AI)
# ==============================================================================
print("Đang khởi tạo AI ddddocr...")
ocr = ddddocr.DdddOcr(show_ad=False, beta=True)

def setup_driver():
    options = Options()
    options.add_argument("--headless") 
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    options.add_argument('--ignore-certificate-errors')
    return webdriver.Chrome(options=options)

def magic_solve_captcha(driver):
    """Hàm tự động phát hiện và giải CAPTCHA"""
    try:
        # Các dấu hiệu nhận biết ảnh captcha
        img_xpaths = [
            "//img[contains(@src, 'captcha')]", 
            "//img[contains(@id, 'captcha')]",
            "//img[contains(@src, 'SecurityImage')]"
        ]
        
        target_img = None
        for path in img_xpaths:
            elems = driver.find_elements(By.XPATH, path)
            if elems:
                target_img = elems[0]
                break
        
        if target_img:
            # Các dấu hiệu nhận biết ô input
            input_names = ["captcha", "code", "security_code", "captcha_code"]
            target_input = None
            for name in input_names:
                elems = driver.find_elements(By.NAME, name)
                if elems:
                    target_input = elems[0]
                    break
            
            if target_input:
                print("   ⚠️  PHÁT HIỆN CAPTCHA! Đang giải...", end=" ")
                img_bytes = target_img.screenshot_as_png
                res = ocr.classification(img_bytes)
                #clean_bytes = process_image_for_better_accuracy(img_bytes)
                #res = ocr.classification(clean_bytes)
                print(f"-> AI đọc là: {res}")
                
                target_input.clear()
                target_input.send_keys(res)
                
                try:
                    target_input.submit()
                except:
                    driver.find_element(By.XPATH, "//input[@type='submit']").click()
                
                time.sleep(4) # Chờ load lại
                return True
    except Exception as e:
        pass
    return False

# ==============================================================================
# 3. LOGIC TRÍCH XUẤT DỮ LIỆU 
# ==============================================================================
def get_full_domain_from_html(html_source):

    soup = BeautifulSoup(html_source, 'html.parser')
    
    target_li = soup.find('li', class_='defaces')
    
    if target_li:
        raw_text = target_li.get_text(strip=True)
        full_url = raw_text.replace('Domain:', '').strip()
        return full_url
    # -----------------------------------------------------
    return None

# ==============================================================================
# 4. MAIN LOOP
# ==============================================================================
if __name__ == "__main__":
    # --- Đọc dữ liệu ---
    if os.path.exists(OUTPUT_FILE):
        print(f"🔄 Tiếp tục chạy từ file: {OUTPUT_FILE}")
        df = pd.read_csv(OUTPUT_FILE)
    else:
        print(f"📂 Bắt đầu mới từ file: {INPUT_FILE}")
        df = pd.read_csv(INPUT_FILE)
        if 'Full_Url_Checked' not in df.columns:
            df['Full_Url_Checked'] = False

    total_rows = len(df)
    driver = setup_driver()
    
    print(f"Bắt đầu xử lý {total_rows} dòng...")

    try:
        for index, row in df.iterrows():
            # Lấy data 1 row
            domain_val = str(row['Domain'])
            mirror_link = str(row['Mirror_Link'])
            is_checked = row['Full_Url_Checked']

            # Điều kiện để chạy: Domain chứa ... VÀ chưa check = false
            if '...' in domain_val and mirror_link != 'nan' and not is_checked:
                
                print(f"[{index}/{total_rows}] Truy cập: {mirror_link} ...", end=" ")
                
                try:
                    driver.get(mirror_link)
                    
                    # 1. Check xem có Captcha không ?
                    if magic_solve_captcha(driver):
                        pass
                    
                    # 2. Get html để phân tích
                    full_url = get_full_domain_from_html(driver.page_source)
                    
                    if full_url:
                        print(f"✅ OK: {full_url}")
                        df.at[index, 'Domain'] = full_url
                        df.at[index, 'Full_Url_Checked'] = True
                    else:
                        print("⚠️ Không thấy class 'defaces'")
                        # Có thể đánh dấu là True để ko check lại, hoặc False để retry
                        df.at[index, 'Full_Url_Checked'] = False 

                except Exception as e:
                    print(f"❌ Lỗi: {e}")
                
                # Lưu file liên tục (Checkpoint)
                df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
                
                # random tgian nghỉ
                time.sleep(random.uniform(2, 6))
                
    except KeyboardInterrupt:
        print("\n🛑 Đã dừng thủ công!")
    finally:
        driver.quit()
        print("Đã đóng trình duyệt.")