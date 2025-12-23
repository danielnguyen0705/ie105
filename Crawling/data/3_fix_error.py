#Nếu run code get mirror còn vài rows chưa get được và bị bỏ qua thì chạy file này!
import pandas as pd

FILE_NAME = 'zone_h_2_1.csv'

try:
    df = pd.read_csv(FILE_NAME)
    
    # Đếm số dòng có status:  check = True NHƯNG Domain vẫn chứa "..."
    # Tức là đã dính capcha ở row mirror đó.
    mask = (df['Full_Url_Checked'] == True) & (df['Domain'].str.contains('\.\.\.', na=False))
    
    error_count = mask.sum()
    
    if error_count > 0:
        print(f"⚠️ Tìm thấy {error_count} dòng bị lỗi (đánh dấu xong nhưng chưa lấy được domain).")
        
        # Reset trạng thái về False để script get_mirror_links.py chạy lại
        df.loc[mask, 'Full_Url_Checked'] = False
        
        df.to_csv(FILE_NAME, index=False, encoding='utf-8-sig')
        print(f"✅ Fixed")
    else:
        print("✅ File clean, không lỗi dòng nào hết!")

except Exception as e:
    print(f"Lỗi: {e}")