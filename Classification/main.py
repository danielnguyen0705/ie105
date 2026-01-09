import os
import pandas as pd
import shutil
import logging

# Cấu hình logging
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(logs_dir, "classification.log"), 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Đường dẫn gốc
base_path = "D:/0_Daniel/HK5/IE105/3_DoAn"
dataset_path = os.path.join(base_path, "dataset.csv")
labeled_dataset_path = os.path.join(base_path, "auto_labeled_dataset.csv")
source_data_dir = os.path.join(base_path, "Extract/dataset")

def update_labels():
    try:
        # 1. Đọc dữ liệu và ép kiểu index về string để đồng nhất
        df = pd.read_csv(dataset_path, dtype={'index': str})
        df_labeled = pd.read_csv(labeled_dataset_path, dtype={'index': str})

        logging.info(f"Đã đọc {len(df)} dòng từ dataset và {len(df_labeled)} dòng từ labeled_dataset")

        # 2. Kết hợp dữ liệu
        df = pd.merge(df, df_labeled[['index', 'label']], on="index", how="left")
        
        # Điền các giá trị nhãn trống là 'uncertain'
        df['label'] = df['label'].fillna('uncertain')

        # 3. Tạo thư mục output
        output_base_dir = "output"
        categories = ['defaced', 'not defaced', 'uncertain']
        for cat in categories:
            os.makedirs(os.path.join(output_base_dir, cat), exist_ok=True)

        # 4. Duyệt qua từng dòng để copy
        count_success = 0
        for _, row in df.iterrows():
            idx = str(row['index'])
            label = row['label']
            
            # Xác định thư mục đích
            dest_cat_dir = os.path.join(output_base_dir, label if label in categories else 'uncertain')
            
            # Tạo cấu trúc thư mục con: output/nhãn/index/html và output/nhãn/index/img
            folder_path = os.path.join(dest_cat_dir, idx)
            html_folder = os.path.join(folder_path, "html")
            img_folder = os.path.join(folder_path, "img")
            
            # Đường dẫn file nguồn
            html_src = os.path.join(source_data_dir, "html", str(row['html_file']))
            img_src = os.path.join(source_data_dir, "img", str(row['img_file']))

            # Kiểm tra file nguồn có tồn tại không trước khi copy
            if os.path.exists(html_src) and os.path.exists(img_src):
                os.makedirs(html_folder, exist_ok=True)
                os.makedirs(img_folder, exist_ok=True)
                
                shutil.copy(html_src, os.path.join(html_folder, str(row['html_file'])))
                shutil.copy(img_src, os.path.join(img_folder, str(row['img_file'])))
                
                count_success += 1
                if count_success % 10 == 0:
                    print(f"Đang xử lý... Đã copy xong {count_success} bộ file.")
            else:
                logging.warning(f"Không tìm thấy file cho index {idx}: {html_src} hoặc {img_src}")

        # 5. Lưu kết quả
        df.to_csv('output/labeled_data.csv', index=False)
        logging.info(f"Hoàn thành! Đã copy thành công {count_success}/{len(df)} bộ file.")
        print(f"Hoàn thành! Kiểm tra file log trong thư mục '{logs_dir}' để biết chi tiết.")

    except Exception as e:
        logging.error(f"Lỗi hệ thống: {e}")
        print(f"Đã xảy ra lỗi: {e}")

if __name__ == "__main__":
    update_labels()