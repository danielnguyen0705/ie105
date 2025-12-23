from package.pipeline.pipeline import DefaceCombinedPipeline
from package.config.config import PipelineConfig
from pathlib import Path

def main():
    # Đảm bảo cập nhật đường dẫn đúng tới file CSV
    csv_path = Path("D:/0_Daniel/HK5/IE105/3_DoAn/dataset.csv")  # Cập nhật lại đường dẫn đúng file CSV

    # Cấu hình của pipeline (cập nhật các đường dẫn đầu ra)
    config = PipelineConfig(
        csv_path=csv_path,
        img_output_dir=Path("D:/0_Daniel/HK5/IE105/3_DoAn/Extract/dataset/img"),
        html_output_dir=Path("D:/0_Daniel/HK5/IE105/3_DoAn/Extract/dataset/html"),
        tesseract_cmd=Path("C:/Program Files/Tesseract-OCR/tesseract.exe")  # Đường dẫn tới tesseract
    )

    # Tạo các thư mục đầu ra nếu chưa tồn tại
    config.img_output_dir.mkdir(parents=True, exist_ok=True)
    config.html_output_dir.mkdir(parents=True, exist_ok=True)

    # Khởi tạo pipeline và chạy
    pipeline = DefaceCombinedPipeline(config)
    # Bắt đầu từ dòng đầu tiên (index=0) và bật 10 workers để chạy song song
    pipeline.run(start_index=21000, max_workers=10, defer_ocr=True)


if __name__ == "__main__":
    main()
