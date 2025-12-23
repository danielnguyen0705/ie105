# package/config/config.py

from pathlib import Path

class PipelineConfig:
    def __init__(self, csv_path: str, tesseract_cmd: str, img_output_dir: str, html_output_dir: str):
        self.csv_path = csv_path
        self.tesseract_cmd = tesseract_cmd
        self.img_output_dir = img_output_dir
        self.html_output_dir = html_output_dir
