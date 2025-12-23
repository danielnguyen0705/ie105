from __future__ import annotations

from pathlib import Path
import pytesseract
from PIL import Image

from ..utils.csv_manager import CsvManager
from ..browser.browser_manager import BrowserManager
from ..utils.helpers import shorten_error


def setup_tesseract(tesseract_cmd: Path):
    pytesseract.pytesseract.tesseract_cmd = str(tesseract_cmd)


def process_img_for_row(
    csv_mgr: CsvManager,
    browser: BrowserManager,
    row_index: int,
    stable_idx: int,
    img_output_dir: Path,
    do_ocr: bool = True,
):
    # Lưu trực tiếp vào img_output_dir mà không tạo subfolder
    img_path = img_output_dir / f"img_data_{stable_idx}.png"
    txt_path = img_output_dir / f"img_text_{stable_idx}.txt"

    try:
        # Screenshot
        browser.full_page_screenshot(img_path)

        if do_ocr:
            # OCR
            image = Image.open(img_path)
            image = image.convert("L")
            text = pytesseract.image_to_string(image, lang="eng+vie")

            with txt_path.open("w", encoding="utf-8") as f:
                f.write(text)

        # Update img_file with the correct filename (no path)
        img_filename = f"img_data_{stable_idx}.png"
        csv_mgr.set_img_success(row_index, img_filename)

    except Exception as e:
        msg = shorten_error(str(e))
        csv_mgr.set_img_error(row_index, msg)
