from __future__ import annotations
from pathlib import Path 
from ..utils.csv_manager import CsvManager
from ..browser.browser_manager import BrowserManager
from ..utils.helpers import shorten_error


def process_html_for_row(
    csv_mgr: CsvManager,
    browser: BrowserManager,
    row_index: int,
    stable_idx: int,
    html_output_dir: Path
):
    try:
        html_text = browser.get_html_source()
        html_filename = f"html_data_{stable_idx}.txt"

        # Lưu nội dung HTML vào file
        out_path = html_output_dir / html_filename
        out_path.write_text(html_text, encoding="utf-8", errors="ignore")

        # Ghi vào cột html_file trong CSV
        csv_mgr.set_html_success(row_index, html_filename)

        print(f"HTML extraction success: {html_filename}")

    except Exception as e:
        csv_mgr.set_html_error(row_index, shorten_error(str(e)))
        try:
            print(f"HTML extraction error (row {row_index}): {str(e)}")
        except UnicodeEncodeError:
            print(f"HTML extraction error (row {row_index}): [encoding error - see log]")
