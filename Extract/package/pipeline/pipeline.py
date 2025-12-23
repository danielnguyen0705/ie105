from __future__ import annotations
import random  # Đảm bảo đã import random
from pathlib import Path
import requests  # Thêm thư viện requests để kiểm tra kết nối trực tiếp

from ..config.config import PipelineConfig
from ..utils.helpers import normalize_url, shorten_error
from ..utils.csv_manager import CsvManager
from ..browser.browser_manager import BrowserManager
from ..processors.img_processor import setup_tesseract, process_img_for_row
from ..processors.html_processor import process_html_for_row
from ..utils.url_tester import test_url, classify_short_status

CHECKPOINT_INTERVAL = 50
# Retry limits (configurable)
# Reduced defaults for faster fail-fast behavior during extraction
PROXY_RETRY_LIMIT = 10
BROWSER_RETRY_LIMIT = 3
LOG_FILE = Path(__file__).parent / "pipeline.log"


class DefaceCombinedPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.csv_mgr = CsvManager(config.csv_path)
        self.browser = BrowserManager(config)
        self.random_proxy = None  # Bỏ proxy
        # ensure log file exists
        try:
            LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            LOG_FILE.write_text("", encoding="utf-8")
        except Exception:
            pass

    def _log(self, msg: str):
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass

    # ==========================================================
    def run(self, start_index: int = 0, max_workers: int = 20, defer_ocr: bool = False, max_rows: int | None = None):  
        try:
            setup_tesseract(self.config.tesseract_cmd)

            # Kiểm tra kết nối trực tiếp từ máy (không dùng proxy)
            print(f"Trying direct connection from machine (no proxy)...")
            if self._test_connection():  # Kiểm tra kết nối mà không dùng proxy
                print("Connection successful using the direct connection.")
            else:
                print("Direct connection failed.")
                return  # Nếu kết nối trực tiếp thất bại, dừng quá trình

            # Nếu kết nối trực tiếp thành công, sử dụng trình duyệt mà không cần proxy
            self.browser.start()

            # Load CSV
            self.csv_mgr.load()
            self.csv_mgr.repair_file_columns(self.config.img_output_dir, self.config.html_output_dir)

            self._ensure_output_dirs()

            # Thay vì gọi _precheck_urls_parallel, bạn có thể gọi một phương thức kiểm tra đơn giản ở đây.
            self._process_all_rows(start_index, defer_ocr=defer_ocr, max_rows=max_rows)  

        finally:
            self.browser.stop()
            self.csv_mgr.save()

    # ==========================================================
    def _test_connection(self):
        """Kiểm tra kết nối với một URL mẫu (không sử dụng proxy)"""
        test_url = "http://www.google.com"
        try:
            response = requests.get(test_url, timeout=5)  # Kiểm tra kết nối
            if response.status_code == 200:
                return True
            else:
                return False
        except requests.RequestException:
            return False

    # ==========================================================
    def _is_browser_error(self, html: str) -> bool:
        """Detect common Chrome/Selenium interstitial or network error pages in HTML."""
        if not html:
            return True
        s = html.lower()
        indicators = [
            "this page isn\'t working",
            "this site can\u2019t be reached",
            "this site can\'t be reached",
            "err_",
            "net::",
            "main-frame-error",
            "your connection was interrupted",
            "dns error",
            "server error",
            "unable to connect",
            "temporarily unavailable",
        ]
        for kw in indicators:
            if kw in s:
                return True
        return False

    # ==========================================================
    def _ensure_output_dirs(self):
        # Tạo thư mục img và html nếu chưa tồn tại
        self.config.img_output_dir.mkdir(parents=True, exist_ok=True)
        self.config.html_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"IMG dir:  {self.config.img_output_dir}")
        print(f"HTML dir: {self.config.html_output_dir}")

    # ==========================================================
    def _process_all_rows(self, start_index: int, defer_ocr: bool = False, max_rows: int | None = None):
        df = self.csv_mgr.df
        assert df is not None
        total = len(df)

        processed_count = 0

        # Duyệt qua từ start_index
        for row_index, row in self.csv_mgr.iter_rows():
            if row_index < start_index:
                continue  # Bỏ qua các dòng trước start_index

            percent = (row_index + 1) / total * 100
            print(f"\n[{row_index + 1}/{total} – {percent:.2f}%] Processing...")

            # -------- Determine if IMG or HTML are done --------
            img_done = self.csv_mgr.is_img_done(row)
            html_done = self.csv_mgr.is_html_done(row)

            # Normalize URL early so it's available for later checks
            raw_url = normalize_url(str(row.get("url", "")).strip())  # Chuẩn hóa URL

            img_file = str(row.get("img_file", "")).strip()
            html_file = str(row.get("html_file", "")).strip()

            # Also check file existence on disk to avoid re-extraction
            img_path = None
            html_path = None
            try:
                if img_file and img_file.lower() != "none":
                    img_path = Path(self.config.img_output_dir) / img_file
                if html_file and html_file.lower() != "none":
                    html_path = Path(self.config.html_output_dir) / html_file
            except Exception:
                img_path = None
                html_path = None

            img_exists = img_done and img_file.lower() != "none" and (img_path is None or img_path.exists())
            html_exists = html_done and html_file.lower() != "none" and (html_path is None or html_path.exists())

            # Skip if both are OK
            if img_exists and html_exists:
                continue

            # -------- Check for img_check or html_check == FALSE --------
            if not img_done:
                if raw_url:
                    # Kiểm tra nếu URL có thể truy cập được
                    check_tmp = test_url(raw_url, timeout=3, proxy=None)  # Không sử dụng proxy
                    if check_tmp["status"] == "alive":
                        # Đặt tên file ảnh vào cột img_file
                        img_filename = f"img_data_{row_index + 1}.png"
                        self.csv_mgr.set_img_success(row_index, img_filename)

            if not html_done:
                # Check HTML và gắn tên file vào cột html_file
                try:
                    html_filename = f"html_data_{row_index + 1}.txt"
                    self.csv_mgr.set_html_success(row_index, html_filename)
                except Exception as e:
                    self.csv_mgr.set_html_error(row_index, shorten_error(str(e)))

            # ======================================================
            # 1) URL TEST (with retry)
            # ======================================================
            # If precheck said OK, avoid another network request and treat as alive
            pre = ""
            if "precheck_ok" in self.csv_mgr.df.columns:
                pre = str(self.csv_mgr.df.at[row_index, "precheck_ok"]).upper()
            
            if pre == "TRUE":
                check = {"status": "alive", "final_url": raw_url}
                status = "alive"
            else:
                check = test_url(raw_url, timeout=3, proxy=None)  # Không sử dụng proxy
                status = check["status"]
            
            # Nếu URL fail, thử lại
            retry_count = 0
            while status not in ("alive", "redirect") and retry_count < PROXY_RETRY_LIMIT:
                check = test_url(raw_url, timeout=3, proxy=None)  # Không sử dụng proxy
                status = check["status"]
                retry_count += 1
            
            short_status = classify_short_status(status)

            # ======================================================
            # CASE: DEAD (http_404, timeout, dns_error...)
            # ======================================================
            if status not in ("alive", "redirect"):
                if not img_exists:
                    self.csv_mgr.set_img_error(row_index, short_status)
                if not html_exists:
                    self.csv_mgr.set_html_error(row_index, short_status)
                continue

            # Page is alive
            final_url = check["final_url"]

            # Disabled headless if needed
            if not check.get("use_headless", True):
                self.browser.disable_headless_temporarily()

            # ======================================================
            # 2) Selenium load page
            # ======================================================
            try:
                self.browser.get(final_url)
                self.browser.wait_for_body()

                # Kiểm tra trang lỗi của trình duyệt (Chrome interstitial/net error)
                page_html = self.browser.get_html_source()
                browser_retry = 0
                # Nếu phát hiện trang lỗi, thử tải lại
                while self._is_browser_error(page_html) and browser_retry < BROWSER_RETRY_LIMIT:
                    try:
                        self.browser.stop()
                    except Exception:
                        pass
                    self.browser.start()  # Sử dụng kết nối trực tiếp nếu không có proxy
                    self.browser.get(final_url)
                    self.browser.wait_for_body()
                    page_html = self.browser.get_html_source()
                    browser_retry += 1

                if self._is_browser_error(page_html):
                    # Vẫn lỗi sau retry → ghi lỗi rõ lý do vào CSV
                    reason = "Browser net error or interstitial page"
                    self._log(f"Persistent browser error for row {row_index+1}: {reason}")
                    if not img_exists:
                        self.csv_mgr.set_img_error(row_index, reason)
                    if not html_exists:
                        self.csv_mgr.set_html_error(row_index, reason)
                    continue

            except Exception as e:
                err = shorten_error(str(e))
                if not img_exists:
                    self.csv_mgr.set_img_error(row_index, err)
                if not html_exists:
                    self.csv_mgr.set_html_error(row_index, err)
                continue

            # ======================================================
            # 3) IMG extraction
            # ======================================================
            if not img_exists:
                process_img_for_row(
                    self.csv_mgr,
                    self.browser,
                    row_index,
                    row_index + 1,
                    self.config.img_output_dir,
                    do_ocr=(not defer_ocr),
                )

            # ======================================================
            # 4) HTML → ALWAYS .txt
            # ======================================================
            if not html_exists:
                try:
                    html_text = self.browser.get_html_source()
                    html_filename = f"html_data_{row_index + 1}.txt"

                    out_path = self.config.html_output_dir / html_filename
                    out_path.write_text(html_text, encoding="utf-8", errors="ignore")

                    self.csv_mgr.set_html_success(row_index, html_filename)

                except Exception as e:
                    self.csv_mgr.set_html_error(row_index, shorten_error(str(e)))

            # ======================================================
            # 5) CHECKPOINT
            # ======================================================
            if (row_index + 1) % CHECKPOINT_INTERVAL == 0:
                print(f"💾 Checkpoint at row {row_index + 1}")
                self.csv_mgr.save()

            # Count processed rows and optionally stop early for tests
            processed_count += 1
            if max_rows is not None and processed_count >= max_rows:
                self._log(f"Reached max_rows={max_rows}, stopping early.")
                return
