from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time


class BrowserManager:
    def __init__(self, config=None):
        self.config = config
        self.driver = None

    # =========================
    # CHROME OPTIONS
    # =========================
    def _build_options(self, proxy=None):
        options = Options()

        if proxy:
            options.add_argument(f'--proxy-server={proxy}')

        options.add_argument('--headless=new')
        options.add_argument('--disable-gpu')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')

        # FIX WIDTH = 1920
        options.add_argument('--window-size=1920,1080')

        options.add_argument('--disable-crash-reporter')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-component-extensions-with-background-pages')
        options.add_argument('--disable-background-timer-throttling')
        options.add_argument('--disable-backgrounding-occluded-windows')

        options.add_argument('--log-level=3')
        options.add_experimental_option('excludeSwitches', ['enable-logging'])

        return options

    # =========================
    # START BROWSER
    # =========================
    def start_with_proxy(self, proxy):
        try:
            options = self._build_options(proxy)
            self.driver = webdriver.Chrome(options=options)
        except Exception as e:
            print(f"[WARN] Proxy failed: {e}")
            print("[INFO] Retrying without proxy...")
            options = self._build_options()
            self.driver = webdriver.Chrome(options=options)

    def start(self):
        options = self._build_options()
        self.driver = webdriver.Chrome(options=options)

    # =========================
    # LOAD PAGE
    # =========================
    def get(self, url):
        self.driver.get(url)

        WebDriverWait(self.driver, 30).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        # chờ JS, CSS, font load
        time.sleep(2)

    def wait_for_body(self):
        WebDriverWait(self.driver, 30).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

    # =========================
    # LAZY LOADING HANDLER
    # =========================
    def _force_lazy_load(self, pause_time=0.3, step=600):
        """
        Cuộn từ trên xuống dưới để ép load:
        - ảnh lazy
        - iframe
        - JS dynamic content
        """

        self.driver.execute_script("window.scrollTo(0, 0)")
        time.sleep(1)

        last_height = self.driver.execute_script(
            "return document.body.scrollHeight"
        )

        current_position = 0

        while current_position < last_height:
            self.driver.execute_script(
                f"window.scrollTo(0, {current_position});"
            )
            time.sleep(pause_time)

            current_position += step
            new_height = self.driver.execute_script(
                "return document.body.scrollHeight"
            )

            # nếu trang load thêm nội dung
            if new_height > last_height:
                last_height = new_height

        # đảm bảo đứng lại ở đầu trang
        self.driver.execute_script("window.scrollTo(0, 0)")
        time.sleep(1)

    # =========================
    # FULL PAGE SCREENSHOT
    # =========================
    def full_page_screenshot(self, file_path):
        """
        - Load toàn bộ lazy content
        - Resize viewport = full height
        - Chụp 1 ảnh duy nhất
        """

        # ép load lazy
        self._force_lazy_load()

        # lấy chiều cao thật của trang
        total_height = self.driver.execute_script("""
            return Math.max(
                document.body.scrollHeight,
                document.documentElement.scrollHeight,
                document.body.offsetHeight,
                document.documentElement.offsetHeight,
                document.body.clientHeight,
                document.documentElement.clientHeight
            );
        """)

        # resize viewport
        self.driver.set_window_size(1920, total_height)
        time.sleep(1)

        # chụp
        self.driver.save_screenshot(file_path)
        print(f"[OK] Full page screenshot saved: {file_path}")

    # =========================
    # HTML SOURCE
    # =========================
    def get_html_source(self):
        return self.driver.page_source

    # =========================
    # STOP
    # =========================
    def stop(self):
        if self.driver:
            self.driver.quit()
