from __future__ import annotations
from pathlib import Path
from typing import Iterator, Tuple

import pandas as pd
from pandas.errors import ParserError

from .helpers import to_bool


class CsvManager:
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.df: pd.DataFrame | None = None

    # ================= LOAD / SAVE =================
    def load(self):
        print(f"Loading CSV: {self.csv_path}")
        try:
            self.df = pd.read_csv(self.csv_path, dtype=str, keep_default_na=False)
        except ParserError:
            self.df = pd.read_csv(
                self.csv_path, dtype=str, keep_default_na=False, on_bad_lines="skip"
            )

        rename_map = {}
        cols = self.df.columns

        if "img_checked" in cols: rename_map["img_checked"] = "img_check"
        if "html_checked" in cols: rename_map["html_checked"] = "html_check"
        if "img_error_msg" in cols: rename_map["img_error_msg"] = "img_error"
        if "html_error_msg" in cols: rename_map["html_error_msg"] = "html_error"

        if rename_map:
            self.df.rename(columns=rename_map, inplace=True)

        required = [
            "user","url"
        ]
        for col in required:
            if col not in self.df.columns:
                self.df[col] = ""

        print(f"Loaded {len(self.df)} rows.")

    def save(self):
        if self.df is not None:
            self.df.to_csv(self.csv_path, index=False)
            print(f"CSV updated: {self.csv_path}")

    # ================= ITER =================
    def iter_rows(self) -> Iterator[Tuple[int, pd.Series]]:
        assert self.df is not None
        for idx, row in self.df.iterrows():
            yield idx, row

    # ================= FLAGS =================
    def is_img_done(self, row): return to_bool(row.get("img_check", ""))
    def is_html_done(self, row): return to_bool(row.get("html_check", ""))

    def get_raw_url(self, row):
        m = str(row.get("mirror_link", "")).strip()
        return m if m else str(row.get("url", "")).strip()

    # ================= REPAIR FILE COLUMNS =================
    def repair_file_columns(self, img_dir: Path, html_dir: Path, start_index: int = 0):
        """
        - Nếu flag TRUE mà file không tồn tại → set flag FALSE + file=None
        - Nếu cột chứa path dài → rút lại còn filename
        - Chỉ repair từ start_index trở đi
        """
        assert self.df is not None
        df = self.df

        for idx, row in df.iterrows():
            if idx < start_index:  # Bỏ qua các row trước start_index
                continue

            # ---------- IMG ----------
            img_file = str(row.get("img_file", "")).strip()
            if img_file:
                fname = Path(img_file).name
                if (img_dir / fname).exists():
                    df.at[idx, "img_file"] = fname
                else:
                    if to_bool(row.get("img_check","")):
                        df.at[idx, "img_check"] = "FALSE"
                    df.at[idx, "img_file"] = "None"
            else:
                if to_bool(row.get("img_check","")):
                    df.at[idx, "img_check"] = "FALSE"
                df.at[idx, "img_file"] = "None"

            # ---------- HTML ----------
            html_file = str(row.get("html_file", "")).strip()
            if html_file:
                fname = Path(html_file).name
                if (html_dir / fname).exists():
                    df.at[idx, "html_file"] = fname
                else:
                    if to_bool(row.get("html_check","")):
                        df.at[idx, "html_check"] = "FALSE"
                    df.at[idx, "html_file"] = "None"
            else:
                if to_bool(row.get("html_check","")):
                    df.at[idx, "html_check"] = "FALSE"
                df.at[idx, "html_file"] = "None"

    # ================= SETTERS =================
    def set_img_error(self, idx, msg):
        df = self.df
        df.at[idx,"img_check"]="FALSE"
        df.at[idx,"img_error"]=msg or ""
        if not str(df.at[idx,"img_file"]).strip():
            df.at[idx,"img_file"]="None"

    def set_html_error(self, idx, msg):
        df = self.df
        df.at[idx,"html_check"]="FALSE"
        df.at[idx,"html_error"]=msg or ""
        if not str(df.at[idx,"html_file"]).strip():
            df.at[idx,"html_file"]="None"

    def set_img_success(self, idx, filename):
        df = self.df
        df.at[idx,"img_check"]="TRUE"
        df.at[idx,"img_error"]=""
        df.at[idx,"img_file"]=filename if filename else "None"

    def set_html_success(self, idx, filename):
        df = self.df
        df.at[idx,"html_check"]="TRUE"
        df.at[idx,"html_error"]=""
        df.at[idx,"html_file"]=filename if filename else "None"

    def set_attachment_txt(self, idx, filename, note="TXT only"):
        df = self.df
        df.at[idx,"img_check"]="TRUE"
        df.at[idx,"html_check"]="TRUE"
        df.at[idx,"img_error"]=note
        df.at[idx,"html_error"]=note
        df.at[idx,"img_file"]="None"
        df.at[idx,"html_file"]=filename
