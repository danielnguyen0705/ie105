import streamlit as st
import pandas as pd
import os
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# =========================
# Configuration
# =========================
DATASET_PATH = "D:\\0_Daniel\\HK5\\IE105\\3_DoAn\\dataset.csv"
HTML_DIR = "D:\\0_Daniel\\HK5\\IE105\\3_DoAn\\Extract\\dataset\\html"
IMG_DIR = "D:\\0_Daniel\\HK5\\IE105\\3_DoAn\\Extract\\dataset\\img"
LABELS_OUTPUT = "D:\\0_Daniel\\HK5\\IE105\\3_DoAn\\labeled_dataset.csv"

VALID_LABELS = {"defaced", "not defaced", "uncertain"}

# =========================
# Page setup
# =========================
st.set_page_config(page_title="Website Defacement Labeler", layout="wide")
st.title("🏷️ Website Defacement Labeler")

# =========================
# Session state init
# =========================
if "item_idx" not in st.session_state:
    st.session_state.item_idx = 0

# =========================
# Load dataset
# =========================
@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_PATH)

df = load_dataset()

# =========================
# Load / init labels
# =========================
if os.path.exists(LABELS_OUTPUT):
    labels_df = pd.read_csv(LABELS_OUTPUT)
else:
    labels_df = pd.DataFrame({
        "index": df["index"],
        "label": ["" for _ in range(len(df))]
    })

# Chuẩn hoá label
labels_df["label"] = labels_df["label"].astype(str).str.strip()

# =========================
# Sidebar statistics (FIXED)
# =========================
st.sidebar.header("📊 Statistics")
total = len(df)
labeled = labels_df["label"].isin(VALID_LABELS).sum()
unlabeled = total - labeled

st.sidebar.metric("Total", total)
st.sidebar.metric("Labeled", labeled)
st.sidebar.metric("Unlabeled", unlabeled)

# =========================
# Sidebar navigation (editable)
# =========================
new_idx = st.sidebar.number_input(
    "Item index",
    min_value=0,
    max_value=len(df) - 1,
    value=st.session_state.item_idx,
    step=1
)

if new_idx != st.session_state.item_idx:
    st.session_state.item_idx = new_idx
    st.rerun()

# =========================
# Current item
# =========================
idx = st.session_state.item_idx
current_row = df.iloc[idx]
current_label_row = labels_df.iloc[idx]

st.header(f"Item #{idx} — {current_row['url']}")

# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs(["HTML Content", "Image Preview", "Metadata"])

with tab1:
    html_file = current_row["html_file"] if isinstance(current_row["html_file"], str) else ""
    html_path = os.path.join(HTML_DIR, html_file)

    if html_file and os.path.exists(html_path):
        with open(html_path, encoding="utf-8", errors="ignore") as f:
            content = f.read()
        st.text_area(
            "HTML Preview (first 2000 chars)",
            content[:2000],
            height=400,
            disabled=True
        )
    else:
        st.warning("HTML file not found")

with tab2:
    img_file = current_row["img_file"] if isinstance(current_row["img_file"], str) else ""
    img_path = os.path.join(IMG_DIR, img_file)

    if img_file and os.path.exists(img_path):
        try:
            with Image.open(img_path) as img:
                img.thumbnail((1200, 1200))
                st.image(img)
        except Image.DecompressionBombError:
            st.warning(
                "⚠️ Image is extremely large (possible decompression bomb). "
                "Preview disabled for safety."
            )
        except Exception as e:
            st.error(f"Error loading image: {e}")
    else:
        st.warning("Image not found")

with tab3:
    st.json(current_row.to_dict())

# =========================
# Labeling
# =========================
st.divider()
st.subheader("🎨 Label")

label_options = ["", "defaced", "not defaced", "uncertain"]
current_label = current_label_row["label"] if current_label_row["label"] in VALID_LABELS else ""

label = st.radio(
    "Is this website defaced?",
    label_options,
    index=label_options.index(current_label),
    horizontal=True
)

# =========================
# Save logic
# =========================
def save_and_next():
    labels_df.at[idx, "label"] = label
    labels_df.to_csv(LABELS_OUTPUT, index=False)
    if idx < len(df) - 1:
        st.session_state.item_idx += 1

# =========================
# Navigation buttons
# =========================
st.divider()
c1, c2, c3 = st.columns(3)

with c1:
    if st.button("⬅️ Previous"):
        if idx > 0:
            st.session_state.item_idx -= 1
        st.rerun()

with c2:
    if st.button("✅ Save"):
        save_and_next()
        st.rerun()

with c3:
    if st.button("➡️ Next"):
        if idx < len(df) - 1:
            st.session_state.item_idx += 1
        st.rerun()

# =========================
# Status
# =========================
if current_label:
    st.success(f"Current label: {current_label}")
else:
    st.info("Not labeled yet")

st.caption(f"Progress: {labeled}/{total} ({(labeled/total)*100:.2f}%)")
