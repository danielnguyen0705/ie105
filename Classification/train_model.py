import os
import re
import pickle
import logging
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)

from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from urllib.parse import urlparse
from scipy.stats import entropy

# ✅ tránh conflict AdamW (torch vs transformers)
from torch.optim import AdamW

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
LABELED_DATA = "D:\\0_Daniel\\HK5\\IE105\\3_DoAn\\labeled_dataset.csv"
DATASET_PATH = "D:\\0_Daniel\\HK5\\IE105\\3_DoAn\\dataset.csv"  # Path to dataset.csv
HTML_DIR = "D:\\0_Daniel\\HK5\\IE105\\3_DoAn\\Extract\\dataset\\html"
IMG_DIR = "D:\\0_Daniel\\HK5\\IE105\\3_DoAn\\Extract\\dataset\\img"
MODEL_SAVE_DIR = "D:\\0_Daniel\\HK5\\IE105\\3_DoAn\\Classification\\models"

# file csv frequent filename (đặt cùng thư mục với train_model.py cho dễ)
SYSTEM_FILENAME_CSV = os.path.join(os.path.dirname(__file__), "filename_appear_3plus.csv")

# Create model directories
os.makedirs(os.path.join(MODEL_SAVE_DIR, "html_model"), exist_ok=True)
os.makedirs(os.path.join(MODEL_SAVE_DIR, "img_model"), exist_ok=True)
os.makedirs(os.path.join(MODEL_SAVE_DIR, "domain_model"), exist_ok=True)

# Helpers
def safe_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Split có stratify nếu đủ điều kiện, không đủ thì fallback split thường.
    Mục tiêu: không crash khi data ít / lệch class.
    """
    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    except Exception as e:
        logger.warning(f"⚠️ Stratified split failed ({e}). Fallback to non-stratified split.")
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=None)

def compute_metrics(y_true, y_pred, model_name, extra=None):
    d = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else np.nan,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)) if len(y_true) else np.nan,
        "recall": float(recall_score(y_true, y_pred, zero_division=0)) if len(y_true) else np.nan,
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)) if len(y_true) else np.nan,
    }
    if extra:
        d.update(extra)
    return d

metrics_summary = []

# Load labeled data
logger.info(f"Loading labeled data from {LABELED_DATA}")
df = pd.read_csv(LABELED_DATA)

# Load additional dataset with columns like img_file, html_file
logger.info(f"Loading additional dataset from {DATASET_PATH}")
dataset_df = pd.read_csv(DATASET_PATH)

# Merge labeled data with dataset.csv to get additional columns
df = pd.merge(df, dataset_df[['index', 'img_file', 'html_file', 'url']], on='index', how='left')
logger.info(f"Found {len(df)} labeled items with additional data")

# Ensure that 'url' column exists after merge
if 'url' not in df.columns:
    logger.error("The 'url' column is missing from the merged dataset.")
    raise SystemExit(1)

# Filter only labeled items
df = df[df['label'] != ''].reset_index(drop=True)
logger.info(f"Found {len(df)} labeled items")

if len(df) < 10:
    logger.error("Not enough labeled data. Please label at least 10 items.")
    raise SystemExit(1)

# Map labels to integers
label_map = {'defaced': 1, 'not defaced': 0, 'uncertain': -1}
df['label_int'] = df['label'].map(label_map)

# Filter out uncertain labels for training
df_train = df[df['label_int'] >= 0].reset_index(drop=True)
logger.info(f"Using {len(df_train)} items for training (excluded uncertain)")

if len(df_train) < 5:
    logger.error("Not enough confident labels. Please label more items.")
    raise SystemExit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# ==================== TRAIN HTML MODEL (BERT) ====================
logger.info("\n=== Training HTML Model (BERT) ===")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
html_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
html_model.to(device)

# ✅ (tuỳ chọn) nếu chạy CPU thì freeze backbone cho nhanh
if device.type == "cpu":
    for p in html_model.bert.parameters():
        p.requires_grad = False
    logger.info("CPU detected: freezing BERT backbone (train classifier only) to speed up.")

# ✅ FAST scan: load danh sách file html 1 lần (siêu nhanh)
try:
    available_html = set(os.listdir(HTML_DIR))
except Exception as e:
    logger.error(f"Cannot list HTML_DIR={HTML_DIR} ({e})")
    raise SystemExit(1)

# Prepare HTML data (NO WARNING SPAM)
html_texts = []
html_labels = []

missing_html = 0
read_error = 0
scanned = 0

for _, row in df_train.iterrows():
    scanned += 1
    html_file = row['html_file']
    if pd.isna(html_file):
        continue

    html_file = str(html_file)
    if html_file not in available_html:
        missing_html += 1
        continue

    html_path = os.path.join(HTML_DIR, html_file)
    try:
        with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().lower()[:2000]  # đọc nhiều hơn 512 chút cũng ok
        html_texts.append(content)
        html_labels.append(int(row['label_int']))
    except Exception:
        read_error += 1
        continue

    if scanned % 1000 == 0:
        logger.info(f"[HTML scan] scanned={scanned}/{len(df_train)} | loaded={len(html_texts)} | missing={missing_html} | error={read_error}")

logger.info(f"[HTML scan done] scanned={scanned} | loaded={len(html_texts)} | missing={missing_html} | error={read_error}")

if len(html_texts) > 0 and len(set(html_labels)) >= 2:
    logger.info(f"Training on {len(html_texts)} HTML samples")

    # ✅ split train/val (safe)
    X_tr, X_val, y_tr, y_val = safe_train_test_split(html_texts, html_labels, test_size=0.2, random_state=42)

    def make_bert_loader(texts, labels, batch_size=8, shuffle=False):
        # ✅ max_length giảm xuống để chạy nhanh hơn
        enc = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
        # ✅ FIX labels dtype long (không còn BCEWithLogits issue)
        y = torch.tensor(labels, dtype=torch.long)
        ds = TensorDataset(enc['input_ids'], enc['attention_mask'], y)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_bert_loader(X_tr, y_tr, batch_size=8, shuffle=True)
    val_loader = make_bert_loader(X_val, y_val, batch_size=8, shuffle=False)

    optimizer = AdamW(filter(lambda p: p.requires_grad, html_model.parameters()), lr=2e-5)

    max_epochs = 10
    total_steps = len(train_loader) * max_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    patience = 2
    best_val_loss = float("inf")
    wait = 0

    best_path = os.path.join(MODEL_SAVE_DIR, "html_model", "bert_model.pt")

    for epoch in range(max_epochs):
        html_model.train()
        train_loss = 0.0

        for batch_idx, (input_ids, attention_mask, labels_batch) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            outputs = html_model(input_ids, attention_mask=attention_mask, labels=labels_batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / max(1, len(train_loader))

        # ===== VALIDATION =====
        html_model.eval()
        val_loss = 0.0
        preds, trues = [], []

        with torch.no_grad():
            for input_ids, attention_mask, labels_batch in val_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels_batch = labels_batch.to(device)

                outputs = html_model(input_ids, attention_mask=attention_mask, labels=labels_batch)
                val_loss += outputs.loss.item()

                logits = outputs.logits
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
                trues.extend(labels_batch.cpu().numpy().tolist())

        avg_val_loss = val_loss / max(1, len(val_loader))

        logger.info(
            f"[BERT] Epoch {epoch+1}/{max_epochs} | train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f}"
        )

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            wait = 0
            torch.save(html_model.state_dict(), best_path)
        else:
            wait += 1
            if wait >= patience:
                logger.info("⏹️ [BERT] Early stopping triggered")
                break

    # load best and compute final val metrics
    html_model.load_state_dict(torch.load(best_path, map_location=device))
    html_model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels_batch in val_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels_batch = labels_batch.to(device)
            outputs = html_model(input_ids, attention_mask=attention_mask)
            preds.extend(outputs.logits.argmax(dim=1).cpu().numpy().tolist())
            trues.extend(labels_batch.cpu().numpy().tolist())

    metrics_summary.append(
        compute_metrics(trues, preds, "BERT_HTML", extra={"early_stop": 1, "best_val_loss": best_val_loss})
    )
    logger.info(f"✅ Saved HTML model to {best_path}")
else:
    logger.warning("⚠️ Not enough valid HTML samples or only 1 class -> skipping BERT training/metrics")
    metrics_summary.append(
        {"model": "BERT_HTML", "accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1_score": np.nan,
         "early_stop": np.nan, "best_val_loss": np.nan}
    )

# ==================== TRAIN IMAGE MODEL (ResNet50) ====================
logger.info("\n=== Training Image Model (ResNet50) ===")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

resnet_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 2)
resnet_model.to(device)

# ✅ FAST scan: load danh sách file img 1 lần
try:
    available_img = set(os.listdir(IMG_DIR))
except Exception as e:
    logger.error(f"Cannot list IMG_DIR={IMG_DIR} ({e})")
    raise SystemExit(1)

# Prepare image data (NO WARNING SPAM)
img_tensors = []
img_labels = []

missing_img = 0
read_error_img = 0
scanned = 0

for _, row in df_train.iterrows():
    scanned += 1
    img_file = row['img_file']
    if pd.isna(img_file):
        continue

    img_file = str(img_file)
    if img_file not in available_img:
        missing_img += 1
        continue

    img_path = os.path.join(IMG_DIR, img_file)
    try:
        image = Image.open(img_path).convert("RGB")
        img_tensor = transform(image)
        img_tensors.append(img_tensor)
        img_labels.append(int(row['label_int']))
    except Exception:
        read_error_img += 1
        continue

    if scanned % 1000 == 0:
        logger.info(f"[IMG scan] scanned={scanned}/{len(df_train)} | loaded={len(img_tensors)} | missing={missing_img} | error={read_error_img}")

logger.info(f"[IMG scan done] scanned={scanned} | loaded={len(img_tensors)} | missing={missing_img} | error={read_error_img}")

if len(img_tensors) > 0 and len(set(img_labels)) >= 2:
    logger.info(f"Training on {len(img_tensors)} image samples")

    # ✅ split train/val (safe)
    X_tr, X_val, y_tr, y_val = safe_train_test_split(img_tensors, img_labels, test_size=0.2, random_state=42)

    def make_img_loader(tensors_list, labels_list, batch_size=8, shuffle=False):
        X = torch.stack(tensors_list)
        y = torch.tensor(labels_list, dtype=torch.long)
        ds = TensorDataset(X, y)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_img_loader(X_tr, y_tr, batch_size=8, shuffle=True)
    val_loader = make_img_loader(X_val, y_val, batch_size=8, shuffle=False)

    optimizer = torch.optim.Adam(resnet_model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    max_epochs = 10
    patience = 2
    best_val_loss = float("inf")
    wait = 0

    best_path = os.path.join(MODEL_SAVE_DIR, "img_model", "resnet_model.pt")

    for epoch in range(max_epochs):
        # TRAIN
        resnet_model.train()
        train_loss = 0.0

        for batch_idx, (images, labels_batch) in enumerate(train_loader):
            images = images.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            outputs = resnet_model(images)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / max(1, len(train_loader))

        # VAL
        resnet_model.eval()
        val_loss = 0.0
        preds, trues = [], []

        with torch.no_grad():
            for images, labels_batch in val_loader:
                images = images.to(device)
                labels_batch = labels_batch.to(device)

                outputs = resnet_model(images)
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item()

                preds.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
                trues.extend(labels_batch.cpu().numpy().tolist())

        avg_val_loss = val_loss / max(1, len(val_loader))

        logger.info(
            f"[CNN] Epoch {epoch+1}/{max_epochs} | train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f}"
        )

        # early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            wait = 0
            torch.save(resnet_model.state_dict(), best_path)
        else:
            wait += 1
            if wait >= patience:
                logger.info("⏹️ [CNN] Early stopping triggered")
                break

    # load best and compute final val metrics
    resnet_model.load_state_dict(torch.load(best_path, map_location=device))
    resnet_model.eval()

    preds, trues = [], []
    with torch.no_grad():
        for images, labels_batch in val_loader:
            images = images.to(device)
            outputs = resnet_model(images)
            preds.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
            trues.extend(labels_batch.cpu().numpy().tolist())

    metrics_summary.append(
        compute_metrics(trues, preds, "CNN_IMAGE", extra={"early_stop": 1, "best_val_loss": best_val_loss})
    )

    logger.info(f"✅ Saved Image model to {best_path}")
else:
    logger.warning("⚠️ Not enough valid image samples or only 1 class -> skipping CNN training/metrics")
    metrics_summary.append(
        {"model": "CNN_IMAGE", "accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1_score": np.nan,
         "early_stop": np.nan, "best_val_loss": np.nan}
    )

# ==================== TRAIN DOMAIN MODEL (Random Forest - URL STRUCTURE) ====================
logger.info("\n=== Training Domain Model (Random Forest - URL Structural Features) ===")

# ===== Load frequent filenames from CSV (LOAD ONCE) =====
if not os.path.exists(SYSTEM_FILENAME_CSV):
    logger.error(f"Missing {SYSTEM_FILENAME_CSV}. Put filename_appear_3plus.csv next to train_model.py")
    raise SystemExit(1)

system_filenames = set(
    pd.read_csv(SYSTEM_FILENAME_CSV)['filename']
    .dropna()
    .astype(str)
    .str.lower()
    .tolist()
)

logger.info(f"Loaded {len(system_filenames)} frequent filenames for system file detection")

# ===== URL feature extraction =====
def extract_url_features(url: str):
    url = str(url).lower()
    parsed = urlparse(url)

    path = parsed.path
    query = parsed.query
    netloc = parsed.netloc
    filename = os.path.basename(path).lower()

    features = {}

    # --- Basic length & structure ---
    features["url_length"] = len(url)
    features["path_length"] = len(path)
    features["num_slash"] = path.count("/")
    features["num_dot"] = path.count(".")
    features["num_dash"] = path.count("-")
    features["num_underscore"] = path.count("_")

    # --- File extension ---
    ext = os.path.splitext(filename)[1]
    features["has_extension"] = int(ext != "")
    features["is_php"] = int(ext == ".php")
    features["is_html"] = int(ext in [".html", ".htm"])
    features["is_txt"] = int(ext == ".txt")
    features["is_ico"] = int(ext == ".ico")
    features["is_json"] = int(ext == ".json")

    # --- Query related ---
    features["has_query"] = int(query != "")
    features["num_equal"] = query.count("=")
    features["num_amp"] = query.count("&")

    # --- Protocol & domain ---
    features["has_https"] = int(parsed.scheme == "https")
    features["has_ip_domain"] = int(bool(re.match(r"\d+\.\d+\.\d+\.\d+", netloc)))

    # --- Suspicious keywords ---
    suspicious_keywords = [
        "hack", "hacked", "deface", "defaced", "shell",
        "admin", "login", "root", "backup", "old"
    ]
    for kw in suspicious_keywords:
        features[f"has_{kw}"] = int(kw in url)

    # --- Asset path ---
    features["is_asset_path"] = int(
        path.startswith(("/_next", "/static", "/assets", "/images", "/css", "/js"))
    )

    # --- System / frequent filename (FROM CSV) ---
    features["is_system_file"] = int(filename in system_filenames)

    # --- Entropy (path randomness) ---
    if len(path) > 0:
        probs = [path.count(c) / len(path) for c in set(path)]
        features["path_entropy"] = float(entropy(probs))
    else:
        features["path_entropy"] = 0.0

    return features

# ===== Extract features from labeled URLs =====
url_features = []
labels = []

for _, row in df_train.iterrows():
    url_features.append(extract_url_features(row["url"]))
    labels.append(int(row["label_int"]))

X = pd.DataFrame(url_features)
y = np.array(labels)

logger.info(f"URL feature matrix shape: {X.shape}")
logger.info(f"Label distribution:\n{pd.Series(y).value_counts()}")

# ===== Split for metrics (safe) =====
X_train, X_val, y_train, y_val = safe_train_test_split(X, y, test_size=0.2, random_state=42)

# ===== Early stopping (OOB) =====
best_oob = -1.0
best_n_estimators = 0
best_model = None

for n_estimators in range(50, 351, 50):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
        oob_score=True,
        bootstrap=True
    )
    rf.fit(X_train, y_train)

    oob = float(rf.oob_score_)
    logger.info(f"[RF] n_estimators={n_estimators}, OOB={oob:.4f}")

    if oob > best_oob:
        best_oob = oob
        best_n_estimators = n_estimators
        best_model = rf
    else:
        logger.info("⏹️ [RF] OOB did not improve -> early stopping")
        break

# ===== Validation metrics =====
rf_preds = best_model.predict(X_val)
metrics_summary.append(
    compute_metrics(y_val.tolist(), rf_preds.tolist(), "RF_URL",
                    extra={"early_stop": 1, "best_n_estimators": best_n_estimators, "oob_score": best_oob})
)

# ===== Retrain final RF on full data (best_n_estimators) =====
final_rf = RandomForestClassifier(
    n_estimators=best_n_estimators,
    max_depth=None,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
final_rf.fit(X, y)

# ===== Save model & feature list =====
domain_model_path = os.path.join(MODEL_SAVE_DIR, "domain_model", "domain_model.pkl")
feature_path = os.path.join(MODEL_SAVE_DIR, "domain_model", "url_features.pkl")

with open(domain_model_path, "wb") as f:
    pickle.dump(final_rf, f)

with open(feature_path, "wb") as f:
    pickle.dump(list(X.columns), f)

logger.info(f"✅ Saved Domain model to {domain_model_path}")
logger.info(f"✅ Saved URL feature list to {feature_path}")

# ==================== SAVE METRICS CSV (ALL MODELS) ====================
metrics_df = pd.DataFrame(metrics_summary)

metrics_csv_path = os.path.join(MODEL_SAVE_DIR, "model_metrics_summary.csv")
metrics_df.to_csv(metrics_csv_path, index=False)

logger.info(f"📊 Saved all metrics to {metrics_csv_path}")
logger.info("✅ Training complete!")
