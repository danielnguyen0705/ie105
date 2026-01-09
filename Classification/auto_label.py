import os
import pandas as pd
import torch
import pickle
from transformers import BertTokenizer, BertForSequenceClassification
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
DATASET = "D:\\0_Daniel\\HK5\\IE105\\3_DoAn\\dataset.csv"  # Path to dataset.csv containing necessary columns
LABELED_DATA = "D:\\0_Daniel\\HK5\\IE105\\3_DoAn\\labeled_dataset.csv"  # Path to labeled data
HTML_DIR = "D:\\0_Daniel\\HK5\\IE105\\3_DoAn\\Extract\\dataset\\html"
IMG_DIR = "D:\\0_Daniel\\HK5\\IE105\\3_DoAn\\Extract\\dataset\\img"
MODEL_DIR = "D:\\0_Daniel\\HK5\\IE105\\3_DoAn\\Classification\\models"
OUTPUT = "D:\\0_Daniel\\HK5\\IE105\\3_DoAn\\auto_labeled_dataset.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load dataset
logger.info(f"Loading dataset from {DATASET}")
df = pd.read_csv(DATASET)

# Load labeled data to get already-labeled items
if os.path.exists(LABELED_DATA):
    labeled_df = pd.read_csv(LABELED_DATA)
    labeled_indices = set(labeled_df[labeled_df['label'] != '']['index'].values)
    logger.info(f"Found {len(labeled_indices)} already labeled items, will skip these")
else:
    labeled_indices = set()

# Add label column if not exists
if 'label' not in df.columns:
    df['label'] = ''

# Merge the labeled dataset to get additional columns (like html_file, img_file, url)
df = pd.merge(df, labeled_df[['index', 'label']], on='index', how='left')
logger.info(f"Found {len(df)} items after merging with labeled dataset")

# Ensure the 'label' column exists and is populated
if 'label' not in df.columns:
    df['label'] = ''  # Initialize label if it's missing

# Load models
logger.info("Loading trained models...")

# HTML Model
try:
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    html_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    html_model_path = os.path.join(MODEL_DIR, "html_model", "bert_model.pt")
    if os.path.exists(html_model_path):
        html_model.load_state_dict(torch.load(html_model_path, map_location=device))
        logger.info("✅ Loaded HTML model")
        html_model_available = True
    else:
        logger.warning("⚠️ HTML model not found, will use pre-trained")
        html_model_available = False
    html_model.to(device)
    html_model.eval()
except Exception as e:
    logger.warning(f"Error loading HTML model: {e}")
    html_model_available = False

# Image Model
try:
    transform = transforms.Compose([ 
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    img_model.fc = torch.nn.Linear(img_model.fc.in_features, 2)
    img_model_path = os.path.join(MODEL_DIR, "img_model", "resnet_model.pt")
    if os.path.exists(img_model_path):
        img_model.load_state_dict(torch.load(img_model_path, map_location=device))
        logger.info("✅ Loaded Image model")
        img_model_available = True
    else:
        logger.warning("⚠️ Image model not found, will use pre-trained")
        img_model_available = False
    img_model.to(device)
    img_model.eval()
except Exception as e:
    logger.warning(f"Error loading Image model: {e}")
    img_model_available = False

# Domain Model
try:
    domain_model_path = os.path.join(MODEL_DIR, "domain_model", "domain_model.pkl")
    vectorizer_path = os.path.join(MODEL_DIR, "domain_model", "vectorizer.pkl")
    
    if os.path.exists(domain_model_path) and os.path.exists(vectorizer_path):
        with open(domain_model_path, 'rb') as f:
            domain_model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        logger.info("✅ Loaded Domain model")
        domain_model_available = True
    else:
        logger.warning("⚠️ Domain model not found")
        domain_model_available = False
except Exception as e:
    logger.warning(f"Error loading Domain model: {e}")
    domain_model_available = False

# Inference functions
def classify_html(html_file):
    """Classify HTML file using BERT"""
    try:
        if pd.isna(html_file) or not html_model_available:
            return None
        
        html_path = os.path.join(HTML_DIR, html_file)
        if not os.path.exists(html_path):
            return None
        
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read().lower()[:512]
        
        inputs = tokenizer(content, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = html_model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).item()
        
        return pred
    except Exception as e:
        logger.debug(f"Error classifying HTML {html_file}: {e}")
        return None

def classify_image(img_file):
    """Classify image file using ResNet50"""
    try:
        if pd.isna(img_file) or not img_model_available:
            return None
        
        img_path = os.path.join(IMG_DIR, img_file)
        if not os.path.exists(img_path):
            return None
        
        image = Image.open(img_path).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = img_model(img_tensor)
            pred = torch.argmax(outputs, dim=-1).item()
        
        return pred
    except Exception as e:
        logger.debug(f"Error classifying image {img_file}: {e}")
        return None

def classify_domain(url):
    """Classify domain using Random Forest"""
    try:
        if pd.isna(url) or not domain_model_available:
            return None
        
        X = vectorizer.transform([str(url).lower()])
        pred = domain_model.predict(X)[0]
        
        return pred
    except Exception as e:
        logger.debug(f"Error classifying domain {url}: {e}")
        return None

# Classify remaining items
logger.info(f"\nStarting auto-labeling of {len(df)} items...")
total = len(df)
labeled_count = 0
skipped_count = 0

for idx, row in df.iterrows():
    # For testing, force labeling even for already labeled items
    # if row['index'] in labeled_indices:
    #     skipped_count += 1
    #     continue
    
    # Ensemble voting: take majority vote from 3 models
    votes = []
    
    # Log the inputs to check if we're getting the right data
    logger.debug(f"Classifying HTML for {row['html_file']}")
    html_pred = classify_html(row['html_file'])
    if html_pred is not None:
        logger.debug(f"HTML Prediction: {html_pred}")
        votes.append(html_pred)
    
    logger.debug(f"Classifying Image for {row['img_file']}")
    img_pred = classify_image(row['img_file'])
    if img_pred is not None:
        logger.debug(f"Image Prediction: {img_pred}")
        votes.append(img_pred)
    
    logger.debug(f"Classifying Domain for {row['url']}")
    domain_pred = classify_domain(row['url'])
    if domain_pred is not None:
        logger.debug(f"Domain Prediction: {domain_pred}")
        votes.append(domain_pred)
    
    # Determine label from votes
    if len(votes) == 0:
        label = ""
    else:
        # If any model says defaced (1), label as defaced
        if 1 in votes:
            label = "defaced"
        else:
            label = "not defaced"
    
    df.at[idx, 'label'] = label
    labeled_count += 1
    
    if (idx + 1) % 100 == 0:
        logger.info(f"Progress: {idx + 1}/{total} items processed")

# Save results with 'url' next to 'index'
df = df[['index', 'url', 'label', 'img_file', 'html_file']]  # Ensure all relevant columns are included
df.to_csv(OUTPUT, index=False)
logger.info(f"\n✅ Auto-labeling complete!")
logger.info(f"Labeled: {labeled_count} new items")
logger.info(f"Skipped: {skipped_count} already labeled items")
logger.info(f"Results saved to {OUTPUT}")
logger.info(f"\nNext steps:")
logger.info(f"1. Review the results manually (check 'uncertain' or low-confidence predictions)")
logger.info(f"2. Use labeling_tool.py to refine labels if needed)")
logger.info(f"3. Run train_model.py again to fine-tune models")
