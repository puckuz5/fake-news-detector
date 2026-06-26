import streamlit as st
import joblib
import re
import os
import numpy as np
import requests
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract
import io

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Indian News Verifier",
    page_icon="🔍",
    layout="centered"
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .result-real {
        background: #EAF3DE;
        border-left: 5px solid #3B6D11;
        padding: 16px 20px;
        border-radius: 0 10px 10px 0;
        margin: 12px 0;
    }
    .result-fake {
        background: #FCEBEB;
        border-left: 5px solid #A32D2D;
        padding: 16px 20px;
        border-radius: 0 10px 10px 0;
        margin: 12px 0;
    }
    .result-title {
        font-size: 22px;
        font-weight: 500;
        margin-bottom: 4px;
    }
    .result-conf {
        font-size: 14px;
        opacity: 0.8;
    }
    .word-fake { background: #FCEBEB; color: #A32D2D; padding: 2px 6px; border-radius: 4px; margin: 2px; display: inline-block; font-size: 13px; }
    .word-real { background: #EAF3DE; color: #3B6D11; padding: 2px 6px; border-radius: 4px; margin: 2px; display: inline-block; font-size: 13px; }
    .stat-box { background: var(--background-color); border: 0.5px solid rgba(0,0,0,0.1); border-radius: 8px; padding: 12px 16px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# LOAD MODEL — cached so it only loads once
# ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    model      = joblib.load("model/model.pkl")
    vectorizer = joblib.load("model/vectorizer.pkl")
    info       = joblib.load("model/model_info.pkl")
    return model, vectorizer, info

try:
    model, vectorizer, model_info = load_model()
    model_loaded = True
except:
    model_loaded = False

# ──────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict(text):
    """Returns (label, confidence, probabilities)"""
    cleaned   = clean_text(text)
    vec       = vectorizer.transform([cleaned])
    proba     = model.predict_proba(vec)[0]
    label     = "FAKE" if proba[1] > proba[0] else "REAL"
    confidence = max(proba) * 100
    return label, confidence, proba

def get_important_words(text, top_n=15):
    """
    Get most important words for this prediction
    using TF-IDF feature weights — no LIME needed,
    faster and works on deployment too
    """
    cleaned  = clean_text(text)
    vec      = vectorizer.transform([cleaned])
    features = vectorizer.get_feature_names_out()
    
    # Get the feature importances from the model
    if hasattr(model, 'feature_importances_'):
        # Random Forest
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # Logistic Regression — coef_ tells us which words push toward FAKE
        importances = model.coef_[0]
    else:
        return [], []

    # Get non-zero features in this specific text
    nonzero_indices = vec.nonzero()[1]
    
    word_scores = []
    for idx in nonzero_indices:
        word   = features[idx]
        score  = importances[idx] if hasattr(model, 'coef_') else importances[idx]
        tfidf  = vec[0, idx]
        word_scores.append((word, float(score) * float(tfidf)))

    # Sort: positive = fake indicators, negative = real indicators
    word_scores.sort(key=lambda x: abs(x[1]), reverse=True)
    
    fake_words = [(w, s) for w, s in word_scores if s > 0][:top_n//2]
    real_words = [(w, s) for w, s in word_scores if s < 0][:top_n//2]
    
    return fake_words, real_words

def scrape_url(url):
    """Scrape article text from a URL"""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        resp    = requests.get(url, headers=headers, timeout=10)
        soup    = BeautifulSoup(resp.content, 'html.parser')

        # Remove script and style tags
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()

        # Try to find article body
        article = soup.find('article')
        if article:
            text = article.get_text(separator=' ', strip=True)
        else:
            # Fall back to all paragraphs
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text(strip=True) for p in paragraphs])

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) < 50:
            return None, "Could not extract enough text from this URL. Try a different news site."
        return text, None

    except requests.exceptions.Timeout:
        return None, "Request timed out. The website took too long to respond."
    except requests.exceptions.ConnectionError:
        return None, "Could not connect to the URL. Check if it's a valid news link."
    except Exception as e:
        return None, f"Error scraping URL: {str(e)}"

def extract_text_from_image(image_file):
    """Extract text from screenshot using easyocr — works on cloud"""
    try:
        import easyocr
        import numpy as np
        image = Image.open(image_file)
        img_array = np.array(image)
        reader = easyocr.Reader(['en'], gpu=False)
        results = reader.readtext(img_array)
        text = ' '.join([r[1] for r in results])
        text = text.strip()
        if len(text) < 20:
            return None, "Could not read text from this image. Make sure the screenshot is clear."
        return text, None
    except Exception as e:
        return None, f"Error reading image: {str(e)}"

def detect_ai_content(text):
    """
    Simple AI-generated content detector using statistical features.
    AI text tends to be: very uniform sentence length, rare words used
    confidently, low burstiness, very few typos/informal language.
    Returns: (score 0-100, verdict, reasons)
    """
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

    if len(sentences) < 3:
        return 50, "Uncertain", ["Too short to analyze"]

    reasons  = []
    ai_score = 0

    # Feature 1: Sentence length variance (AI = very uniform)
    lengths  = [len(s.split()) for s in sentences]
    variance = np.var(lengths)
    if variance < 15:
        ai_score += 25
        reasons.append("Very uniform sentence lengths (AI trait)")
    elif variance > 60:
        reasons.append("Natural variation in sentence lengths (human trait)")

    # Feature 2: Average sentence length (AI tends to be 18-25 words)
    avg_len = np.mean(lengths)
    if 16 <= avg_len <= 26:
        ai_score += 15
        reasons.append(f"Avg sentence length {avg_len:.0f} words (typical of AI)")

    # Feature 3: Informal markers (humans use these, AI avoids)
    informal = len(re.findall(
        r"\b(lol|btw|tbh|imo|omg|gonna|wanna|kinda|sorta|yeah|nah|ok|okay)\b",
        text.lower()
    ))
    if informal == 0 and len(sentences) > 5:
        ai_score += 15
        reasons.append("No informal language (AI trait)")
    elif informal > 2:
        reasons.append("Contains informal language (human trait)")

    # Feature 4: Transition words (AI overuses these)
    transitions = len(re.findall(
        r"\b(furthermore|additionally|moreover|consequently|nevertheless|"
        r"therefore|subsequently|accordingly|notably|importantly)\b",
        text.lower()
    ))
    if transitions >= 2:
        ai_score += 20
        reasons.append(f"Heavy use of transition words ({transitions} found, AI trait)")

    # Feature 5: Repetitive phrases / hedging (AI hedges a lot)
    hedges = len(re.findall(
        r"\b(it is important to note|it should be noted|it is worth|"
        r"in conclusion|to summarize|as mentioned|as noted)\b",
        text.lower()
    ))
    if hedges >= 1:
        ai_score += 15
        reasons.append(f"Contains AI-style hedging phrases ({hedges} found)")

    # Feature 6: Question marks and exclamations (humans use more)
    punct_ratio = (text.count('!') + text.count('?')) / max(len(sentences), 1)
    if punct_ratio < 0.1 and len(sentences) > 5:
        ai_score += 10
        reasons.append("Very few exclamations/questions (AI trait)")

    ai_score = min(ai_score, 100)

    if ai_score >= 60:
        verdict = "Likely AI-generated"
    elif ai_score >= 35:
        verdict = "Possibly AI-assisted"
    else:
        verdict = "Likely human-written"

    return ai_score, verdict, reasons

def show_result(text, source_label=""):
    """Run prediction and display full results"""
    if len(text.strip()) < 20:
        st.warning("Text is too short. Please provide a full headline or article.")
        return

    label, confidence, proba = predict(text)
    fake_words, real_words   = get_important_words(text)
    ai_score, ai_verdict, ai_reasons = detect_ai_content(text)

    # ── Main result box ──
    if label == "FAKE":
        st.markdown(f"""
        <div class="result-fake">
            <div class="result-title">🚨 LIKELY FAKE NEWS</div>
            <div class="result-conf">Model confidence: {confidence:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-real">
            <div class="result-title">✅ LIKELY REAL NEWS</div>
            <div class="result-conf">Model confidence: {confidence:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    # ── Confidence bars ──
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Real probability",  f"{proba[0]*100:.1f}%")
        st.progress(float(proba[0]))
    with col2:
        st.metric("Fake probability",  f"{proba[1]*100:.1f}%")
        st.progress(float(proba[1]))

    # ── AI content detector ──
    st.divider()
    st.subheader("🤖 AI Content Detection")
    ai_col1, ai_col2 = st.columns([1, 2])
    with ai_col1:
        color = "#A32D2D" if ai_score >= 60 else "#854F0B" if ai_score >= 35 else "#3B6D11"
        st.markdown(f"""
        <div style="text-align:center; padding:16px; border-radius:10px; border: 0.5px solid {color}20;">
            <div style="font-size:32px; font-weight:500; color:{color}">{ai_score}</div>
            <div style="font-size:12px; color:{color}">AI score / 100</div>
            <div style="font-size:13px; margin-top:6px; font-weight:500">{ai_verdict}</div>
        </div>""", unsafe_allow_html=True)
    with ai_col2:
        for reason in ai_reasons:
            st.markdown(f"• {reason}")

    # ── Word importance ──
    if fake_words or real_words:
        st.divider()
        st.subheader("🔍 Key Words Analysis")
        st.caption("Words that influenced the prediction")

        wcol1, wcol2 = st.columns(2)
        with wcol1:
            st.markdown("**🚨 Fake indicators**")
            if fake_words:
                html = " ".join([f'<span class="word-fake">{w}</span>'
                                 for w, s in fake_words[:8]])
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.caption("None found")

        with wcol2:
            st.markdown("**✅ Real indicators**")
            if real_words:
                html = " ".join([f'<span class="word-real">{w}</span>'
                                 for w, s in real_words[:8]])
                st.markdown(html, unsafe_allow_html=True)
            else:
                st.caption("None found")

    # ── Disclaimer ──
    st.divider()
    st.caption("⚠️ This tool uses ML and is not 100% accurate. Always verify news from multiple trusted sources.")

# ──────────────────────────────────────────────
# MAIN UI
# ──────────────────────────────────────────────
st.title("🔍 Indian News Verifier")
st.markdown("Check if a news article is **real or fake** — paste text, share a URL, or upload a screenshot.")

if not model_loaded:
    st.error("Model not found. Please run `python train.py` first to train and save the model.")
    st.stop()

# Show model stats in header
with st.expander("ℹ️ About this model"):
    c1, c2, c3 = st.columns(3)
    c1.metric("Model",    model_info.get('name', 'ML Model'))
    c2.metric("Accuracy", f"{model_info.get('accuracy', 0)}%")
    c3.metric("F1 Score", f"{model_info.get('f1', 0)}")
    st.caption("Trained on 56,000+ Indian news articles from the IFND dataset.")

st.divider()

# ── Input method tabs ──
tab1, tab2, tab3 = st.tabs(["📝 Paste Text", "🔗 URL", "📸 Screenshot"])

with tab1:
    st.subheader("Paste news text")

    # Example buttons
    st.caption("Try an example:")
    ex1, ex2, ex3 = st.columns(3)
    if ex1.button("Example: Real news"):
        st.session_state['paste_text'] = "The Reserve Bank of India kept interest rates unchanged at its latest monetary policy meeting, citing concerns about inflation and global economic uncertainty. The decision was unanimous among the six-member committee."
    if ex2.button("Example: Fake news"):
        st.session_state['paste_text'] = "BREAKING: Government secretly adding mind-control chemicals to tap water across India! Whistleblower reveals shocking truth that mainstream media is hiding from you. Share before this gets deleted!"
    if ex3.button("Example: AI-written"):
        st.session_state['paste_text'] = "It is important to note that the economic situation in India has been experiencing significant changes. Furthermore, it should be noted that various factors have contributed to this development. Additionally, experts have noted that the situation warrants careful consideration."

    text_input = st.text_area(
        "Paste headline or full article here",
        value=st.session_state.get('paste_text', ''),
        height=180,
        placeholder="e.g. WHO praises India's COVID response, recommends vaccine for all age groups..."
    )

    if st.button("Analyse ↗", type="primary", key="btn_text"):
        if text_input.strip():
            with st.spinner("Analysing..."):
                show_result(text_input, "Pasted text")
        else:
            st.warning("Please paste some text first.")

with tab2:
    st.subheader("Paste a news article URL")
    st.caption("Works with most Indian news sites: NDTV, Times of India, Hindu, India Today, etc.")

    url_input = st.text_input(
        "News article URL",
        placeholder="https://www.ndtv.com/india-news/..."
    )

    if st.button("Fetch & Analyse ↗", type="primary", key="btn_url"):
        if url_input.strip():
            with st.spinner("Fetching article..."):
                text, error = scrape_url(url_input.strip())
            if error:
                st.error(error)
            else:
                st.success(f"Extracted {len(text.split())} words from article")
                with st.expander("View extracted text"):
                    st.write(text[:1000] + "..." if len(text) > 1000 else text)
                with st.spinner("Analysing..."):
                    show_result(text, "URL article")
        else:
            st.warning("Please enter a URL first.")

with tab3:
    st.subheader("Upload a screenshot")
    st.caption("Upload a screenshot of a WhatsApp forward, tweet, or news article.")

    uploaded = st.file_uploader(
        "Upload screenshot (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        help="Clear screenshots work best. Make sure text is readable."
    )

    if uploaded:
        st.image(uploaded, caption="Uploaded screenshot", use_column_width=True)

        if st.button("Extract Text & Analyse ↗", type="primary", key="btn_img"):
            with st.spinner("Reading text from image..."):
                text, error = extract_text_from_image(uploaded)
            if error:
                st.error(error)
            else:
                st.success(f"Extracted {len(text.split())} words from screenshot")
                with st.expander("View extracted text"):
                    st.write(text)
                with st.spinner("Analysing..."):
                    show_result(text, "Screenshot")

# ── Footer ──
st.divider()
st.markdown(
    "<center><small>Built with scikit-learn · TF-IDF · LIME · Streamlit · Trained on IFND Dataset</small></center>",
    unsafe_allow_html=True
)
