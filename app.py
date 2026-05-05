import os
import re
import ast
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentiHotel — Analisis Sentimen Hotel Indonesia",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}
h1, h2, h3, .stTitle {
    font-family: 'Sora', sans-serif;
}

/* Hide Streamlit default elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display:none;}

/* Main background */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Header Hero */
.hero-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f2942 100%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    border: 1px solid rgba(99, 179, 237, 0.2);
    position: relative;
    overflow: hidden;
}
.hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(99,179,237,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Sora', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #f0f9ff;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    color: #93c5fd;
    font-size: 1rem;
    margin: 0;
    font-weight: 400;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,179,237,0.15);
    color: #93c5fd;
    border: 1px solid rgba(99,179,237,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 1rem;
    text-transform: uppercase;
}

/* Sentiment Cards */
.sentiment-card {
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.4rem 0;
    border-left: 4px solid;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}
.sentiment-pos {
    background: rgba(16, 185, 129, 0.08);
    border-color: #10b981;
}
.sentiment-neg {
    background: rgba(239, 68, 68, 0.08);
    border-color: #ef4444;
}
.sentiment-neu {
    background: rgba(148, 163, 184, 0.08);
    border-color: #94a3b8;
}
.aspect-name {
    font-weight: 600;
    font-size: 0.95rem;
    color: #1e293b;
    min-width: 90px;
}
.sentiment-label {
    font-size: 0.85rem;
    font-weight: 500;
}
.prob-bar-bg {
    flex: 1;
    height: 6px;
    background: #e2e8f0;
    border-radius: 3px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s ease;
}

/* Info box */
.info-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
}

/* Metric chip */
.metric-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #f1f5f9;
    border-radius: 8px;
    padding: 8px 16px;
    margin: 4px;
    font-size: 0.88rem;
    font-weight: 500;
    color: #334155;
}

/* Sidebar */
.sidebar-section {
    background: #f8fafc;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    border: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# KONFIGURASI & PREPROCESSING (sama persis seperti notebook)
# ══════════════════════════════════════════════════════════════════════════════

ASPECTS = ['Pelayanan', 'Kamar', 'Fasilitas', 'Makanan', 'Lokasi', 'Suasana', 'Harga']
REVERSE_LABEL = {0: 'Negatif', 1: 'Tidak Dibahas', 2: 'Positif'}
panjang_maksimal = 100

MODEL_DIR      = "models"      # folder tempat file .keras disimpan
TOKENIZER_DIR  = "tokenizers"  # folder tempat file tokenizer JSON disimpan

NORM_DICT = {
    'yg': 'yang', 'dgn': 'dengan', 'dg': 'dengan', 'utk': 'untuk',
    'krn': 'karena', 'karna': 'karena', 'krna': 'karena', 'sdh': 'sudah',
    'udh': 'sudah', 'udah': 'sudah', 'bs': 'bisa', 'bsa': 'bisa',
    'sy': 'saya', 'aku': 'saya', 'jgn': 'jangan', 'kl': 'kalau',
    'klo': 'kalau', 'kalo': 'kalau', 'klu': 'kalau', 'dtg': 'datang',
    'lg': 'lagi', 'lgi': 'lagi', 'tp': 'tapi', 'tpi': 'tapi',
    'hrs': 'harus', 'msh': 'masih', 'msih': 'masih', 'bgt': 'banget',
    'bngt': 'banget', 'bgtt': 'banget',
    'aja': 'saja', 'aj': 'saja', 'deh': '', 'dong': '', 'sih': '',
    'mah': '', 'kok': '', 'sich': '', 'doong': '',
    'ga': 'tidak', 'gak': 'tidak', 'ngak': 'tidak', 'nggak': 'tidak',
    'ngga': 'tidak', 'gk': 'tidak', 'tak': 'tidak', 'tdk': 'tidak',
    'enggak': 'tidak', 'nggk': 'tidak', 'bkn': 'bukan', 'blm': 'belum',
    'blum': 'belum', 'blom': 'belum', 'belom': 'belum',
    'emg': 'memang', 'emang': 'memang', 'kayak': 'seperti', 'kyk': 'seperti',
    'gimana': 'bagaimana', 'gmn': 'bagaimana', 'gitu': 'begitu',
    'gini': 'begini', 'gt': 'begitu', 'abis': 'habis', 'smpe': 'sampai',
    'ampe': 'sampai', 'sampe': 'sampai', 'cuma': 'hanya', 'cm': 'hanya',
    'krg': 'kurang', 'kurg': 'kurang', 'sgt': 'sangat', 'bener': 'benar',
    'bner': 'benar', 'lbh': 'lebih', 'lbih': 'lebih',
    'staffnya': 'staf', 'stafnya': 'staf', 'staffs': 'staf',
    'chekout': 'checkout', 'checkin': 'check in',
    'ac': 'pendingin ruangan', 'wfi': 'wifi', 'wfe': 'wifi',
    'good': 'bagus', 'nice': 'bagus', 'great': 'bagus', 'best': 'terbaik',
    'clean': 'bersih', 'ok': 'oke', 'okay': 'oke', 'overall': 'semua',
    'harga_jutaan': 'harga mahal', 'harga_ribuan': 'harga terjangkau',
    'harga_nominal': 'harga',
}

STOPWORDS = {
    'saya', 'kami', 'kita', 'mereka', 'dia', 'anda', 'kamu', 'ia',
    'di', 'ke', 'dari', 'pada', 'dalam', 'untuk', 'dan', 'atau', 'juga',
    'dengan', 'karena', 'seperti', 'sehingga', 'oleh', 'tentang', 'bahwa',
    'hingga', 'sampai', 'sejak', 'antara', 'terhadap', 'selama', 'setelah',
    'sebelum', 'ketika', 'agar', 'supaya', 'walau', 'walaupun', 'meski',
    'meskipun', 'apabila', 'bagi', 'kalau', 'tapi', 'namun',
    'ini', 'itu', 'sini', 'situ', 'sana', 'tersebut',
    'adalah', 'ialah',
    'akan', 'sudah', 'telah', 'bisa', 'dapat', 'boleh', 'harus',
    'perlu', 'mau', 'ingin', 'punya', 'sedang', 'masih', 'lagi',
    'pun', 'lah', 'kah', 'ya', 'yah', 'oh', 'ah',
    'apa', 'siapa', 'mana', 'kapan', 'kenapa', 'mengapa', 'bagaimana',
    'yg', 'yang', 'nya', 'ada',
}

# ── Preprocessing Functions ──────────────────────────────────────────────────

def case_folding(text):
    return text.lower()

def text_cleaning(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'#\w+|@\w+', '', text)
    text = re.sub(r'rp\.?\s*\d+[\.,]?\d*', ' harga nominal ', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+\s*(juta|jt)\b', ' harga mahal ', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+\s*(ribu|rb|k)\b', ' harga terjangkau ', text, flags=re.IGNORECASE)
    bintang_map = {'1': 'sangat jelek', '2': 'jelek', '3': 'standar', '4': 'bagus', '5': 'mewah'}
    text = re.sub(
        r'bintang\s*(\d)',
        lambda m: f" {bintang_map.get(m.group(1), 'hotel berbintang')} ",
        text, flags=re.IGNORECASE
    )
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\b\d+\b', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenization(text):
    return text.split()

def normalization(tokens):
    normalized = [NORM_DICT.get(word, word) for word in tokens]
    return [w for w in normalized if w.strip()]

def stopword_removal(tokens):
    return [t for t in tokens if t not in STOPWORDS]

def join_tokens(tokens):
    if isinstance(tokens, str):
        try:
            tokens = ast.literal_eval(tokens)
        except (ValueError, SyntaxError):
            pass
    if isinstance(tokens, list):
        return ' '.join(tokens)
    return str(tokens)

def preprocess_input(teks):
    teks = case_folding(teks)
    teks = text_cleaning(teks)
    tokens = tokenization(teks)
    tokens = normalization(tokens)
    tokens = stopword_removal(tokens)
    return join_tokens(tokens)

def aspect_slug(aspek):
    return aspek.lower().replace(' ', '_')

# ── Load Models (cached) ─────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_models():
    """Load semua model keras dan tokenizer dari disk."""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.text import tokenizer_from_json

        # Import AttentionLayer — harus dari file yang sama atau diimport
        # Jika kamu punya file terpisah, sesuaikan import-nya
        class AttentionLayer(tf.keras.layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            def build(self, input_shape):
                self.W = self.add_weight(
                    name='att_weight', shape=(input_shape[-1], 1),
                    initializer='glorot_uniform', trainable=True
                )
                self.b = self.add_weight(
                    name='att_bias', shape=(input_shape[1], 1),
                    initializer='zeros', trainable=True
                )
                super().build(input_shape)

            def call(self, x):
                e = tf.nn.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
                a = tf.nn.softmax(e, axis=1)
                output = x * a
                return tf.reduce_sum(output, axis=1)

            def get_config(self):
                return super().get_config()

        models_loaded = {}
        tokenizers_loaded = {}

        for aspek in ASPECTS:
            slug = aspect_slug(aspek)
            model_path     = f"{MODEL_DIR}/undersampling_{slug}.keras"
            tokenizer_path = f"{TOKENIZER_DIR}/tokenizer_{slug}.json"

            if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
                continue

            models_loaded[aspek] = load_model(
                model_path,
                custom_objects={'AttentionLayer': AttentionLayer}
            )
            with open(tokenizer_path, 'r', encoding='utf-8') as f:
                tokenizers_loaded[aspek] = tokenizer_from_json(f.read())

        return models_loaded, tokenizers_loaded, None

    except ImportError as e:
        return {}, {}, f"TensorFlow tidak terinstall: {e}"
    except Exception as e:
        return {}, {}, str(e)

# ── Prediction Function ──────────────────────────────────────────────────────

def predict_review(teks_asli, loaded_models, loaded_tokenizers):
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    teks_bersih = preprocess_input(teks_asli)
    results = {}

    for aspek in ASPECTS:
        if aspek not in loaded_models:
            continue

        tokenizer_aspek = loaded_tokenizers[aspek]
        seq  = tokenizer_aspek.texts_to_sequences([teks_bersih])
        pad  = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
        prob = loaded_models[aspek].predict(pad, verbose=0)[0]
        label = int(np.argmax(prob))

        results[aspek] = {
            'label': label,
            'label_str': REVERSE_LABEL[label],
            'prob_neg': float(prob[0]),
            'prob_neu': float(prob[1]),
            'prob_pos': float(prob[2]),
        }

    return results, teks_bersih

# ══════════════════════════════════════════════════════════════════════════════
# UI LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

# ── Hero Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
  <div class="hero-badge">🏨 NLP · Analisis Sentimen</div>
  <div class="hero-title">SentiHotel Indonesia</div>
  <p class="hero-subtitle">
    Analisis sentimen ulasan hotel secara otomatis menggunakan model BiLSTM + Attention
    dengan FastText embedding Bahasa Indonesia — 7 aspek sekaligus.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Pengaturan")

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    show_proba = st.toggle("Tampilkan probabilitas", value=True)
    show_preprocessed = st.toggle("Tampilkan teks setelah preprocessing", value=False)
    show_chart = st.toggle("Tampilkan grafik probabilitas", value=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Aspek yang Dianalisis")
    aspect_icons = {
        'Pelayanan': '👥', 'Kamar': '🛏️', 'Fasilitas': '🏊',
        'Makanan': '🍽️', 'Lokasi': '📍', 'Suasana': '✨', 'Harga': '💰'
    }
    for asp in ASPECTS:
        st.markdown(f"{aspect_icons.get(asp, '•')} **{asp}**")

    st.markdown("---")
    st.markdown("### ℹ️ Info Model")
    st.markdown("""
    - **Arsitektur**: BiLSTM + Attention  
    - **Embedding**: FastText ID  
    - **Balancing**: Random Undersampling  
    - **Kelas**: Negatif · Tidak Dibahas · Positif
    """)

# ── Main Content ─────────────────────────────────────────────────────────────
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("#### ✍️ Masukkan Ulasan Hotel")

    # Contoh ulasan
    contoh = {
        "Pilih contoh ulasan...": "",
        "⭐ Ulasan Campuran": "stafnya cuek banget, kamar bau rokok dan makanannya asin. yang bagus kolam renangnya",
        "✅ Ulasan Positif": "sarapannya enak sekali, dengan harga yang murah bisa dapat kamar yang bagus dan fasilitas mewah.",
        "❌ Ulasan Negatif": "kamarnya kotor, staf tidak ramah, dan lokasinya jauh dari mana-mana. sangat mengecewakan.",
        "🔀 Ulasan Detail": "kamarnya luas dan bisa liat pemandangan yang bagus, namun stafnya cuek dan harganya terlalu mahal untuk fasilitas yang ada",
    }

    selected = st.selectbox("Atau pilih contoh:", list(contoh.keys()))
    default_text = contoh[selected]

    user_input = st.text_area(
        "Teks ulasan:",
        value=default_text,
        height=150,
        placeholder="Contoh: Kamarnya sangat bersih dan nyaman, staf ramah, tapi makanannya biasa saja...",
        label_visibility="collapsed"
    )

    analyze_btn = st.button("🔍 Analisis Sentimen", type="primary", use_container_width=True)

    # Info box
    if not user_input:
        st.info("💡 Ketik ulasan hotel dalam Bahasa Indonesia, lalu klik **Analisis Sentimen**.")

with col_result:
    st.markdown("#### 📊 Hasil Analisis")

    if not analyze_btn or not user_input.strip():
        st.markdown("""
        <div class="info-box" style="text-align:center; color:#94a3b8; padding: 3rem;">
            <div style="font-size:3rem">🏨</div>
            <div style="margin-top:1rem; font-size:0.95rem">
                Hasil analisis akan muncul di sini<br>setelah kamu mengklik <strong>Analisis Sentimen</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Load models
        with st.spinner("Memuat model..."):
            loaded_models, loaded_tokenizers, error = load_models()

        if error:
            st.error(f"❌ Error memuat model: {error}")
            st.info("Pastikan folder `models/` dan `tokenizers/` ada dan berisi file yang tepat.")
        elif not loaded_models:
            st.warning("⚠️ Tidak ada model yang ditemukan. Pastikan struktur folder sudah benar:")
            st.code("""
📁 app.py
📁 models/
   ├── undersampling_pelayanan.keras
   ├── undersampling_kamar.keras
   └── ...
📁 tokenizers/
   ├── tokenizer_pelayanan.json
   ├── tokenizer_kamar.json
   └── ...
            """)
        else:
            with st.spinner("Menganalisis ulasan..."):
                results, teks_bersih = predict_review(user_input, loaded_models, loaded_tokenizers)

            if show_preprocessed:
                with st.expander("🔧 Teks setelah preprocessing"):
                    st.code(teks_bersih, language=None)

            # Summary chips
            pos_count = sum(1 for r in results.values() if r['label'] == 2)
            neg_count = sum(1 for r in results.values() if r['label'] == 0)
            neu_count = sum(1 for r in results.values() if r['label'] == 1)

            st.markdown(
                f'<div style="margin-bottom:1rem">'
                f'<span class="metric-chip">✅ Positif: {pos_count}</span>'
                f'<span class="metric-chip">❌ Negatif: {neg_count}</span>'
                f'<span class="metric-chip">➖ Tidak Dibahas: {neu_count}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

            # Per-aspect results
            COLOR_MAP = {
                2: ('#10b981', 'sentiment-pos', '✅'),
                0: ('#ef4444', 'sentiment-neg', '❌'),
                1: ('#94a3b8', 'sentiment-neu', '➖'),
            }

            for aspek in ASPECTS:
                if aspek not in results:
                    continue
                r = results[aspek]
                color, css_class, icon = COLOR_MAP[r['label']]
                conf = max(r['prob_neg'], r['prob_neu'], r['prob_pos'])

                st.markdown(f"""
                <div class="sentiment-card {css_class}">
                  <span style="font-size:1.3rem">{aspect_icons.get(aspek, '•')}</span>
                  <span class="aspect-name">{aspek}</span>
                  <span class="sentiment-label" style="color:{color}; min-width:120px">
                    {icon} {r['label_str']}
                  </span>
                  {'<span style="color:#94a3b8; font-size:0.8rem">(' + f"{conf:.0%}" + ')</span>' if show_proba else ''}
                </div>
                """, unsafe_allow_html=True)

            # Probability chart
            if show_chart and results:
                st.markdown("---")
                st.markdown("**📈 Distribusi Probabilitas per Aspek**")

                fig, ax = plt.subplots(figsize=(8, 4))
                aspek_list = [a for a in ASPECTS if a in results]
                x = np.arange(len(aspek_list))
                w = 0.25

                neg_vals = [results[a]['prob_neg'] for a in aspek_list]
                neu_vals = [results[a]['prob_neu'] for a in aspek_list]
                pos_vals = [results[a]['prob_pos'] for a in aspek_list]

                ax.bar(x - w, neg_vals, w, label='Negatif', color='#ef4444', alpha=0.85)
                ax.bar(x,     neu_vals, w, label='Tidak Dibahas', color='#94a3b8', alpha=0.85)
                ax.bar(x + w, pos_vals, w, label='Positif', color='#10b981', alpha=0.85)

                ax.set_xticks(x)
                ax.set_xticklabels(aspek_list, fontsize=9)
                ax.set_ylim(0, 1.1)
                ax.set_ylabel('Probabilitas', fontsize=9)
                ax.legend(fontsize=8, loc='upper right')
                ax.spines[['top', 'right']].set_visible(False)
                ax.set_facecolor('#f8fafc')
                fig.patch.set_facecolor('#f8fafc')
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#94a3b8; font-size:0.82rem'>"
    "SentiHotel · BiLSTM + Attention · FastText ID · Dibangun dengan Streamlit"
    "</div>",
    unsafe_allow_html=True
)
