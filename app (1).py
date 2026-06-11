import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .stApp {
        background-color: #0D1117;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #21262D;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #161B22;
        border: 1px solid #21262D;
        border-radius: 10px;
        padding: 16px 20px;
    }
    [data-testid="stMetric"] label {
        color: #8B949E !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #E6EDF3 !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        font-size: 13px !important;
    }

    /* Headers */
    h1, h2, h3 {
        color: #E6EDF3 !important;
    }
    h1 { font-weight: 700 !important; }
    h2 { font-weight: 600 !important; border-bottom: 1px solid #21262D; padding-bottom: 8px; }
    h3 { font-weight: 500 !important; color: #C9D1D9 !important; }

    /* Dividers */
    hr { border-color: #21262D !important; }

    /* DataFrames */
    .stDataFrame { border: 1px solid #21262D; border-radius: 8px; }

    /* Buttons */
    .stButton > button {
        background-color: #238636;
        color: white;
        border: 1px solid #2EA043;
        border-radius: 6px;
        font-weight: 600;
        padding: 10px 24px;
        font-size: 14px;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background-color: #2EA043;
        border-color: #3FB950;
    }

    /* Alert boxes */
    .fraud-alert {
        background-color: #3D1B1B;
        border: 1px solid #CF2222;
        border-left: 4px solid #CF2222;
        border-radius: 8px;
        padding: 14px 18px;
        color: #FF7B7B;
        font-weight: 600;
        margin: 8px 0;
    }
    .safe-alert {
        background-color: #1A2D1A;
        border: 1px solid #238636;
        border-left: 4px solid #238636;
        border-radius: 8px;
        padding: 14px 18px;
        color: #56D364;
        font-weight: 600;
        margin: 8px 0;
    }
    .info-box {
        background-color: #1C2A3A;
        border: 1px solid #1F6FEB;
        border-left: 4px solid #1F6FEB;
        border-radius: 8px;
        padding: 14px 18px;
        color: #79C0FF;
        margin: 8px 0;
    }

    /* Stat pill */
    .stat-pill {
        display: inline-block;
        background-color: #21262D;
        border: 1px solid #30363D;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 13px;
        color: #8B949E;
        margin: 3px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Section headers */
    .section-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8B949E;
        margin-bottom: 12px;
    }

    /* Progress indicator */
    .step-indicator {
        background-color: #21262D;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        border-left: 3px solid #388BFD;
        font-size: 14px;
        color: #C9D1D9;
    }

    /* Table styling */
    .stTable { color: #E6EDF3 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def fix_coord(val):
    """Fix malformed coordinate strings like '36.011.293' → 36.011293."""
    s = str(val)
    try:
        return float(s)
    except ValueError:
        pass
    parts = s.replace('-', '').split('.')
    if len(parts) >= 2:
        integer_part = parts[0]
        decimal_part = ''.join(parts[1:])[:6]
        try:
            result = float(integer_part + '.' + decimal_part)
            return -result if '-' in s else result
        except ValueError:
            return 0.0
    return 0.0


def preprocess(df: pd.DataFrame, label_encoders=None, fit: bool = True):
    """
    Feature engineering pipeline. Returns (X_df, label_encoders).
    If fit=True, fits new encoders; otherwise uses the provided ones.
    """
    df = df.copy()

    # ── Datetime features ──
    df['trans_datetime'] = pd.to_datetime(
        df['trans_date_trans_time'], format='%d/%m/%Y %H:%M', errors='coerce'
    )
    df['trans_hour']    = df['trans_datetime'].dt.hour
    df['trans_day']     = df['trans_datetime'].dt.dayofweek
    df['trans_month']   = df['trans_datetime'].dt.month
    df['is_weekend']    = (df['trans_day'] >= 5).astype(int)
    df['is_night']      = ((df['trans_hour'] >= 22) | (df['trans_hour'] <= 5)).astype(int)

    # ── Age ──
    df['dob_parsed'] = pd.to_datetime(df['dob'], format='%d/%m/%Y', errors='coerce')
    df['age'] = (df['trans_datetime'] - df['dob_parsed']).dt.days // 365

    # ── Fix coordinates ──
    for col in ['merch_lat', 'merch_long', 'lat', 'long']:
        if col in df.columns:
            df[col + '_num'] = df[col].apply(fix_coord)

    # ── Distance between cardholder and merchant ──
    if all(c in df.columns for c in ['lat_num', 'long_num', 'merch_lat_num', 'merch_long_num']):
        df['dist'] = np.sqrt(
            (df['lat_num'] - df['merch_lat_num'])**2 +
            (df['long_num'] - df['merch_long_num'])**2
        )
    else:
        df['dist'] = 0

    # ── Categorical encoding ──
    cat_cols = ['category', 'gender']
    if label_encoders is None:
        label_encoders = {}

    for col in cat_cols:
        if col not in df.columns:
            df[col + '_enc'] = 0
            continue
        if fit:
            le = LabelEncoder()
            df[col + '_enc'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            le = label_encoders.get(col)
            if le is not None:
                known = set(le.classes_)
                df[col + '_enc'] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in known else -1
                )
            else:
                df[col + '_enc'] = 0

    feature_cols = [
        'amt', 'trans_hour', 'trans_day', 'trans_month',
        'is_weekend', 'is_night', 'age', 'city_pop',
        'category_enc', 'gender_enc',
        'lat_num', 'long_num', 'merch_lat_num', 'merch_long_num', 'dist'
    ]
    available = [c for c in feature_cols if c in df.columns]
    return df[available].fillna(0), label_encoders


def get_feature_names():
    return [
        'Jumlah Transaksi (amt)', 'Jam Transaksi', 'Hari (0=Sen)', 'Bulan',
        'Akhir Pekan', 'Jam Malam (22:00–05:00)', 'Usia Pemilik Kartu',
        'Populasi Kota', 'Kategori Merchant', 'Jenis Kelamin',
        'Lat. Kartu', 'Long. Kartu', 'Lat. Merchant', 'Long. Merchant',
        'Jarak Kartu–Merchant'
    ]


@st.cache_data(show_spinner=False)
def load_data_safe():
    """Load data langsung dari URL GitHub Release."""
    # GANTI URL DI BAWAH INI DENGAN LINK YANG ANDA SALIN
    train_path = "https://github.com/IzzanKhlsh/TestDatMin/releases/download/Dataset/fraudTrain.csv"
    test_path  = "https://github.com/IzzanKhlsh/TestDatMin/releases/download/Dataset/fraudTest.csv"
    
    try:
        # Pandas bisa membaca file langsung dari URL internet
        train_df = pd.read_csv(train_path)
        test_df  = pd.read_csv(test_path)
    except Exception as e:
        st.error(f"❌ Gagal mengunduh dataset dari GitHub Release. Error: {e}")
        st.stop()
        
    # Lakukan sampling seperti kode asli Anda agar aplikasi tetap cepat
    if len(train_df) > 50_000:
        train_df = train_df.sample(n=50_000, random_state=42)
    if len(test_df) > 10_000:
        test_df = test_df.sample(n=10_000, random_state=42)
        
    return train_df, test_df


@st.cache_resource(show_spinner=False)
def train_model(_train_df):
    """Train Random Forest with SMOTE. Cached across reruns."""
    X, le = preprocess(_train_df, fit=True)
    y = _train_df['is_fraud']

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_res, y_res)
    return model, le


def plot_confusion_matrix(cm):
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Prediksi: Normal', 'Prediksi: Fraud'],
        y=['Aktual: Normal', 'Aktual: Fraud'],
        colorscale=[[0, '#0D1117'], [0.5, '#1F6FEB'], [1.0, '#388BFD']],
        text=cm,
        texttemplate='<b>%{text}</b>',
        textfont={"size": 18, "color": "white"},
        showscale=False,
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#C9D1D9',
        margin=dict(l=10, r=10, t=10, b=10),
        height=280,
        xaxis=dict(tickfont=dict(size=13)),
        yaxis=dict(tickfont=dict(size=13)),
    )
    return fig


def plot_roc(fpr, tpr, auc_score):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode='lines',
        line=dict(color='#388BFD', width=2.5),
        name=f'ROC (AUC={auc_score:.3f})'
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        line=dict(color='#30363D', width=1.5, dash='dash'),
        name='Random'
    ))
    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#0D1117',
        font_color='#C9D1D9',
        margin=dict(l=10, r=10, t=10, b=10),
        height=300,
        legend=dict(bgcolor='rgba(0,0,0,0)', font_color='#8B949E'),
        xaxis=dict(gridcolor='#21262D', zeroline=False),
        yaxis=dict(gridcolor='#21262D', zeroline=False),
    )
    return fig


def plot_pr_curve(precision, recall, ap_score):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision, mode='lines',
        line=dict(color='#56D364', width=2.5),
        name=f'PR (AP={ap_score:.3f})'
    ))
    fig.update_layout(
        xaxis_title='Recall',
        yaxis_title='Precision',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#0D1117',
        font_color='#C9D1D9',
        margin=dict(l=10, r=10, t=10, b=10),
        height=300,
        legend=dict(bgcolor='rgba(0,0,0,0)', font_color='#8B949E'),
        xaxis=dict(gridcolor='#21262D', zeroline=False),
        yaxis=dict(gridcolor='#21262D', zeroline=False),
    )
    return fig


def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    idx = np.argsort(importances)
    colors = ['#388BFD' if i == idx[-1] else '#1F6FEB' for i in range(len(idx))]

    fig = go.Figure(go.Bar(
        x=importances[idx],
        y=[feature_names[i] if i < len(feature_names) else f'f{i}' for i in idx],
        orientation='h',
        marker_color=[colors[i] for i in range(len(idx))],
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#0D1117',
        font_color='#C9D1D9',
        margin=dict(l=10, r=10, t=10, b=10),
        height=420,
        xaxis=dict(title='Importance', gridcolor='#21262D', zeroline=False),
        yaxis=dict(tickfont=dict(size=12)),
    )
    return fig


def plot_score_distribution(y_true, y_proba):
    df_plot = pd.DataFrame({'score': y_proba, 'label': y_true.map({0: 'Normal', 1: 'Fraud'})})
    fig = px.histogram(
        df_plot, x='score', color='label',
        barmode='overlay',
        color_discrete_map={'Normal': '#388BFD', 'Fraud': '#FF7B7B'},
        nbins=50,
        opacity=0.75,
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#0D1117',
        font_color='#C9D1D9',
        margin=dict(l=10, r=10, t=10, b=10),
        height=300,
        xaxis=dict(title='Fraud Probability Score', gridcolor='#21262D'),
        yaxis=dict(title='Count', gridcolor='#21262D'),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
    )
    return fig


def plot_category_fraud(df):
    fraud_by_cat = df.groupby('category').agg(
        total=('is_fraud', 'count'),
        fraud=('is_fraud', 'sum')
    ).reset_index()
    fraud_by_cat['fraud_rate'] = fraud_by_cat['fraud'] / fraud_by_cat['total'] * 100
    fraud_by_cat = fraud_by_cat.sort_values('fraud_rate', ascending=True)

    fig = go.Figure(go.Bar(
        x=fraud_by_cat['fraud_rate'],
        y=fraud_by_cat['category'],
        orientation='h',
        marker_color='#FF7B7B',
        text=fraud_by_cat['fraud_rate'].round(2).astype(str) + '%',
        textposition='outside',
        textfont=dict(color='#8B949E', size=11),
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#0D1117',
        font_color='#C9D1D9',
        margin=dict(l=10, r=10, t=10, b=10),
        height=400,
        xaxis=dict(title='Fraud Rate (%)', gridcolor='#21262D'),
        yaxis=dict(tickfont=dict(size=12)),
    )
    return fig


def plot_hourly_fraud(df):
    hourly = df.copy()
    hourly['hour'] = pd.to_datetime(
        hourly['trans_date_trans_time'], format='%d/%m/%Y %H:%M', errors='coerce'
    ).dt.hour
    h = hourly.groupby('hour').agg(total=('is_fraud', 'count'), fraud=('is_fraud', 'sum')).reset_index()
    h['fraud_rate'] = h['fraud'] / h['total'] * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(x=h['hour'], y=h['total'], name='Total Transaksi', marker_color='#21262D'))
    fig.add_trace(go.Scatter(x=h['hour'], y=h['fraud_rate'], name='Fraud Rate (%)',
                             line=dict(color='#FF7B7B', width=2.5), yaxis='y2'))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#0D1117',
        font_color='#C9D1D9',
        margin=dict(l=10, r=10, t=10, b=10),
        height=300,
        yaxis=dict(title='Total Transaksi', gridcolor='#21262D'),
        yaxis2=dict(title='Fraud Rate (%)', overlaying='y', side='right', gridcolor='#21262D'),
        xaxis=dict(title='Jam (0–23)', gridcolor='#21262D'),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🔍 Fraud Detection")
    st.markdown('<div class="section-label">Navigasi</div>', unsafe_allow_html=True)

    page = st.radio(
        "Pilih halaman",
        ["📊 Dashboard", "🤖 Training & Evaluasi", "🔎 Prediksi Batch", "🧪 Simulasi Transaksi"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown('<div class="section-label">Konfigurasi Model</div>', unsafe_allow_html=True)

    threshold = st.slider(
        "Threshold Fraud", 0.1, 0.9, 0.5, 0.05,
        help="Transaksi dengan skor ≥ threshold akan diklasifikasikan sebagai fraud"
    )
    n_estimators = st.select_slider("Jumlah Pohon (n_estimators)", [50, 100, 150, 200], value=150)

    st.markdown("---")
    st.markdown('<div class="section-label">Tentang</div>', unsafe_allow_html=True)
    st.caption(
        "Model: Random Forest + SMOTE\n\n"
        "Dataset: Credit Card Transactions\n\n"
        "Features: 15 engineered features"
    )

# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_data_safe():
    """Load from actual CSV or bundled sample."""
    train_path = 'fraudTrain.csv'
    test_path  = 'fraudTest.csv'
    if not os.path.exists(train_path):
        st.error("❌ File `fraudTrain.csv` tidak ditemukan. Pastikan file berada di folder yang sama dengan `app.py`.")
        st.stop()
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)
    if len(train_df) > 50_000:
        train_df = train_df.sample(n=50_000, random_state=42)
    if len(test_df) > 10_000:
        test_df = test_df.sample(n=10_000, random_state=42)
    return train_df, test_df


with st.spinner("⏳ Memuat data..."):
    train_df, test_df = load_data_safe()

with st.spinner("⏳ Melatih model..."):
    model, label_encoders = train_model(train_df)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

if page == "📊 Dashboard":
    st.markdown("# 📊 Dashboard Analisis Data")
    st.markdown("Eksplorasi pola dan karakteristik transaksi pada dataset.")
    st.markdown("---")

    # Top stats
    total_train    = len(train_df)
    fraud_train    = train_df['is_fraud'].sum()
    normal_train   = total_train - fraud_train
    fraud_rate_pct = fraud_train / total_train * 100
    avg_amt        = train_df['amt'].mean()
    avg_fraud_amt  = train_df[train_df['is_fraud'] == 1]['amt'].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Transaksi", f"{total_train:,}")
    c2.metric("Transaksi Normal", f"{normal_train:,}", delta=f"{(normal_train/total_train*100):.1f}%")
    c3.metric("Transaksi Fraud", f"{fraud_train:,}", delta=f"{fraud_rate_pct:.2f}%", delta_color="inverse")
    c4.metric("Avg. Nominal", f"${avg_amt:,.0f}")
    c5.metric("Avg. Nominal Fraud", f"${avg_fraud_amt:,.0f}", delta=f"+${avg_fraud_amt - avg_amt:,.0f}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Distribusi Kelas")
        fig_pie = go.Figure(go.Pie(
            labels=['Normal', 'Fraud'],
            values=[normal_train, fraud_train],
            hole=0.55,
            marker_colors=['#388BFD', '#FF7B7B'],
            textinfo='percent+label',
            textfont_size=13,
        ))
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#C9D1D9',
            margin=dict(l=0, r=0, t=10, b=10),
            height=280,
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.markdown("### Fraud Rate per Kategori Merchant")
        st.plotly_chart(plot_category_fraud(train_df), use_container_width=True)

    st.markdown("### Pola Fraud per Jam Transaksi")
    st.plotly_chart(plot_hourly_fraud(train_df), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### Distribusi Nominal Transaksi")
        fig_amt = px.histogram(
            train_df[train_df['amt'] < 500],
            x='amt', color=train_df[train_df['amt'] < 500]['is_fraud'].map({0: 'Normal', 1: 'Fraud'}),
            color_discrete_map={'Normal': '#388BFD', 'Fraud': '#FF7B7B'},
            nbins=60, opacity=0.75, barmode='overlay',
        )
        fig_amt.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#0D1117',
            font_color='#C9D1D9', margin=dict(l=10, r=10, t=10, b=10),
            height=280, legend=dict(bgcolor='rgba(0,0,0,0)'),
            xaxis=dict(gridcolor='#21262D'), yaxis=dict(gridcolor='#21262D'),
        )
        st.plotly_chart(fig_amt, use_container_width=True)

    with col4:
        st.markdown("### Fraud Rate per Jenis Kelamin")
        gender_stats = train_df.groupby('gender').agg(
            fraud=('is_fraud', 'sum'), total=('is_fraud', 'count')
        ).reset_index()
        gender_stats['rate'] = gender_stats['fraud'] / gender_stats['total'] * 100
        fig_g = px.bar(
            gender_stats, x='gender', y='rate',
            text=gender_stats['rate'].round(2).astype(str) + '%',
            color_discrete_sequence=['#388BFD'],
        )
        fig_g.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#0D1117',
            font_color='#C9D1D9', margin=dict(l=10, r=10, t=10, b=10),
            height=280,
            xaxis=dict(title='Jenis Kelamin', gridcolor='#21262D'),
            yaxis=dict(title='Fraud Rate (%)', gridcolor='#21262D'),
        )
        st.plotly_chart(fig_g, use_container_width=True)

    st.markdown("---")
    st.markdown("### Sample Data (Training)")
    st.dataframe(
        train_df[['trans_date_trans_time', 'merchant', 'category', 'amt', 'gender', 'city', 'state', 'is_fraud']].head(100),
        use_container_width=True, height=280
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TRAINING & EVALUASI
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 Training & Evaluasi":
    st.markdown("# 🤖 Training & Evaluasi Model")
    st.markdown("Performa model Random Forest dengan oversampling SMOTE pada data test.")
    st.markdown("---")

    # Prepare test data & predict
    with st.spinner("Mengevaluasi model pada data test..."):
        X_test, _ = preprocess(test_df, label_encoders=label_encoders, fit=False)
        y_test    = test_df['is_fraud']
        y_proba   = model.predict_proba(X_test)[:, 1]
        y_pred    = (y_proba >= threshold).astype(int)

        roc_auc   = roc_auc_score(y_test, y_proba)
        ap_score  = average_precision_score(y_test, y_proba)
        cm        = confusion_matrix(y_test, y_pred)
        cr        = classification_report(y_test, y_pred, output_dict=True)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        prec, rec, _ = precision_recall_curve(y_test, y_proba)

    tn, fp, fn, tp = cm.ravel()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC", f"{roc_auc:.4f}", help="Area Under Curve — semakin dekat 1 semakin baik")
    col2.metric("Average Precision", f"{ap_score:.4f}", help="Ringkasan precision-recall curve")
    col3.metric("Precision (Fraud)", f"{cr['1']['precision']:.2%}")
    col4.metric("Recall (Fraud)", f"{cr['1']['recall']:.2%}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("True Positive", f"{tp:,}", help="Fraud terdeteksi dengan benar")
    col6.metric("False Positive", f"{fp:,}", delta=f"−{fp}", delta_color="inverse", help="Normal diklasifikasikan sebagai fraud")
    col7.metric("False Negative", f"{fn:,}", delta=f"−{fn}", delta_color="inverse", help="Fraud tidak terdeteksi")
    col8.metric("True Negative", f"{tn:,}", help="Normal terdeteksi dengan benar")

    st.markdown("---")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### ROC Curve")
        st.plotly_chart(plot_roc(fpr, tpr, roc_auc), use_container_width=True)

    with col_r:
        st.markdown("### Precision-Recall Curve")
        st.plotly_chart(plot_pr_curve(prec, rec, ap_score), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Confusion Matrix")
        st.plotly_chart(plot_confusion_matrix(cm), use_container_width=True)

    with col_b:
        st.markdown("### Distribusi Skor Probabilitas")
        st.plotly_chart(plot_score_distribution(y_test, y_proba), use_container_width=True)

    st.markdown("### Feature Importance")
    feat_names = get_feature_names()[:len(model.feature_importances_)]
    st.plotly_chart(plot_feature_importance(model, feat_names), use_container_width=True)

    st.markdown("---")
    st.markdown("### Pipeline Model")
    steps = [
        ("1", "Feature Engineering", "Datetime parsing, age extraction, koordinat normalisasi, distance calculation, label encoding"),
        ("2", "SMOTE Oversampling", "Minority class (Fraud) di-oversample hingga seimbang dengan class Normal"),
        ("3", "Random Forest Classifier", f"n_estimators=150, max_depth=12, class_weight='balanced', n_jobs=-1"),
        ("4", "Threshold Tuning", f"Threshold saat ini: {threshold:.2f} — sesuaikan di sidebar untuk trade-off precision/recall"),
    ]
    for num, title, desc in steps:
        st.markdown(f"""
        <div class="step-indicator">
            <strong>Step {num} — {title}</strong><br>
            <span style="color:#8B949E; font-size:13px">{desc}</span>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDIKSI BATCH
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔎 Prediksi Batch":
    st.markdown("# 🔎 Prediksi Batch pada Data Test")
    st.markdown("Jalankan prediksi terhadap seluruh data test dan unduh hasilnya.")
    st.markdown("---")

    X_test, _ = preprocess(test_df, label_encoders=label_encoders, fit=False)
    y_proba   = model.predict_proba(X_test)[:, 1]
    y_pred    = (y_proba >= threshold).astype(int)

    result_df = test_df[['trans_date_trans_time', 'cc_num', 'merchant', 'category', 'amt',
                          'first', 'last', 'city', 'state', 'is_fraud']].copy().reset_index(drop=True)
    result_df['fraud_score']     = y_proba.round(4)
    result_df['prediksi']        = y_pred
    result_df['prediksi_label']  = result_df['prediksi'].map({0: '✅ Normal', 1: '🚨 FRAUD'})
    result_df['benar']           = (result_df['prediksi'] == result_df['is_fraud']).map({True: '✓', False: '✗'})

    # Summary
    n_fraud_pred = y_pred.sum()
    n_total      = len(y_pred)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Baris Diprediksi", f"{n_total:,}")
    col2.metric("Prediksi Fraud", f"{n_fraud_pred:,}", delta=f"{n_fraud_pred/n_total*100:.2f}%", delta_color="inverse")
    col3.metric("Prediksi Normal", f"{n_total - n_fraud_pred:,}")
    col4.metric("Threshold", f"{threshold:.2f}")

    st.markdown("---")

    # Filter
    col_f1, col_f2 = st.columns([1, 2])
    with col_f1:
        filter_view = st.selectbox("Filter tampilan", ["Semua", "Hanya Fraud Terdeteksi", "Hanya Normal", "Salah Klasifikasi"])
    with col_f2:
        score_min, score_max = st.slider("Rentang Fraud Score", 0.0, 1.0, (0.0, 1.0), 0.01)

    display_df = result_df.copy()
    if filter_view == "Hanya Fraud Terdeteksi":
        display_df = display_df[display_df['prediksi'] == 1]
    elif filter_view == "Hanya Normal":
        display_df = display_df[display_df['prediksi'] == 0]
    elif filter_view == "Salah Klasifikasi":
        display_df = display_df[display_df['benar'] == '✗']

    display_df = display_df[(display_df['fraud_score'] >= score_min) & (display_df['fraud_score'] <= score_max)]

    st.dataframe(
        display_df[['trans_date_trans_time', 'merchant', 'category', 'amt',
                     'first', 'last', 'fraud_score', 'prediksi_label', 'benar']],
        use_container_width=True, height=460
    )
    st.caption(f"Menampilkan {len(display_df):,} dari {n_total:,} baris")

    # Download
    csv_out = result_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Unduh Hasil Prediksi (CSV)",
        data=csv_out,
        file_name='hasil_prediksi_fraud.csv',
        mime='text/csv',
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SIMULASI TRANSAKSI
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🧪 Simulasi Transaksi":
    st.markdown("# 🧪 Simulasi Transaksi")
    st.markdown("Masukkan detail transaksi secara manual dan lihat apakah model mendeteksinya sebagai fraud.")
    st.markdown("---")

    all_categories = sorted(train_df['category'].dropna().unique().tolist())

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Informasi Transaksi")
        amt          = st.number_input("Nominal Transaksi ($)", min_value=0.01, max_value=100000.0, value=150.0, step=0.01)
        category     = st.selectbox("Kategori Merchant", all_categories)
        trans_hour   = st.slider("Jam Transaksi", 0, 23, 14)
        trans_day    = st.selectbox("Hari", ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"])
        trans_month  = st.slider("Bulan", 1, 12, 6)

    with col2:
        st.markdown("#### Informasi Pemilik Kartu")
        age          = st.slider("Usia Pemilik Kartu", 18, 90, 35)
        gender       = st.radio("Jenis Kelamin", ["M", "F"], horizontal=True)
        city_pop     = st.number_input("Populasi Kota", min_value=100, max_value=5_000_000, value=50000, step=100)
        lat          = st.number_input("Latitude Kartu", value=37.5, format="%.4f")
        long_val     = st.number_input("Longitude Kartu", value=-95.5, format="%.4f")
        merch_lat    = st.number_input("Latitude Merchant", value=38.0, format="%.4f")
        merch_long   = st.number_input("Longitude Merchant", value=-96.0, format="%.4f")

    day_map  = {"Senin": 0, "Selasa": 1, "Rabu": 2, "Kamis": 3, "Jumat": 4, "Sabtu": 5, "Minggu": 6}
    day_num  = day_map[trans_day]
    is_wknd  = int(day_num >= 5)
    is_night = int(trans_hour >= 22 or trans_hour <= 5)
    dist_val = np.sqrt((lat - merch_lat)**2 + (long_val - merch_long)**2)

    # Encode category & gender
    cat_le   = label_encoders.get('category')
    gen_le   = label_encoders.get('gender')
    cat_enc  = cat_le.transform([category])[0] if cat_le and category in cat_le.classes_ else 0
    gen_enc  = gen_le.transform([gender])[0]   if gen_le and gender in gen_le.classes_   else 0

    feature_vector = np.array([[
        amt, trans_hour, day_num, trans_month,
        is_wknd, is_night, age, city_pop,
        cat_enc, gen_enc,
        lat, long_val, merch_lat, merch_long, dist_val
    ]])

    st.markdown("---")
    if st.button("🔍 Analisis Transaksi Ini"):
        proba   = model.predict_proba(feature_vector)[0][1]
        is_fraud_pred = proba >= threshold

        st.markdown("### Hasil Analisis")
        col_res1, col_res2, col_res3 = st.columns(3)
        col_res1.metric("Fraud Score", f"{proba:.4f}")
        col_res2.metric("Threshold", f"{threshold:.2f}")
        col_res3.metric("Status", "🚨 FRAUD" if is_fraud_pred else "✅ NORMAL")

        if is_fraud_pred:
            st.markdown(f"""
            <div class="fraud-alert">
                🚨 PERINGATAN: Transaksi ini terdeteksi sebagai <strong>FRAUD</strong><br>
                Skor probabilitas: <strong>{proba:.2%}</strong> — melebihi threshold {threshold:.2f}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="safe-alert">
                ✅ Transaksi ini diprediksi <strong>NORMAL</strong><br>
                Skor probabilitas: <strong>{proba:.2%}</strong> — di bawah threshold {threshold:.2f}
            </div>
            """, unsafe_allow_html=True)

        # Risk gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            number={'suffix': '%', 'font': {'color': '#FF7B7B' if is_fraud_pred else '#56D364', 'size': 36}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#8B949E'},
                'bar': {'color': '#FF7B7B' if is_fraud_pred else '#56D364'},
                'bgcolor': '#21262D',
                'bordercolor': '#30363D',
                'steps': [
                    {'range': [0, threshold * 100], 'color': '#1A2D1A'},
                    {'range': [threshold * 100, 100], 'color': '#3D1B1B'},
                ],
                'threshold': {
                    'line': {'color': '#FFD700', 'width': 3},
                    'thickness': 0.75,
                    'value': threshold * 100
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#C9D1D9',
            margin=dict(l=20, r=20, t=30, b=20),
            height=240,
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Feature context
        st.markdown("#### Ringkasan Fitur yang Digunakan")
        feat_display = {
            "Nominal": f"${amt:,.2f}",
            "Kategori": category,
            "Jam": f"{trans_hour}:00",
            "Hari": trans_day,
            "Akhir Pekan": "Ya" if is_wknd else "Tidak",
            "Jam Malam": "Ya" if is_night else "Tidak",
            "Usia": f"{age} tahun",
            "Pop. Kota": f"{city_pop:,}",
            "Jarak ke Merchant": f"{dist_val:.4f}°",
        }
        pills_html = "".join([f'<span class="stat-pill">{k}: {v}</span>' for k, v in feat_display.items()])
        st.markdown(f'<div style="margin-top:8px">{pills_html}</div>', unsafe_allow_html=True)
