import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import model libraries
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.stattools import adfuller
import yfinance as yf
from datetime import datetime, timedelta

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Saham dengan Transfer Function",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header utama
st.markdown('<h1 class="main-header">📈 Sistem Prediksi Saham dengan Transfer Function</h1>', unsafe_allow_html=True)

# Sidebar untuk navigasi
st.sidebar.title("🔧 Panel Kontrol")
page = st.sidebar.selectbox("Pilih Halaman", [
    "🏠 Beranda", 
    "📊 Analisis Data", 
    "🔮 Prediksi Saham", 
    "📈 Analisis Teknikal",
    "💹 Portfolio Tracker",
    "📰 Berita Saham"
])

# Fungsi untuk memuat data
@st.cache_data
def load_data():
    try:
        df_asii = pd.read_csv('gui.csv')
        df_kurs = pd.read_csv('usd.csv')
        
        df_asii['Date'] = pd.to_datetime(df_asii['Date'])
        df_kurs['Date'] = pd.to_datetime(df_kurs['Date'])
        
        df_asii.set_index('Date', inplace=True)
        df_kurs.set_index('Date', inplace=True)
        
        merged_df = pd.merge(df_asii[['ANTM']], df_kurs[['Rate']], 
                           left_index=True, right_index=True, how='inner')
        merged_df = merged_df.rename(columns={'ANTM': 'Close', 'Rate': 'Kurs'})
        merged_df = merged_df.sort_index()
        
        return merged_df, df_asii, df_kurs
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

# Fungsi untuk prediksi
def predict_stock_price(merged_df, forecast_days, kurs_input):
    try:
        y = merged_df['Close'].astype(float)
        x = merged_df['Kurs'].astype(float)
        
        # Differencing
        dy = y.diff().dropna()
        dx = x.diff().dropna()
        
        # Model Transfer Function
        dx_lagged = dx.shift(1).dropna()
        dy_aligned = dy.iloc[1:]
        
        model_tf = SARIMAX(dy_aligned, exog=dx_lagged, order=(1, 0, 0))
        results_tf = model_tf.fit(disp=False)
        
        # Buat input untuk prediksi
        future_kurs_diff = np.diff([x.iloc[-1]] + kurs_input)
        
        # Forecast
        forecast = results_tf.get_forecast(steps=forecast_days, exog=future_kurs_diff)
        predicted_mean = forecast.predicted_mean
        conf_int = forecast.conf_int()
        
        # Kembalikan ke level harga
        last_real_price = y.iloc[-1]
        predicted_level = predicted_mean.cumsum() + last_real_price
        
        return predicted_level, conf_int, results_tf
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, None, None

# Fungsi untuk analisis teknikal
def calculate_technical_indicators(df):
    df = df.copy()
    
    # Moving Averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    return df

# Halaman Beranda
if page == "🏠 Beranda":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Selamat Datang di Sistem Prediksi Saham! 🎯
        
        Aplikasi ini menggunakan **Transfer Function Model** untuk memprediksi harga saham berdasarkan:
        - 📊 Data historis harga saham
        - 💱 Data kurs USD/IDR sebagai variabel eksogen
        - 🔬 Analisis time series yang mendalam
        
        ### Fitur Utama:
        - 🔮 **Prediksi Harga Saham** dengan input kurs custom
        - 📈 **Analisis Teknikal** lengkap dengan indikator
        - 💹 **Portfolio Tracker** untuk monitoring investasi
        - 📰 **Berita Saham** terkini
        - 📊 **Visualisasi Data** interaktif
        """)
    
    with col2:
        st.image("https://via.placeholder.com/300x200/1f77b4/ffffff?text=Stock+Prediction", 
                caption="Prediksi Saham dengan AI")
    
    # Load dan tampilkan ringkasan data
    merged_df, df_asii, df_kurs = load_data()
    if merged_df is not None:
        st

# Lanjutan dari kode Anda yang terpotong
    if merged_df is not None:
        st.markdown("### 📊 Ringkasan Data")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Harga Terakhir", f"Rp {merged_df['Close'].iloc[-1]:,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            change = merged_df['Close'].iloc[-1] - merged_df['Close'].iloc[-2]
            st.metric("Perubahan", f"Rp {change:,.0f}", delta=f"{change:,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Kurs USD/IDR", f"Rp {merged_df['Kurs'].iloc[-1]:,.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Data", f"{len(merged_df)} hari")
            st.markdown('</div>', unsafe_allow_html=True)

# Halaman Analisis Data
elif page == "📊 Analisis Data":
    st.header("📊 Analisis Data Historis")
    
    merged_df, df_asii, df_kurs = load_data()
    if merged_df is not None:
        # Tampilkan data
        st.subheader("Data Gabungan (Harga Saham + Kurs)")
        st.dataframe(merged_df.tail(10))
        
        # Grafik harga saham
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Harga Saham ANTM', 'Kurs USD/IDR'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=merged_df.index, y=merged_df['Close'], 
                      name='Harga ANTM', line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=merged_df.index, y=merged_df['Kurs'], 
                      name='Kurs USD/IDR', line=dict(color='red')),
            row=2, col=1
        )
        
        fig.update_layout(height=600, title_text="Analisis Data Historis")
        st.plotly_chart(fig, use_container_width=True)

# Halaman Prediksi Saham
elif page == "🔮 Prediksi Saham":
    st.header("🔮 Prediksi Harga Saham")
    
    merged_df, df_asii, df_kurs = load_data()
    if merged_df is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Parameter Prediksi")
            forecast_days = st.slider("Jumlah Hari Prediksi", 1, 30, 7)
            
            st.subheader("Input Kurs USD/IDR")
            kurs_input = []
            for i in range(forecast_days):
                kurs_val = st.number_input(
                    f"Hari {i+1}", 
                    value=float(merged_df['Kurs'].iloc[-1]),
                    key=f"kurs_{i}"
                )
                kurs_input.append(kurs_val)
            
            if st.button("🚀 Mulai Prediksi", type="primary"):
                with st.spinner("Sedang melakukan prediksi..."):
                    predicted_prices, conf_int, model_results = predict_stock_price(
                        merged_df, forecast_days, kurs_input
                    )
                    
                    if predicted_prices is not None:
                        st.session_state['predictions'] = predicted_prices
                        st.session_state['conf_int'] = conf_int
                        st.success("Prediksi berhasil!")
        
        with col2:
            if 'predictions' in st.session_state:
                st.subheader("Hasil Prediksi")
                
                # Tampilkan hasil prediksi
                pred_df = pd.DataFrame({
                    'Hari': range(1, len(st.session_state['predictions']) + 1),
                    'Prediksi Harga': st.session_state['predictions'].values
                })
                
                st.dataframe(pred_df)
                
                # Grafik prediksi
                fig = go.Figure()
                
                # Data historis
                fig.add_trace(go.Scatter(
                    x=merged_df.index[-30:], 
                    y=merged_df['Close'][-30:],
                    name='Data Historis',
                    line=dict(color='blue')
                ))
                
                # Prediksi
                future_dates = pd.date_range(
                    start=merged_df.index[-1] + timedelta(days=1),
                    periods=len(st.session_state['predictions'])
                )
                
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=st.session_state['predictions'],
                    name='Prediksi',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.update_layout(
                    title="Prediksi Harga Saham ANTM",
                    xaxis_title="Tanggal",
                    yaxis_title="Harga (Rp)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)

# Halaman Analisis Teknikal
elif page == "📈 Analisis Teknikal":
    st.header("📈 Analisis Teknikal")
    
    merged_df, df_asii, df_kurs = load_data()
    if merged_df is not None:
        # Hitung indikator teknikal
        tech_df = calculate_technical_indicators(merged_df)
        
        # Grafik dengan indikator
        fig = make_subplots(
            rows=
