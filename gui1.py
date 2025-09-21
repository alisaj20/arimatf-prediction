import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, ccf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import kpss
from scipy import stats
import warnings
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools

warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Saham LQ45 - ARIMA Transfer Function",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk mendapatkan data saham
@st.cache_data
def get_stock_data(ticker, start_date, end_date):
    """Mengambil data saham dari Yahoo Finance"""
    try:
        stock = yf.download(ticker, start=start_date, end=end_date)
        return stock
    except Exception as e:
        st.error(f"Error mengambil data saham: {e}")
        return None

# Fungsi untuk mendapatkan data kurs USD/IDR
@st.cache_data
def get_usd_idr_data(start_date, end_date):
    """Mengambil data kurs USD/IDR"""
    try:
        usd_idr = yf.download('USDIDR=X', start=start_date, end=end_date)
        return usd_idr
    except Exception as e:
        st.error(f"Error mengambil data kurs: {e}")
        return None

# Fungsi untuk uji stasioneritas
def adf_test(series, title=""):
    """Uji Augmented Dickey-Fuller untuk stasioneritas"""
    result = adfuller(series.dropna(), autolag='AIC')
    output = {
        'Test Statistic': result[0],
        'p-value': result[1],
        'Critical Values': result[4],
        'Used Lag': result[2],
        'Number of Observations': result[3]
    }
    
    st.write(f"**Hasil Uji ADF - {title}:**")
    st.write(f"- Test Statistic: {output['Test Statistic']:.6f}")
    st.write(f"- p-value: {output['p-value']:.6f}")
    st.write(f"- Critical Values:")
    for key, value in output['Critical Values'].items():
        st.write(f"  - {key}: {value:.6f}")
    
    if output['p-value'] <= 0.05:
        st.success("✅ Data stasioner (p-value ≤ 0.05)")
        return True
    else:
        st.warning("⚠️ Data tidak stasioner (p-value > 0.05)")
        return False

# Fungsi untuk uji normalitas
def normality_test(residuals):
    """Uji normalitas residual"""
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    jarque_stat, jarque_p = stats.jarque_bera(residuals)
    
    st.write("**Uji Normalitas Residual:**")
    st.write(f"- Shapiro-Wilk Test: statistic={shapiro_stat:.6f}, p-value={shapiro_p:.6f}")
    st.write(f"- Jarque-Bera Test: statistic={jarque_stat:.6f}, p-value={jarque_p:.6f}")
    
    if shapiro_p > 0.05:
        st.success("✅ Residual berdistribusi normal")
        return True
    else:
        st.warning("⚠️ Residual tidak berdistribusi normal")
        return False

# Fungsi untuk uji Ljung-Box
def ljung_box_test(residuals, lags=10):
    """Uji Ljung-Box untuk autokorelasi residual"""
    result = acorr_ljungbox(residuals, lags=lags, return_df=True)
    st.write("**Uji Ljung-Box (Autokorelasi Residual):**")
    st.write(result)
    
    if all(result['lb_pvalue'] > 0.05):
        st.success("✅ Tidak ada autokorelasi residual (white noise)")
        return True
    else:
        st.warning("⚠️ Ada autokorelasi residual")
        return False

# Fungsi untuk menghitung metrik evaluasi
def calculate_metrics(actual, predicted):
    """Menghitung MAE, RMSE, dan MAPE"""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return mae, rmse, mape

# Fungsi untuk prewhitening
def prewhiten(input_series, output_series, p, d, q):
    """Prewhitening untuk deret input dan output"""
    # Fit ARIMA model pada deret input
    arima_input = ARIMA(input_series, order=(p, d, q))
    arima_input_fit = arima_input.fit()
    
    # Dapatkan residual (prewhitened input)
    input_residuals = arima_input_fit.resid
    
    # Apply filter yang sama pada deret output
    # Ini simulasi sederhana, dalam praktik perlu implementasi yang lebih detail
    output_filtered = output_series - output_series.shift(1)
    
    return input_residuals, output_filtered

# Daftar saham LQ45
LQ45_STOCKS = {
    'ADRO.JK': 'Adaro Energy',
    'ANTM.JK': 'Aneka Tambang',
    'ASII.JK': 'Astra International',
    'BBCA.JK': 'Bank Central Asia',
    'BBNI.JK': 'Bank Negara Indonesia',
    'BBRI.JK': 'Bank Rakyat Indonesia',
    'BMRI.JK': 'Bank Mandiri',
    'BSDE.JK': 'Bumi Serpong Damai',
    'CPIN.JK': 'Charoen Pokphand Indonesia',
    'EXCL.JK': 'XL Axiata',
    'GGRM.JK': 'Gudang Garam',
    'HMSP.JK': 'HM Sampoerna',
    'ICBP.JK': 'Indofood CBP Sukses Makmur',
    'INCO.JK': 'Vale Indonesia',
    'INDF.JK': 'Indofood Sukses Makmur',
    'INKP.JK': 'Indah Kiat Pulp & Paper',
    'INTP.JK': 'Indocement Tunggal Prakasa',
    'ITMG.JK': 'Indo Tambangraya Megah',
    'JSMR.JK': 'Jasa Marga',
    'KLBF.JK': 'Kalbe Farma',
    'LPKR.JK': 'Lippo Karawaci',
    'LPPF.JK': 'Matahari Department Store',
    'MNCN.JK': 'Media Nusantara Citra',
    'PGAS.JK': 'Perusahaan Gas Negara',
    'PTBA.JK': 'Bukit Asam',
    'PWON.JK': 'Pakuwon Jati',
    'SMGR.JK': 'Semen Indonesia',
    'SSMS.JK': 'Sawit Sumbermas Sarana',
    'TINS.JK': 'Timah',
    'TLKM.JK': 'Telkom Indonesia',
    'TPIA.JK': 'Chandra Asri Petrochemical',
    'UNTR.JK': 'United Tractors',
    'UNVR.JK': 'Unilever Indonesia',
    'WIKA.JK': 'Wijaya Karya',
    'WSKT.JK': 'Waskita Karya'
}

# Sidebar
st.sidebar.title("🔧 Pengaturan Analisis")

# Deskripsi
st.sidebar.markdown("""
### 📊 Tentang Aplikasi
Aplikasi ini menggunakan metode **ARIMA Transfer Function** untuk prediksi saham LQ45 dengan variabel eksogen kurs USD/IDR.

**Fitur Utama:**
- Analisis stasioneritas dengan ADF Test
- Transformasi Box-Cox untuk stabilisasi varians
- Model ARIMA untuk deret input dan output
- Cross-correlation Function (CCF)
- Estimasi parameter Transfer Function
- Prediksi dan evaluasi model

**Metode ARIMA Transfer Function:**
Model ini menggabungkan deret input (kurs USD/IDR) dengan deret output (harga saham) untuk menghasilkan prediksi yang lebih akurat dengan mempertimbangkan hubungan dinamis antar variabel.
""")

# Pilihan saham
selected_stock = st.sidebar.selectbox(
    "Pilih Saham LQ45:",
    options=list(LQ45_STOCKS.keys()),
    format_func=lambda x: f"{x} - {LQ45_STOCKS[x]}",
    index=2  # Default ke ASII
)

# Pilihan periode
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Tanggal Mulai:",
        value=datetime.now() - timedelta(days=365*2),
        max_value=datetime.now()
    )
with col2:
    end_date = st.date_input(
        "Tanggal Akhir:",
        value=datetime.now(),
        max_value=datetime.now()
    )

# Header utama
st.markdown('<h1 class="main-header">📈 Prediksi Saham LQ45 dengan ARIMA Transfer Function</h1>', unsafe_allow_html=True)

# Tab untuk mengorganisir konten
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Data Historis", 
    "🔍 Preprocessing", 
    "📈 ARIMA Input", 
    "📉 ARIMA Output", 
    "🔗 CCF Analysis", 
    "⚙️ Transfer Function", 
    "🎯 Prediksi"
])

# Tab 1: Data Historis
with tab1:
    st.markdown('<h2 class="section-header">📊 Data Historis dan Statistik Deskriptif</h2>', unsafe_allow_html=True)
    
    if st.button("📥 Ambil Data", key="fetch_data"):
        with st.spinner("Mengambil data saham dan kurs..."):
            # Ambil data saham
            stock_data = get_stock_data(selected_stock, start_date, end_date)
            # Ambil data kurs USD/IDR
            usd_data = get_usd_idr_data(start_date, end_date)
            
            if stock_data is not None and usd_data is not None:
                # Simpan data ke session state
                st.session_state.stock_data = stock_data
                st.session_state.usd_data = usd_data
                st.session_state.selected_stock = selected_stock
                st.success("✅ Data berhasil diambil!")
    
    if 'stock_data' in st.session_state and 'usd_data' in st.session_state:
        stock_data = st.session_state.stock_data
        usd_data = st.session_state.usd_data
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"📈 Data Saham {selected_stock}")
            st.dataframe(stock_data.tail(10))
            
            # Statistik deskriptif saham
            st.subheader("📊 Statistik Deskriptif - Saham")
            close_prices = stock_data['Close'].dropna()
            stats_data = {
                'Metrik': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
                'Nilai': [
                    len(close_prices),
                    close_prices.mean(),
                    close_prices.std(),
                    close_prices.min(),
                    close_prices.quantile(0.25),
                    close_prices.quantile(0.50),
                    close_prices.quantile(0.75),
                    close_prices.max()
                ]
            }
            st.dataframe(pd.DataFrame(stats_data))
        
        with col2:
            st.subheader("💱 Data Kurs USD/IDR")
            st.dataframe(usd_data.tail(10))
            
            # Statistik deskriptif kurs
            st.subheader("📊 Statistik Deskriptif - Kurs")
            usd_close = usd_data['Close'].dropna()
            stats_usd = {
                'Metrik': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
                'Nilai': [
                    len(usd_close),
                    usd_close.mean(),
                    usd_close.std(),
                    usd_close.min(),
                    usd_close.quantile(0.25),
                    usd_close.quantile(0.50),
                    usd_close.quantile(0.75),
                    usd_close.max()
                ]
            }
            st.dataframe(pd.DataFrame(stats_usd))
        
        # Plot data historis
        st.subheader("📊 Visualisasi Data Historis")
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=[f'Harga Saham {selected_stock}', 'Kurs USD/IDR'],
            vertical_spacing=0.1
        )
        
        # Plot saham
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['Close'],
                mode='lines',
                name=f'{selected_stock}',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Plot kurs
        fig.add_trace(
            go.Scatter(
                x=usd_data.index,
                y=usd_data['Close'],
                mode='lines',
                name='USD/IDR',
                line=dict(color='red')
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=600, showlegend=True)
        fig.update_xaxes(title_text="Tanggal", row=2, col=1)
        fig.update_yaxes(title_text="Harga (IDR)", row=1, col=1)
        fig.update_yaxes(title_text="Kurs (IDR)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: Preprocessing
with tab2:
    st.markdown('<h2 class="section-header">🔍 Preprocessing Data</h2>', unsafe_allow_html=True)
    
    if 'stock_data' in st.session_state and 'usd_data' in st.session_state:
        stock_data = st.session_state.stock_data
        usd_data = st.session_state.usd_data
        
        # Persiapkan data
        stock_close = stock_data['Close'].dropna()
        usd_close = usd_data['Close'].dropna()
        
        # Align data berdasarkan tanggal
        combined_data = pd.DataFrame({
            'stock': stock_close,
            'usd': usd_close
        }).dropna()
        
        st.subheader("🔍 Pengecekan Missing Values")
        missing_stock = stock_data.isnull().sum().sum()
        missing_usd = usd_data.isnull().sum().sum()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Missing Values - Saham", missing_stock)
        with col2:
            st.metric("Missing Values - Kurs", missing_usd)
        
        if missing_stock == 0 and missing_usd == 0:
            st.success("✅ Tidak ada missing values")
        else:
            st.warning(f"⚠️ Ditemukan missing values: Saham={missing_stock}, Kurs={missing_usd}")
        
        # Uji stasioneritas
        st.subheader("📊 Uji Stasioneritas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Deret Input (Kurs USD/IDR):**")
            input_stationary = adf_test(combined_data['usd'], "Kurs USD/IDR")
            st.session_state.input_stationary = input_stationary
            
            # Jika tidak stasioner, lakukan differencing
            if not input_stationary:
                st.write("Melakukan differencing...")
                usd_diff = combined_data['usd'].diff().dropna()
                st.write("**Setelah Differencing:**")
                input_stationary_diff = adf_test(usd_diff, "Kurs USD/IDR (Differenced)")
                st.session_state.usd_diff = usd_diff
                st.session_state.input_d = 1
            else:
                st.session_state.input_d = 0
        
        with col2:
            st.write("**Deret Output (Harga Saham):**")
            output_stationary = adf_test(combined_data['stock'], f"Saham {selected_stock}")
            st.session_state.output_stationary = output_stationary
            
            # Jika tidak stasioner, lakukan differencing
            if not output_stationary:
                st.write("Melakukan differencing...")
                stock_diff = combined_data['stock'].diff().dropna()
                st.write("**Setelah Differencing:**")
                output_stationary_diff = adf_test(stock_diff, f"Saham {selected_stock} (Differenced)")
                st.session_state.stock_diff = stock_diff
                st.session_state.output_d = 1
            else:
                st.session_state.output_d = 0
        
        # Transformasi Box-Cox untuk stabilisasi varians
        st.subheader("📦 Transformasi Box-Cox")
        
        try:
            from scipy.stats import boxcox
            
            # Box-Cox untuk saham (harus positif)
            if (combined_data['stock'] > 0).all():
                stock_boxcox, lambda_stock = boxcox(combined_data['stock'])
                st.write(f"**Parameter λ untuk saham:** {lambda_stock:.4f}")
                
                # Uji varians sebelum dan sesudah transformasi
                var_before = combined_data['stock'].var()
                var_after = pd.Series(stock_boxcox).var()
                
                st.write(f"**Varians sebelum transformasi:** {var_before:.4f}")
                st.write(f"**Varians setelah transformasi:** {var_after:.4f}")
                
                st.session_state.stock_boxcox = stock_boxcox
                st.session_state.lambda_stock = lambda_stock
            else:
                st.warning("⚠️ Data saham mengandung nilai negatif, transformasi Box-Cox tidak dapat dilakukan")
                
        except Exception as e:
            st.error(f"Error dalam transformasi Box-Cox: {e}")
        
        # Simpan data yang sudah diproses
        st.session_state.combined_data = combined_data
        st.success("✅ Preprocessing selesai")

# Tab 3: ARIMA Input
with tab3:
    st.markdown('<h2 class="section-header">📈 Model ARIMA Deret Input (Kurs USD/IDR)</h2>', unsafe_allow_html=True)
    
    if 'combined_data' in st.session_state:
        combined_data = st.session_state.combined_data
        
        # Tentukan data yang akan digunakan
        if st.session_state.get('input_d', 0) == 1:
            input_data = st.session_state.get('usd_diff', combined_data['usd'].diff().dropna())
        else:
            input_data = combined_data['usd']
        
        # Plot ACF dan PACF
        st.subheader("📊 Plot ACF dan PACF")
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        plot_acf(input_data.dropna(), ax=axes[0], lags=20)
        axes[0].set_title('Autocorrelation Function (ACF)')
        
        plot_pacf(input_data.dropna(), ax=axes[1], lags=20)
        axes[1].set_title('Partial Autocorrelation Function (PACF)')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Parameter ARIMA
        st.subheader("⚙️ Parameter Model ARIMA")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            p_input = st.slider("Parameter p (AR)", 0, 5, 1, key="p_input")
        
        with col2:
            d_input = st.number_input("Parameter d (I)", 
                                    value=st.session_state.get('input_d', 0), 
                                    min_value=0, max_value=2, 
                                    key="d_input", disabled=True)
        
        with col3:
            q_input = st.slider("Parameter q (MA)", 0, 5, 1, key="q_input")
        
        # Estimasi model ARIMA
        if st.button("🔄 Estimasi Model ARIMA Input", key="fit_arima_input"):
            with st.spinner("Mengestimasi model ARIMA..."):
                try:
                    # Fit model ARIMA
                    arima_input = ARIMA(combined_data['usd'], order=(p_input, d_input, q_input))
                    arima_input_fit = arima_input.fit()
                    
                    # Simpan model
                    st.session_state.arima_input_fit = arima_input_fit
                    st.session_state.arima_input_order = (p_input, d_input, q_input)
                    
                    # Tampilkan hasil
                    st.subheader("📊 Hasil Estimasi Model")
                    st.text(str(arima_input_fit.summary()))
                    
                    # Kriteria informasi
                    st.subheader("📈 Kriteria Informasi")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("AIC", f"{arima_input_fit.aic:.4f}")
                    with col2:
                        st.metric("BIC", f"{arima_input_fit.bic:.4f}")
                    with col3:
                        st.metric("Log-Likelihood", f"{arima_input_fit.llf:.4f}")
                    
                    # Uji residual
                    st.subheader("🔍 Uji Residual")
                    residuals = arima_input_fit.resid
                    
                    # Plot residual
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    
                    # Time series plot
                    axes[0, 0].plot(residuals)
                    axes[0, 0].set_title('Residual Time Series')
                    axes[0, 0].grid(True)
                    
                    # Histogram
                    axes[0, 1].hist(residuals, bins=20, alpha=0.7)
                    axes[0, 1].set_title('Histogram of Residuals')
                    axes[0, 1].grid(True)
                    
                    # ACF residual
                    plot_acf(residuals, ax=axes[1, 0], lags=20)
                    axes[1, 0].set_title('ACF of Residuals')
                    
                    # Q-Q plot
                    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
                    axes[1, 1].set_title('Q-Q Plot')
                    axes[1, 1].grid(True)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Uji Ljung-Box
                    ljung_box_test(residuals, lags=10)
                    
                    # Uji normalitas
                    normality_test(residuals)
                    
                    st.success("✅ Model ARIMA input berhasil diestimasi!")
                    
                except Exception as e:
                    st.error(f"Error dalam estimasi model: {e}")

# Tab 4: ARIMA Output
with tab4:
    st.markdown('<h2 class="section-header">📉 Model ARIMA Deret Output (Harga Saham)</h2>', unsafe_allow_html=True)
    
    if 'combined_data' in st.session_state:
        combined_data = st.session_state.combined_data
        
        # Tentukan data yang akan digunakan
        if st.session_state.get('output_d', 0) == 1:
            output_data = st.session_state.get('stock_diff', combined_data['stock'].diff().dropna())
        else:
            output_data = combined_data['stock']
        
        # Plot ACF dan PACF
        st.subheader("📊 Plot ACF dan PACF")
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        plot_acf(output_data.dropna(), ax=axes[0], lags=20)
        axes[0].set_title('Autocorrelation Function (ACF)')
        
        plot_pacf(output_data.dropna(), ax=axes[1], lags=20)
        axes[1].set_title('Partial Autocorrelation Function (PACF)')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Parameter ARIMA
        st.subheader("⚙️ Parameter Model ARIMA")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            p_output = st.slider("Parameter p (AR)", 0, 5, 1, key="p_output")
        
        with col2:
            d_output = st.number_input("Parameter d (I)", 
                                     value=st.session_state.get('output_d', 0), 
                                     min_value=0, max_value=2, 
                                     key="d_output", disabled=True)
        
        with col3:
            q_output = st.slider("Parameter q (MA)", 0, 5, 1, key="q_output")
        
        # Estimasi model ARIMA
        if st.button("🔄 Estimasi Model ARIMA Output", key="fit_arima_output"):
            with st.spinner("Mengestimasi model ARIMA..."):
                try:
                    # Fit model ARIMA
                    arima_output = ARIMA(combined_data['stock'], order=(p_output, d_output, q_output))
                    arima_output_fit = arima_output.fit()
                    
                    # Simpan model
                    st.session_state.arima_output_fit = arima_output_fit
                    st.session_state.arima_output_order = (p_output, d_output, q_output)
                    
                    # Tampilkan hasil
                    st.subheader("📊 Hasil Estimasi Model")
                    st.text(str(arima_output_fit.summary()))
                    
                    # Kriteria informasi
                    st.subheader("📈 Kriteria Informasi")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("AIC", f"{arima_output_fit.aic:.4f}")
                    with col2:
                                                st.metric("BIC", f"{arima_output_fit.bic:.4f}")
                    with col3:
                        st.metric("Log-Likelihood", f"{arima_output_fit.llf:.4f}")
                    
                    # Uji residual
                    st.subheader("🔍 Uji Residual")
                    residuals = arima_output_fit.resid
                    
                    # Plot residual
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                    
                    # Time series plot
                    axes[0, 0].plot(residuals)
                    axes[0, 0].set_title('Residual Time Series')
                    axes[0, 0].grid(True)
                    
                    # Histogram
                    axes[0, 1].hist(residuals, bins=20, alpha=0.7)
                    axes[0, 1].set_title('Histogram of Residuals')
                    axes[0, 1].grid(True)
                    
                    # ACF residual
                    plot_acf(residuals, ax=axes[1, 0], lags=20)
                    axes[1, 0].set_title('ACF of Residuals')
                    
                    # Q-Q plot
                    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
                    axes[1, 1].set_title('Q-Q Plot')
                    axes[1, 1].grid(True)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Uji Ljung-Box
                    ljung_box_test(residuals, lags=10)
                    
                    # Uji normalitas
                    normality_test(residuals)
                    
                    st.success("✅ Model ARIMA output berhasil diestimasi!")
                    
                except Exception as e:
                    st.error(f"Error dalam estimasi model: {e}")
# Tab 5: CCF Analysis
with tab5:
    st.markdown('<h2 class="section-header">🔗 Cross-Correlation Function (CCF)</h2>', unsafe_allow_html=True)

    if 'arima_input_fit' in st.session_state and 'combined_data' in st.session_state:
        input_resid = st.session_state.arima_input_fit.resid
        output_data = combined_data['stock']

        if st.session_state.get('output_d', 0) == 1:
            output_data = output_data.diff().dropna()

        input_resid = input_resid[-len(output_data):]
        output_data = output_data[-len(input_resid):]

        st.subheader("📊 Plot Cross-Correlation")
        fig, ax = plt.subplots(figsize=(12, 6))
        lag = 20
        ccf_values = ccf(input_resid, output_data)[:lag + 1]
        ax.bar(range(lag + 1), ccf_values)
        ax.axhline(0, color='black', linewidth=1)
        ax.set_title("Cross-Correlation Function (CCF)")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Correlation")
        st.pyplot(fig)

        st.info("✅ Gunakan hasil CCF untuk menentukan lag antara input dan output dalam Transfer Function.")
        st.session_state.ccf_values = ccf_values
# Tab 6: Transfer Function
with tab6:
    st.markdown('<h2 class="section-header">⚙️ Transfer Function Modeling</h2>', unsafe_allow_html=True)

    if 'arima_input_order' in st.session_state and 'combined_data' in st.session_state:
        p, d, q = st.session_state.arima_input_order
        input_series = combined_data['usd']
        output_series = combined_data['stock']

        if st.session_state.get('output_d', 0) == 1:
            output_series = output_series.diff().dropna()

        if st.session_state.get('input_d', 0) == 1:
            input_series = input_series.diff().dropna()

        input_series = input_series[-len(output_series):]
        output_series = output_series[-len(input_series):]

        st.subheader("🔍 Prewhitening")
        input_resid, output_filtered = prewhiten(input_series, output_series, p, d, q)

        st.session_state.input_resid = input_resid
        st.session_state.output_filtered = output_filtered

        st.success("✅ Prewhitening selesai. Siap untuk digunakan dalam modeling Transfer Function.")
# Tab 7: Prediksi
with tab7:
    st.markdown('<h2 class="section-header">🎯 Prediksi & Evaluasi Model</h2>', unsafe_allow_html=True)

    if 'arima_input_fit' in st.session_state and 'arima_output_fit' in st.session_state:
        # Ambil model dan data
        model_input = st.session_state.arima_input_fit
        model_output = st.session_state.arima_output_fit

        forecast_periods = st.slider("🔮 Jumlah Hari ke Depan untuk Prediksi:", min_value=5, max_value=30, value=10)

        st.subheader("📈 Prediksi Input (Kurs USD/IDR)")
        forecast_input = model_input.forecast(steps=forecast_periods)
        st.line_chart(forecast_input)

        st.subheader("📉 Prediksi Output (Harga Saham)")
        forecast_output = model_output.forecast(steps=forecast_periods)
        st.line_chart(forecast_output)

        # Evaluasi model (menggunakan data historis terakhir)
        actual = combined_data['stock'][-forecast_periods:]
        predicted = model_output.predict(start=actual.index[0], end=actual.index[-1])

        mae, rmse, mape = calculate_metrics(actual, predicted)

        st.subheader("📊 Evaluasi Model")
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{mae:.4f}")
        col2.metric("RMSE", f"{rmse:.4f}")
        col3.metric("MAPE (%)", f"{mape:.2f}")

        st.success("✅ Prediksi dan evaluasi selesai.")
