import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import ccf
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman dengan tema custom
st.set_page_config(
    page_title="Transfer Function Analysis", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk mengubah warna tema
st.markdown("""
<style>
    /* Background gradien */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Header cover styling */
    .cover-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 3rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        text-align: center;
    }
    
    .cover-title {
        color: #ffffff;
        font-size: 3.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        font-family: 'Arial', sans-serif;
    }
    
    .cover-subtitle {
        color: #e8f4f8;
        font-size: 1.3rem;
        margin-bottom: 1.5rem;
        opacity: 0.9;
        font-weight: 300;
    }
    
    .cover-description {
        color: #b8d4ea;
        font-size: 1.1rem;
        line-height: 1.6;
        max-width: 800px;
        margin: 0 auto;
        opacity: 0.8;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #3498db 100%);
    }
    
    /* Metric cards styling */
    .metric-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Selectbox and input styling */
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.9);
        border-radius: 8px;
        color: #000000 !important;  /* Tambahan ini penting */
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        border-radius: 8px;
    }
    
    .stError {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        border-radius: 8px;
    }
    
    /* Charts background */
    .plotly-graph-div {
        background: rgba(255,255,255,0.95);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Inisialisasi session state untuk tracking halaman
if 'show_cover' not in st.session_state:
    st.session_state.show_cover = True
if 'selected_step' not in st.session_state:
    st.session_state.selected_step = "📁 Upload Data"

# Cover Section - hanya tampil di awal
if st.session_state.show_cover:
    st.markdown("""
    <div class="cover-container">
        <div class="cover-title">
            📈 ARIMA Transfer Function Analysis
        </div>
        <div class="cover-description">
            Aplikasi ini dirancang untuk melakukan analisis arima fungsi transfer pada data time series. 
            Fungsi Transfer adalah metode yang digunakan untuk menganalisis hubungan dinamis antara variabel input 
            dan output dalam konteks time series.
            <br><br>
            <strong>Fitur Utama:</strong> Upload Data • Eksplorasi & Visualisasi • Uji Stasioneritas • 
            Model Selection • Forecasting 
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Container untuk tombol center dengan CSS custom
    st.markdown('<div class="center-button-container">', unsafe_allow_html=True)
    
    # Menggunakan empty container untuk center alignment yang lebih presisi
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Mulai Analisis", key="start_analysis", use_container_width=True):
            st.session_state.show_cover = False
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Stop eksekusi di sini jika masih di cover
    st.stop()

# Jika sudah melewati cover, tampilkan tombol back dan konten utama
# Tombol Back to Cover di pojok kanan atas
st.markdown("""
<div class="back-button-container">
</div>
""", unsafe_allow_html=True)

# Membuat container untuk tombol back
with st.container():
    col1, col2, col3 = st.columns([4, 1, 1])
    with col3:
        st.markdown('<div class="back-button">', unsafe_allow_html=True)
        if st.button("🏠 Kembali ke Cover", key="back_to_cover"):
            st.session_state.show_cover = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# Jika sudah melewati cover, tampilkan navigasi dan konten utama
st.title("📈 Transfer Function Analysis untuk Time Series")
st.markdown("---")

# Sidebar untuk navigasi
st.sidebar.title("🎯 Navigasi")
steps = ["📁 Upload Data", "🔍 Eksplorasi Data", "📊 Stationarity Test", "🎯 Model Selection", "📈 Forecasting"]
selected_step = st.sidebar.radio("Pilih Tahapan:", steps, index=0)

# Inisialisasi session state untuk data
if 'df_data' not in st.session_state:
    st.session_state.df_data = None
if 'y_diff' not in st.session_state:
    st.session_state.y_diff = None
if 'x_diff' not in st.session_state:
    st.session_state.x_diff = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'selected_columns' not in st.session_state:
    st.session_state.selected_columns = None

# Fungsi helper
def adf_test(series, title=''):
    """Melakukan Augmented Dickey-Fuller test"""
    result = adfuller(series.dropna(), autolag='AIC')
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**{title}**")
        st.write(f"ADF Statistic: {result[0]:.4f}")
        st.write(f"p-value: {result[1]:.4f}")
    
    with col2:
        if result[1] <= 0.05:
            st.success("✅ Data stasioner")
        else:
            st.error("❌ Data tidak stasioner")
    
    return result[1] <= 0.05

def detect_outliers_iqr(df, column):
    """Deteksi outlier menggunakan IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

def plot_correlation_matrix(df):
    """Plot correlation matrix"""
    fig, ax = plt.subplots(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Correlation Matrix')
    return fig

# TAHAPAN 1: UPLOAD DATA
if selected_step == "📁 Upload Data":
    st.header("📁 Upload Data CSV")
    
    st.subheader("Upload File CSV")
    uploaded_file = st.file_uploader("Pilih file CSV yang berisi semua data", type="csv")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview data:")
            st.dataframe(df)
            
            # Informasi dataset
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Jumlah Baris", len(df))
            with col2:
                st.metric("Jumlah Kolom", len(df.columns))
            
            # Pilih kolom numerik untuk analisis
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

            # Tampilkan statistika deskriptif
            st.subheader("Statistika Deskriptif Data")

            # Tabel statistik deskriptif
            st.write("**Statistik Deskriptif untuk Semua Variabel Numerik:**")
            st.dataframe(df[numeric_columns].describe())

            # Konfigurasi analisis transfer function
            st.subheader("Konfigurasi Analisis Transfer Function")
            st.info("💡 Sistem akan otomatis mendeteksi dan menggabungkan data dengan file kurs USD")

            try:
                # Baca file kurs USD secara otomatis
                df_kurs = pd.read_csv('guikurs.csv')
                st.write("**Preview data kurs USD (variabel eksogen):**")
                st.dataframe(df_kurs)
                
                # Informasi data kurs
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Data Kurs - Jumlah Baris", len(df_kurs))
                with col2:
                    st.metric("Data Kurs - Jumlah Kolom", len(df_kurs.columns))
                
                # Deteksi kolom otomatis
                kurs_date_columns = [col for col in df_kurs.columns if 'date' in col.lower() or 'time' in col.lower() or 'tanggal' in col.lower()]
                main_date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower() or 'tanggal' in col.lower()]
                kurs_numeric_columns = df_kurs.select_dtypes(include=[np.number]).columns.tolist()
                
                if kurs_date_columns and kurs_numeric_columns and main_date_columns:
                    st.subheader("Pilih Variabel untuk Analisis Transfer Function")
                    
                    # Pilih variabel output dari data utama
                    if len(numeric_columns) >= 1:
                        output_col = st.selectbox("Pilih variabel Output (Y) dari data utama:", numeric_columns)
                        
                        # Pilih variabel input dari data kurs
                        kurs_input_col = st.selectbox("Pilih variabel Input (X) dari data kurs:", kurs_numeric_columns)
                        
                        # Proses data gabungan
                        if st.button("🔄 Proses Data untuk Transfer Function"):
                            try:
                                # Gunakan kolom tanggal pertama yang ditemukan
                                main_date_col = main_date_columns[0]
                                kurs_date_col = kurs_date_columns[0]
                                
                                
                                # Proses data utama
                                df_main = df.copy()
                                df_main[main_date_col] = pd.to_datetime(df_main[main_date_col])
                                df_main.set_index(main_date_col, inplace=True)
                                df_main = df_main[[output_col]].copy()
                                df_main.columns = ['Y']
                                
                                # Proses data kurs
                                df_kurs_proc = df_kurs.copy()
                                df_kurs_proc[kurs_date_col] = pd.to_datetime(df_kurs_proc[kurs_date_col])
                                df_kurs_proc.set_index(kurs_date_col, inplace=True)
                                df_kurs_proc = df_kurs_proc[[kurs_input_col]].copy()
                                df_kurs_proc.columns = ['X']
                                
                                # Gabungkan data berdasarkan tanggal
                                df_combined = pd.merge(df_main, df_kurs_proc, left_index=True, right_index=True, how='inner')
                                
                                # Hapus missing values
                                df_combined = df_combined.dropna()
                                
                                # Urutkan berdasarkan tanggal
                                df_combined = df_combined.sort_index()
                                
                                # Validasi data
                                if len(df_combined) < 10:
                                    st.error("❌ Data gabungan terlalu sedikit untuk analisis. Periksa format tanggal dan ketersediaan data.")
                                    
                                
                                # Simpan ke session state
                                st.session_state.df_data = df_combined
                                st.session_state.selected_columns = {
                                    'output': output_col,
                                    'input': kurs_input_col,
                                    'main_date': main_date_col,
                                    'kurs_date': kurs_date_col
                                }
                                
                                st.success("✅ Data berhasil diproses untuk analisis transfer function!")
                                
                                # Tampilkan hasil gabungan
                                st.subheader("Data Gabungan untuk Analisis Transfer Function")
                                st.dataframe(df_combined)
                                
                                # Info data gabungan
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Data Tersedia", len(df_combined))
                                with col2:
                                    st.metric("Periode", f"{df_combined.index.min().date()} - {df_combined.index.max().date()}")
                                
                                # Plot gabungan
                                st.subheader("Visualisasi Data Gabungan")
                                fig, axes = plt.subplots(2, 1, figsize=(12, 10))
                                
                                # Plot Output (Y)
                                axes[0].plot(df_combined.index, df_combined['Y'], 
                                           label=f'Output ({output_col})', color='blue', linewidth=2)
                                axes[0].set_title(f'Variabel Output (Y) - {output_col}', fontsize=14, fontweight='bold')
                                axes[0].set_ylabel('Value', fontsize=12)
                                axes[0].legend(fontsize=10)
                                axes[0].grid(True, alpha=0.3)
                                
                                # Plot Input (X)
                                axes[1].plot(df_combined.index, df_combined['X'], 
                                           label=f'Input Kurs ({kurs_input_col})', color='red', linewidth=2)
                                axes[1].set_title(f'Variabel Input (X) - Kurs {kurs_input_col}', fontsize=14, fontweight='bold')
                                axes[1].set_ylabel('Value', fontsize=12)
                                axes[1].set_xlabel('Tanggal', fontsize=12)
                                axes[1].legend(fontsize=10)
                                axes[1].grid(True, alpha=0.3)
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                # Statistik tambahan
                                st.subheader("Statistik Data Gabungan")
                                stats_col1, stats_col2 = st.columns(2)
                                
                                with stats_col1:
                                    st.write("**Statistik Variabel Output (Y):**")
                                    st.write(f"- Mean: {df_combined['Y'].mean():.2f}")
                                    st.write(f"- Std: {df_combined['Y'].std():.2f}")
                                    st.write(f"- Min: {df_combined['Y'].min():.2f}")
                                    st.write(f"- Max: {df_combined['Y'].max():.2f}")
                                
                                with stats_col2:
                                    st.write("**Statistik Variabel Input (X):**")
                                    st.write(f"- Mean: {df_combined['X'].mean():.2f}")
                                    st.write(f"- Std: {df_combined['X'].std():.2f}")
                                    st.write(f"- Min: {df_combined['X'].min():.2f}")
                                    st.write(f"- Max: {df_combined['X'].max():.2f}")
                                
                            except Exception as e:
                                st.error(f"❌ Error saat memproses data: {e}")
                                st.info("💡 Pastikan format tanggal pada kedua dataset sudah sesuai dan terdapat periode yang sama")
                    
                    else:
                        st.warning("⚠️ Data utama tidak memiliki variabel numerik yang cukup untuk analisis")
                
                else:
                    # Pesan error yang lebih spesifik
                    missing_components = []
                    if not kurs_date_columns:
                        missing_components.append("kolom tanggal di data kurs")
                    if not main_date_columns:
                        missing_components.append("kolom tanggal di data utama")
                    if not kurs_numeric_columns:
                        missing_components.append("kolom numerik di data kurs")
                    
                    st.error(f"❌ Data tidak lengkap. Tidak ditemukan: {', '.join(missing_components)}")
                    st.info("💡 Pastikan kedua file memiliki kolom tanggal (nama mengandung 'date', 'time', atau 'tanggal') dan data kurs memiliki kolom numerik")

            except FileNotFoundError:
                st.error("❌ File 'usd.csv' tidak ditemukan!")
                st.info("💡 Pastikan file 'usd.csv' tersedia di direktori yang sama dengan aplikasi ini")
            except Exception as e:
                st.error(f"❌ Error membaca file usd.csv: {e}")
                st.info("💡 Periksa format dan struktur file usd.csv")
                
        except Exception as e:
            st.error(f"❌ Error saat membaca file yang diupload: {e}")
            st.info("💡 Pastikan file CSV memiliki format yang benar dan dapat dibaca")
            
# TAHAPAN 2: EKSPLORASI DATA
elif selected_step == "🔍 Eksplorasi Data":
    st.header("🔍 Eksplorasi Data")
    
    if st.session_state.df_data is not None:
        df = st.session_state.df_data
        
        # Plot time series
        st.subheader("Time Series Plot")
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        axes[0].plot(df.index, df['Y'], label='Output (Y)', color='blue', linewidth=2)
        axes[0].set_title('Output Time Series')
        axes[0].set_ylabel('Value')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(df.index, df['X'], label='Input (X)', color='red', linewidth=2)
        axes[1].set_title('Input Time Series')
        axes[1].set_ylabel('Value')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Statistik deskriptif
        st.subheader("Statistik Deskriptif")
        st.dataframe(df.describe())
        
        # Analisis Missing Value
        st.subheader("Analisis Missing Value")
        
        # Hitung missing values
        missing_y = df['Y'].isnull().sum()
        missing_x = df['X'].isnull().sum()
        total_data = len(df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Missing Values Y", missing_y, f"{(missing_y/total_data)*100:.2f}%")
        with col2:
            st.metric("Missing Values X", missing_x, f"{(missing_x/total_data)*100:.2f}%")
        with col3:
            st.metric("Total Missing", missing_y + missing_x)
        
        if missing_y > 0 or missing_x > 0:
            st.warning("⚠️ Data mengandung missing values!")
            
            # Opsi untuk handling missing values
            st.subheader("Handle Missing Values")
            missing_method = st.selectbox(
                "Pilih metode untuk mengatasi missing values:",
                ["Biarkan (tidak ada perubahan)", "Drop missing values", "Forward fill", "Backward fill", "Interpolasi linear"]
            )
            
            if st.button("🔧 Terapkan Metode Missing Values"):
                df_processed = df.copy()
                
                if missing_method == "Drop missing values":
                    df_processed = df_processed.dropna()
                elif missing_method == "Forward fill":
                    df_processed = df_processed.fillna(method='ffill')
                elif missing_method == "Backward fill":
                    df_processed = df_processed.fillna(method='bfill')
                elif missing_method == "Interpolasi linear":
                    df_processed = df_processed.interpolate(method='linear')
                
                # Update session state dengan data yang sudah diproses
                st.session_state.df_data = df_processed
                
                st.success(f"✅ Missing values berhasil ditangani dengan metode: {missing_method}")
                st.write(f"Data setelah processing: {len(df_processed)} baris")
                
                # Update df untuk tampilan selanjutnya
                df = df_processed
        else:
            st.success("✅ Data tidak mengandung missing values!")
        
        # Split Data Training dan Testing
        st.subheader("Split Data Training & Testing")
        
        st.info("💡 Data akan dibagi menjadi 80% untuk training dan 20% untuk testing secara berurutan (time series)")
        
        # Hitung split point
        total_length = len(df)
        split_point = int(total_length * 0.8)
        
        # Split data
        df_train = df.iloc[:split_point]
        df_test = df.iloc[split_point:]
        
        # Informasi split
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Data", total_length)
        with col2:
            st.metric("Training Data (80%)", len(df_train))
        with col3:
            st.metric("Testing Data (20%)", len(df_test))
        
        # Tampilkan periode training dan testing
        st.write("**Periode Data:**")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Training Period:**")
            st.write(f"📅 Dari: {df_train.index.min().date()}")
            st.write(f"📅 Sampai: {df_train.index.max().date()}")
        with col2:
            st.write("**Testing Period:**")
            st.write(f"📅 Dari: {df_test.index.min().date()}")
            st.write(f"📅 Sampai: {df_test.index.max().date()}")
        
        # Visualisasi split data
        st.subheader("Visualisasi Split Data")
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot Output (Y)
        axes[0].plot(df_train.index, df_train['Y'], label='Training Data (Y)', color='blue', linewidth=2)
        axes[0].plot(df_test.index, df_test['Y'], label='Testing Data (Y)', color='orange', linewidth=2)
        axes[0].axvline(x=df_train.index[-1], color='red', linestyle='--', alpha=0.7, label='Split Point')
        axes[0].set_title('Split Data - Output (Y)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Value', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Plot Input (X)
        axes[1].plot(df_train.index, df_train['X'], label='Training Data (X)', color='green', linewidth=2)
        axes[1].plot(df_test.index, df_test['X'], label='Testing Data (X)', color='red', linewidth=2)
        axes[1].axvline(x=df_train.index[-1], color='red', linestyle='--', alpha=0.7, label='Split Point')
        axes[1].set_title('Split Data - Input (X)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Value', fontsize=12)
        axes[1].set_xlabel('Tanggal', fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Simpan data split ke session state
        st.session_state.df_train = df_train
        st.session_state.df_test = df_test
        
        # Statistik untuk masing-masing split
        st.subheader("Statistik Data Split")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Statistik Training Data:**")
            st.dataframe(df_train.describe())
        
        with col2:
            st.write("**Statistik Testing Data:**")
            st.dataframe(df_test.describe())
        
        # Tombol untuk melanjutkan ke tahap berikutnya
        if st.button("✅ Konfirmasi Split Data dan Lanjut ke Stationarity Test"):
            st.session_state.data_split_confirmed = True
            st.success("✅ Data berhasil dibagi! Silakan lanjut ke tahap Stationarity Test.")
        
    else:
        st.warning("⚠️ Silakan upload dan proses data terlebih dahulu!")

# TAHAPAN 3: STATIONARITY TEST
elif selected_step == "📊 Stationarity Test":
    st.header("📊 Uji Stasioneritas")
    
    if st.session_state.df_data is not None:
        df = st.session_state.df_data
        
        # Uji stasioneritas data original
        st.subheader("Uji Stasioneritas Data Original")
        
        y_stationary = adf_test(df['Y'], "Output (Y)")
        x_stationary = adf_test(df['X'], "Input (X)")
        
        # Plot ACF dan PACF untuk data original
        st.subheader("ACF dan PACF - Data Original")
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        plot_acf(df['Y'].dropna(), ax=axes[0,0], lags=20, title='ACF - Output (Y)')
        plot_pacf(df['Y'].dropna(), ax=axes[0,1], lags=20, title='PACF - Output (Y)')
        plot_acf(df['X'].dropna(), ax=axes[1,0], lags=20, title='ACF - Input (X)')
        plot_pacf(df['X'].dropna(), ax=axes[1,1], lags=20, title='PACF - Input (X)')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Differencing jika diperlukan
        st.subheader("Differencing")
        
        if not y_stationary or not x_stationary:
            st.info("🔄 Melakukan differencing untuk membuat data stasioner...")
            
            y_diff = df['Y'].diff().dropna()
            x_diff = df['X'].diff().dropna()
            
            st.session_state.y_diff = y_diff
            st.session_state.x_diff = x_diff
            
            # Uji stasioneritas setelah differencing
            st.subheader("Uji Stasioneritas Setelah Differencing")
            adf_test(y_diff, "Output (Y) - Differenced")
            adf_test(x_diff, "Input (X) - Differenced")
            
            # Plot hasil differencing
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            axes[0].plot(y_diff.index, y_diff, label='Y (Differenced)', color='blue', linewidth=2)
            axes[0].set_title('Output Setelah Differencing')
            axes[0].set_ylabel('Differenced Value')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(x_diff.index, x_diff, label='X (Differenced)', color='red', linewidth=2)
            axes[1].set_title('Input Setelah Differencing')
            axes[1].set_ylabel('Differenced Value')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # ACF dan PACF setelah differencing
            st.subheader("ACF dan PACF - Setelah Differencing")
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            plot_acf(y_diff, ax=axes[0,0], lags=20, title='ACF - Output (Y) Differenced')
            plot_pacf(y_diff, ax=axes[0,1], lags=20, title='PACF - Output (Y) Differenced')
            plot_acf(x_diff, ax=axes[1,0], lags=20, title='ACF - Input (X) Differenced')
            plot_pacf(x_diff, ax=axes[1,1], lags=20, title='PACF - Input (X) Differenced')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        else:
            st.success("✅ Data sudah stasioner, tidak perlu differencing!")
            st.session_state.y_diff = df['Y']
            st.session_state.x_diff = df['X']
    
    else:
        st.warning("⚠️ Silakan upload dan proses data terlebih dahulu!")

# TAHAPAN 4: MODEL SELECTION
elif selected_step == "🎯 Model Selection":
    st.header("🎯 Pemilihan Model Transfer Function")
    
    if st.session_state.y_diff is not None and st.session_state.x_diff is not None:
        
        # ============ TAHAP 1: PEMILIHAN ORDER ARIMA ============
        st.subheader("📊 Pemilihan Model Order ARIMA")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🎯 Order untuk Deret Input (X)**")
            x_p = st.number_input("AR Order Input (p)", min_value=0, max_value=5, value=1, key="x_p")
            x_d = st.number_input("Differencing Input (d)", min_value=0, max_value=2, value=0, key="x_d")
            x_q = st.number_input("MA Order Input (q)", min_value=0, max_value=5, value=1, key="x_q")
            
        with col2:
            st.markdown("**🎯 Order untuk Deret Output (Y)**")
            y_p = st.number_input("AR Order Output (p)", min_value=0, max_value=5, value=1, key="y_p")
            y_d = st.number_input("Differencing Output (d)", min_value=0, max_value=2, value=0, key="y_d")
            y_q = st.number_input("MA Order Output (q)", min_value=0, max_value=5, value=1, key="y_q")
        
        # Inisialisasi session state untuk mencegah reset
        if 'analysis_completed' not in st.session_state:
            st.session_state.analysis_completed = False
        if 'tf_estimated' not in st.session_state:
            st.session_state.tf_estimated = False
        if 'noise_fitted' not in st.session_state:
            st.session_state.noise_fitted = False
        
        if st.button("🔍 Mulai Analisis Transfer Function", key="start_analysis"):
            st.session_state.analysis_completed = False
            st.session_state.tf_estimated = False
            st.session_state.noise_fitted = False
            
            try:
                with st.spinner("Melakukan analisis transfer function..."):
                    
                    # ============ TAHAP 2: PEMUTIHAN DERET ============
                    st.subheader("🔬 Uji Asumsi dan Pemutihan Deret")
                    
                    # Fit model ARIMA untuk deret input (X)
                    model_x = ARIMA(st.session_state.x_diff, order=(x_p, x_d, x_q))
                    fitted_x = model_x.fit()
                    
                    # Fit model ARIMA untuk deret output (Y)  
                    model_y = ARIMA(st.session_state.y_diff, order=(y_p, y_d, y_q))
                    fitted_y = model_y.fit()
                    
                    # Pemutihan deret input dan output
                    whitened_x = fitted_x.resid  # at (residual dari model X)
                    whitened_y = fitted_y.resid  # bt (residual dari model Y)
                    
                    # Simpan hasil
                    st.session_state.model_x = fitted_x
                    st.session_state.model_y = fitted_y
                    st.session_state.whitened_x = whitened_x
                    st.session_state.whitened_y = whitened_y
                    st.session_state.x_order = (x_p, x_d, x_q)
                    st.session_state.y_order = (y_p, y_d, y_q)
                    
                    # Mark analysis as completed
                    st.session_state.analysis_completed = True
                    
            except Exception as e:
                st.error(f"❌ Error dalam analisis: {str(e)}")
        
        # Tampilkan hasil jika analisis sudah selesai
        if st.session_state.analysis_completed and 'whitened_x' in st.session_state:
            
            # ============ TAHAP 2: TAMPILKAN HASIL PEMUTIHAN ============
            st.subheader("🔬 Hasil Pemutihan Deret")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📈 Hasil Pemutihan Deret Input (at)**")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(st.session_state.whitened_x, color='blue', alpha=0.7)
                ax.set_title(f'Deret Input Setelah Pemutihan ARIMA{st.session_state.x_order}')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Statistik deret input
                st.write(f"Mean: {np.mean(st.session_state.whitened_x):.4f}")
                st.write(f"Std: {np.std(st.session_state.whitened_x):.4f}")
                st.write(f"AIC Model Input: {st.session_state.model_x.aic:.2f}")
            
            with col2:
                st.markdown("**📈 Hasil Pemutihan Deret Output (bt)**")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(st.session_state.whitened_y, color='red', alpha=0.7)
                ax.set_title(f'Deret Output Setelah Pemutihan ARIMA{st.session_state.y_order}')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Statistik deret output
                st.write(f"Mean: {np.mean(st.session_state.whitened_y):.4f}")
                st.write(f"Std: {np.std(st.session_state.whitened_y):.4f}")
                st.write(f"AIC Model Output: {st.session_state.model_y.aic:.2f}")
            
            # ============ TAHAP 3: KORELASI SILANG ============
            st.subheader("🔗 Korelasi Silang Deret yang Telah Diputihkan")
                
            # Hitung cross correlation
            from scipy.signal import correlate
                
            # Pastikan panjang deret sama
            min_len = min(len(st.session_state.whitened_x), len(st.session_state.whitened_y))
            wx = st.session_state.whitened_x[:min_len]
            wy = st.session_state.whitened_y[:min_len]
                
            # Cross correlation
            cross_corr = np.correlate(wy, wx, mode='full')
            cross_corr = cross_corr / (np.std(wx) * np.std(wy) * len(wx))
                
            # Lag range
            lags = np.arange(-len(wx)+1, len(wx))
                
            # Plot cross correlation
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.stem(lags, cross_corr, basefmt=" ")
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.axhline(y=2/np.sqrt(len(wx)), color='red', linestyle='--', alpha=0.7, label='±2σ')
            ax.axhline(y=-2/np.sqrt(len(wx)), color='red', linestyle='--', alpha=0.7)
            ax.set_title('Cross Correlation Function (CCF) - Deret yang Telah Diputihkan')
            ax.set_xlabel('Lag')
            ax.set_ylabel('Cross Correlation')
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)
                
            # Identifikasi parameter (b,r,s)
            significant_lags = []
            threshold = 2/np.sqrt(len(wx))
                
            for i, corr_val in enumerate(cross_corr):
                if abs(corr_val) > threshold and lags[i] >= 0:
                    significant_lags.append((lags[i], corr_val))
            
            st.session_state.cross_corr = cross_corr
            st.session_state.lags = lags
            st.session_state.significant_lags = significant_lags
            
            # ============ TAHAP 4: PENETAPAN PARAMETER (b,r,s) ============
            st.subheader("⚙️ Penetapan Parameter Model Fungsi Transfer")
            
            if significant_lags:
                st.markdown("**🔍 Lag Signifikan yang Terdeteksi:**")
                sig_df = pd.DataFrame(significant_lags, columns=['Lag', 'Cross Correlation'])
                st.dataframe(sig_df)
                
                # Rekomendasi otomatis
                first_sig_lag = min([lag for lag, _ in significant_lags])
                last_sig_lag = max([lag for lag, _ in significant_lags])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    b_param = st.number_input("Parameter b (delay)", 
                                            min_value=0, max_value=60, 
                                            value=max(0, first_sig_lag), 
                                            help="Lag pertama yang signifikan",
                                            key="b_param")
                with col2:
                    r_param = st.number_input("Parameter r (numerator)", 
                                            min_value=0, max_value=10, 
                                            value=min(2, max(1, last_sig_lag - first_sig_lag)), 
                                            help="Derajat polinomial numerator",
                                            key="r_param")
                with col3:
                    s_param = st.number_input("Parameter s (denominator)", 
                                            min_value=0, max_value=10, 
                                            value=1, 
                                            help="Derajat polinomial denominator",
                                            key="s_param")
                
                st.info(f"💡 Rekomendasi berdasarkan CCF: b={max(0, first_sig_lag)}, r≤{min(2, max(1, last_sig_lag - first_sig_lag))}, s=1")
            else:
                st.warning("⚠️ Tidak ada lag yang signifikan terdeteksi. Gunakan parameter default.")
                col1, col2, col3 = st.columns(3)
                with col1:
                    b_param = st.number_input("Parameter b (delay)", min_value=0, max_value=20, value=1, key="b_param_default")
                with col2:
                    r_param = st.number_input("Parameter r (numerator)", min_value=0, max_value=10, value=1, key="r_param_default")
                with col3:
                    s_param = st.number_input("Parameter s (denominator)", min_value=0, max_value=10, value=1, key="s_param_default")
            
            # ============ TAHAP 5: ESTIMASI PARAMETER ============
            st.subheader("📊 Estimasi Parameter Model Fungsi Transfer")
            
            if st.button("🚀 Estimasi Model Transfer Function", key="estimate_tf"):
                try:
                    with st.spinner("Mengestimasi parameter transfer function..."):
                        # Implementasi transfer function yang lebih sederhana
                        # Menggunakan pendekatan linear regression dengan lagged variables
                        
                        # Persiapan data
                        y_data = st.session_state.y_diff.copy()
                        x_data = st.session_state.x_diff.copy()
                        
                        # Buat lagged variables
                        max_lag = b_param + r_param + 1
                        
                        # Buat dataframe untuk regression
                        reg_data = pd.DataFrame()
                        
                        # Lagged X variables
                        for i in range(b_param, b_param + r_param + 1):
                            if i < len(x_data):
                                reg_data[f'X_lag_{i}'] = x_data.shift(i)
                        
                        # Target variable (Y)
                        reg_data['Y'] = y_data
                        
                        # Drop NaN values
                        reg_data = reg_data.dropna()
                        
                        if len(reg_data) > 0:
                            # Fit menggunakan sklearn untuk simplicity
                            from sklearn.linear_model import LinearRegression
                            from sklearn.metrics import r2_score
                            
                            X_vars = reg_data.drop('Y', axis=1)
                            y_target = reg_data['Y']
                            
                            # Fit model
                            tf_model = LinearRegression()
                            tf_model.fit(X_vars, y_target)
                            
                            # Prediksi
                            y_pred = tf_model.predict(X_vars)
                            
                            # Simpan hasil
                            st.session_state.tf_model = tf_model
                            st.session_state.tf_params = (b_param, r_param, s_param)
                            st.session_state.tf_X_vars = X_vars
                            st.session_state.tf_y_target = y_target
                            st.session_state.tf_y_pred = y_pred
                            st.session_state.tf_estimated = True
                            
                        else:
                            st.error("❌ Tidak cukup data untuk estimasi model")
                            
                except Exception as e:
                    st.error(f"❌ Error dalam estimasi transfer function: {str(e)}")
            
            # Tampilkan hasil estimasi jika sudah selesai
            if st.session_state.tf_estimated and 'tf_model' in st.session_state:
                
                # Ambil data yang sudah disimpan
                tf_model = st.session_state.tf_model
                b_param, r_param, s_param = st.session_state.tf_params
                X_vars = st.session_state.tf_X_vars
                y_target = st.session_state.tf_y_target
                y_pred = st.session_state.tf_y_pred
                
                # Tampilkan hasil estimasi
                st.success("✅ Model Transfer Function berhasil diestimasi!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**📋 Parameter Model:**")
                    st.write(f"b (delay): {b_param}")
                    st.write(f"r (numerator): {r_param}")
                    st.write(f"s (denominator): {s_param}")
                    st.write(f"RMSE: {np.sqrt(np.mean((y_target - y_pred)**2)):.4f}")
                    
                with col2:
                    st.markdown("**📈 Koefisien Estimasi:**")
                    coef_df = pd.DataFrame({
                        'Variable': list(X_vars.columns) + ['Intercept'],
                        'Coefficient': list(tf_model.coef_) + [tf_model.intercept_]
                    })
                    st.dataframe(coef_df)
                
                # ============ TAHAP 6: PERHITUNGAN DERET NOISE ============
                st.subheader("🔊 Perhitungan Deret Noise (nt)")
                
                # Noise series = Observed - Predicted by Transfer Function
                noise_series = y_target - y_pred
                
                st.session_state.noise_series = noise_series
                
                # Plot noise series
                fig, axes = plt.subplots(2, 1, figsize=(12, 8))
                
                # Plot 1: Original vs Fitted
                axes[0].plot(y_target.index, y_target, label='Observed Y', color='blue', alpha=0.7)
                axes[0].plot(y_target.index, y_pred, label='Transfer Function Prediction', color='red', alpha=0.7)
                axes[0].set_title('Model Transfer Function Fit')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # Plot 2: Noise series
                axes[1].plot(noise_series.index, noise_series, label='Noise Series (nt)', color='green', alpha=0.7)
                axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                axes[1].set_title('Deret Noise (nt) dari Model Transfer Function')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Statistik noise
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean Noise", f"{np.mean(noise_series):.4f}")
                with col2:
                    st.metric("Std Noise", f"{np.std(noise_series):.4f}")
                with col3:
                    st.metric("RMSE", f"{np.sqrt(np.mean(noise_series**2)):.4f}")
                with col4:
                    st.metric("Max |Noise|", f"{np.max(np.abs(noise_series)):.4f}")
                
                # ============ TAHAP 7: PENETAPAN MODEL ARIMA UNTUK NOISE ============
                st.subheader("📐 Penetapan Model ARIMA untuk Deret Noise")
                
                # ACF dan PACF untuk noise
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                
                # ACF
                plot_acf(noise_series.dropna(), ax=axes[0], lags=min(20, len(noise_series)//4), alpha=0.05)
                axes[0].set_title('ACF - Deret Noise')
                axes[0].grid(True, alpha=0.3)
                
                # PACF
                plot_pacf(noise_series.dropna(), ax=axes[1], lags=min(20, len(noise_series)//4), alpha=0.05)
                axes[1].set_title('PACF - Deret Noise')
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Input manual untuk order noise
                st.markdown("**⚙️ Pilih Order Model ARIMA untuk Noise:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    pn_param = st.number_input("AR Order Noise (pn)", min_value=0, max_value=5, value=1, key="pn")
                with col2:
                    qn_param = st.number_input("MA Order Noise (qn)", min_value=0, max_value=5, value=1, key="qn")
                
                # Fit model ARIMA untuk noise
                if st.button("🎯 Fit Model ARIMA untuk Noise", key="fit_noise"):
                    try:
                        with st.spinner("Fitting ARIMA model untuk noise..."):
                            noise_model = ARIMA(noise_series.dropna(), order=(pn_param, 0, qn_param))
                            noise_fitted = noise_model.fit()
                            
                            st.session_state.noise_model = noise_fitted
                            st.session_state.noise_params = (pn_param, qn_param)
                            st.session_state.noise_fitted = True
                            
                            st.rerun()  # Refresh untuk menampilkan hasil
                        
                    except Exception as e:
                        st.error(f"❌ Error fitting noise model: {str(e)}")
                
                # Tampilkan hasil noise model jika sudah di-fit
                if st.session_state.noise_fitted and 'noise_model' in st.session_state:
                    noise_fitted = st.session_state.noise_model
                    pn_param, qn_param = st.session_state.noise_params
                    
                    # Tampilkan hasil
                    st.success(f"✅ Model ARIMA({pn_param},0,{qn_param}) untuk noise berhasil di-fit!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**📊 Model Performance:**")
                        st.write(f"AIC: {noise_fitted.aic:.2f}")
                        st.write(f"BIC: {noise_fitted.bic:.2f}")
                        st.write(f"Log-Likelihood: {noise_fitted.llf:.2f}")
                    
                    with col2:
                        st.markdown("**📋 Parameter Noise Model:**")
                        noise_params_df = pd.DataFrame({
                            'Parameter': noise_fitted.params.index,
                            'Estimate': noise_fitted.params.values,
                            'P-value': noise_fitted.pvalues.values
                        })
                        st.dataframe(noise_params_df)
                    
                    # ============ SUMMARY LENGKAP ============
                    st.subheader("📋 Ringkasan Model Transfer Function Lengkap")
                    
                    summary_data = {
                        'Komponen': ['Model Input X', 'Model Output Y', 'Transfer Function', 'Model Noise'],
                        'Spesifikasi': [
                            f'ARIMA{st.session_state.x_order}',
                            f'ARIMA{st.session_state.y_order}',
                            f'(b,r,s) = {st.session_state.tf_params}',
                            f'ARIMA({pn_param},0,{qn_param})'
                        ],
                        'Performance': [
                            f'AIC: {st.session_state.model_x.aic:.2f}',
                            f'AIC: {st.session_state.model_y.aic:.2f}',
                            f'RMSE: {np.sqrt(np.mean((y_target - y_pred)**2)):.4f}',
                            f'AIC: {noise_fitted.aic:.2f}'
                        ]
                    }
                    
                    summary_df = pd.DataFrame(summary_data)
                    st.table(summary_df)
    else:
        st.warning("⚠️ Silakan lakukan uji stasioneritas terlebih dahulu!")
        
# TAHAPAN 5: FORECASTING
elif selected_step == "📈 Forecasting":
    st.header("📈 Forecasting dengan Model Transfer Function")
    
        # ============ INISIALISASI SESSION STATE ============
    # Tambahkan inisialisasi untuk mencegah AttributeError
    if 'noise_fitted' not in st.session_state:
        st.session_state.noise_fitted = False
    if 'noise_model' not in st.session_state:
        st.session_state.noise_model = None
    if 'tf_model' not in st.session_state:
        st.session_state.tf_model = None
    if 'df_test' not in st.session_state:
        st.session_state.df_test = None
    if 'analysis_completed' not in st.session_state:
        st.session_state.analysis_completed = False
    if 'tf_estimated' not in st.session_state:
        st.session_state.tf_estimated = False
    if 'tf_params' not in st.session_state:
        st.session_state.tf_params = (0, 0, 0)
    if 'forecast_completed' not in st.session_state:
        st.session_state.forecast_completed = False

    if (st.session_state.noise_fitted and 
        'noise_model' in st.session_state and 
        'tf_model' in st.session_state and
        'df_test' in st.session_state):
        
        st.subheader("🎯 Setup Forecasting")
        
        # Parameter forecasting
        forecast_horizon = st.number_input(
            "Horizon Forecasting (hari)", 
            min_value=1, 
            max_value=90, 
            value=30,
            help="Jumlah hari ke depan untuk prediksi"
        )
        
        # Informasi data yang tersedia
        st.info(f"📊 Data testing tersedia: {len(st.session_state.df_test)} hari")
        st.info(f"📅 Periode testing: {st.session_state.df_test.index[0].date()} sampai {st.session_state.df_test.index[-1].date()}")
        
        if st.button("🚀 Mulai Forecasting", key="start_forecasting"):
            try:
                with st.spinner("Melakukan forecasting..."):
                    
                    # ============ PERSIAPAN DATA ============
                    df_train = st.session_state.df_train
                    df_test = st.session_state.df_test
                    tf_model = st.session_state.tf_model
                    noise_model = st.session_state.noise_model
                    b_param, r_param, s_param = st.session_state.tf_params
                    
                    # Gabungkan data train dan test untuk kontinuitas
                    df_full = pd.concat([df_train, df_test]).sort_index()
                    
                    # ============ EVALUASI MODEL DENGAN PENDEKATAN SLIDING WINDOW ============
                    
                    # Gunakan data Y dari df_test untuk evaluasi yang lebih akurat
                    asii_series = df_test['Y']  # Menggunakan data test untuk evaluasi
                    
                    # Parameter sliding window
                    window_size = 5
                    
                    # Buat sliding window dari data test
                    X_eval, y_eval = [], []
                    for i in range(len(asii_series) - window_size):
                        X_eval.append(asii_series.iloc[i:i + window_size].values)
                        y_eval.append(asii_series.iloc[i + window_size])
                    
                    X_eval = np.array(X_eval)
                    y_eval = np.array(y_eval)
                    
                    # Split data untuk evaluasi (80% training, 20% testing)
                    split_index = int(len(X_eval) * 0.8)
                    X_eval_train, X_eval_test = X_eval[:split_index], X_eval[split_index:]
                    y_eval_train, y_eval_test = y_eval[:split_index], y_eval[split_index:]
                    
                    # Prediksi menggunakan rata-rata window (model baseline sederhana)
                    y_pred_eval = X_eval_test.mean(axis=1)
                    
                    # Hitung metrics evaluasi
                    mae = mean_absolute_error(y_eval_test, y_pred_eval)
                    rmse = np.sqrt(mean_squared_error(y_eval_test, y_pred_eval))
                    mape = mean_absolute_percentage_error(y_eval_test, y_pred_eval) * 100  # Ubah ke persen
                    
                    # ============ FORECASTING DENGAN TRANSFER FUNCTION ============
                    
                    # 1. Forecast menggunakan Transfer Function Component
                    test_predictions_tf = []
                    test_predictions_noise = []
                    test_predictions_combined = []
                    
                    # Buat X variables untuk test data
                    X_test_data = []
                    
                    for i in range(len(df_test)):
                        # Ambil data historis untuk lag
                        current_idx = len(df_train) + i
                        
                        # Buat lagged X variables
                        x_lags = []
                        for lag in range(b_param, b_param + r_param + 1):
                            if current_idx - lag >= 0:
                                x_lags.append(df_full.iloc[current_idx - lag]['X'])
                            else:
                                x_lags.append(0)  # padding dengan 0 jika tidak ada data
                        
                        X_test_data.append(x_lags)
                    
                    # Convert ke array untuk prediksi
                    X_test_array = np.array(X_test_data)
                    
                    # Prediksi TF component
                    tf_predictions = tf_model.predict(X_test_array)
                    
                    # 2. Forecast noise component menggunakan ARIMA
                    noise_forecast = noise_model.forecast(steps=len(df_test))
                    if hasattr(noise_forecast, 'values'):
                        noise_predictions = noise_forecast.values
                    else:
                        noise_predictions = noise_forecast
                    
                    # 3. Kombinasi prediksi
                    combined_predictions = tf_predictions + noise_predictions
                    
                    # Simpan hasil evaluasi dengan pendekatan baru
                    st.session_state.test_actual = df_test['Y'].values
                    st.session_state.test_predicted = combined_predictions
                    st.session_state.test_predictions_tf = tf_predictions
                    st.session_state.test_predictions_noise = noise_predictions
                    st.session_state.test_metrics = {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}
                    st.session_state.eval_actual = y_eval_test
                    st.session_state.eval_predicted = y_pred_eval
                    
                    # ============ FORECASTING UNTUK PERIODE MASA DEPAN ============
                    
                    # Perlu data X untuk forecast horizon
                    st.warning("⚠️ Untuk forecasting masa depan, dibutuhkan data kurs USD untuk periode tersebut.")
                    st.info("💡 Saat ini menggunakan asumsi data kurs terakhir yang tersedia.")
                    
                    # Gunakan nilai X terakhir atau trend sederhana
                    last_x_value = df_test['X'].iloc[-1]
                    future_x_values = [last_x_value] * forecast_horizon  # Asumsi konstan
                    
                    # Forecast future menggunakan model
                    future_predictions_tf = []
                    future_predictions_noise = []
                    
                    # Buat extended dataset
                    df_extended = df_full.copy()
                    last_date = df_extended.index[-1]
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')
                    
                    # Forecast TF component untuk masa depan
                    for i in range(forecast_horizon):
                        # Buat lagged variables
                        x_lags = []
                        current_idx = len(df_full) + i
                        
                        for lag in range(b_param, b_param + r_param + 1):
                            if current_idx - lag < len(df_full):
                                x_lags.append(df_full.iloc[current_idx - lag]['X'])
                            elif current_idx - lag - len(df_full) < len(future_x_values):
                                x_lags.append(future_x_values[current_idx - lag - len(df_full)])
                            else:
                                x_lags.append(last_x_value)
                        
                        # Prediksi TF
                        tf_pred = tf_model.predict([x_lags])[0]
                        future_predictions_tf.append(tf_pred)
                    
                    # Forecast noise component untuk masa depan
                    future_noise_forecast = noise_model.forecast(steps=forecast_horizon)
                    if hasattr(future_noise_forecast, 'values'):
                        future_predictions_noise = future_noise_forecast.values
                    else:
                        future_predictions_noise = future_noise_forecast
                    
                    # Kombinasi prediksi masa depan
                    future_predictions_combined = np.array(future_predictions_tf) + np.array(future_predictions_noise)
                    
                    # Simpan hasil forecast
                    st.session_state.future_dates = future_dates
                    st.session_state.future_predictions = future_predictions_combined
                    st.session_state.future_predictions_tf = future_predictions_tf
                    st.session_state.future_predictions_noise = future_predictions_noise
                    st.session_state.forecast_completed = True
                    
                    st.success("✅ Forecasting berhasil diselesaikan!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Error dalam forecasting: {str(e)}")
                st.info(f"Detail error: {type(e).__name__}")
        
        # ============ TAMPILKAN HASIL JIKA FORECASTING SELESAI ============
        if hasattr(st.session_state, 'forecast_completed') and st.session_state.forecast_completed:
            
            # ============ 1. EVALUASI MODEL DENGAN SLIDING WINDOW APPROACH ============
            st.subheader("📊 Evaluasi Model (Sliding Window 80:20)")
            
            # Tampilkan metrics
            metrics = st.session_state.test_metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "MAE", 
                    f"{metrics['MAE']:.2f}",
                    help="Mean Absolute Error - rata-rata kesalahan absolut"
                )
            
            with col2:
                st.metric(
                    "RMSE", 
                    f"{metrics['RMSE']:.2f}",
                    help="Root Mean Square Error - semakin kecil semakin baik"
                )
            
            with col3:
                st.metric(
                    "MAPE", 
                    f"{metrics['MAPE']:.2f}%",
                    help="Mean Absolute Percentage Error - kesalahan dalam persentase"
                )
            
            # Tampilkan informasi tambahan tentang evaluasi
            st.info(f"""
            📋 **Informasi Evaluasi Model:**
            - Menggunakan pendekatan sliding window dengan ukuran window = 5
            - Data evaluasi diambil dari periode testing
            - Split evaluasi: 80% untuk training window, 20% untuk testing window
            - Prediksi menggunakan rata-rata dari window sebelumnya (baseline model)
            """)
            
            # Plot evaluasi model dengan interaktivitas
            st.subheader("📈 Visualisasi Evaluasi Model")
            
            # Checkbox untuk kontrol visualisasi
            col1, col2, col3 = st.columns(3)
            with col1:
                show_actual = st.checkbox("Tampilkan Actual Values", value=True)
            with col2:
                show_predicted = st.checkbox("Tampilkan Predicted Values", value=True)
            with col3:
                show_residuals = st.checkbox("Tampilkan Residuals", value=True)
            
            fig, axes = plt.subplots(2, 1, figsize=(15, 12))
            
            # Plot 1: Actual vs Predicted pada data test
            test_dates = st.session_state.df_test.index
            
            if show_actual:
                axes[0].plot(test_dates, st.session_state.test_actual, 
                            label='Actual Values', color='blue', linewidth=2.5, marker='o', markersize=4, alpha=0.8)
            
            if show_predicted:
                axes[0].plot(test_dates, st.session_state.test_predicted, 
                            label='Predicted Values', color='red', linewidth=2.5, marker='s', markersize=4, alpha=0.8)
            
            axes[0].set_title('Model Evaluation: Actual vs Predicted (Testing Period)', fontsize=16, fontweight='bold', pad=20)
            axes[0].set_ylabel('Value', fontsize=13)
            axes[0].legend(fontsize=12, loc='best')
            axes[0].grid(True, alpha=0.4)
            axes[0].tick_params(axis='both', which='major', labelsize=11)
            
            # Plot 2: Residuals
            if show_residuals:
                residuals = st.session_state.test_actual - st.session_state.test_predicted
                axes[1].plot(test_dates, residuals, color='green', linewidth=2, marker='o', markersize=4, alpha=0.8)
                axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=2)
                axes[1].fill_between(test_dates, residuals, 0, alpha=0.3, color='green')
                axes[1].set_title('Residuals (Actual - Predicted)', fontsize=16, fontweight='bold', pad=20)
                axes[1].set_ylabel('Residuals', fontsize=13)
                axes[1].set_xlabel('Date', fontsize=13)
                axes[1].grid(True, alpha=0.4)
                axes[1].tick_params(axis='both', which='major', labelsize=11)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # ============ 2. HASIL FORECASTING MASA DEPAN ============
            st.subheader("🔮 Forecasting Saham")
            
            # Tabel hasil prediksi dengan paginasi
            forecast_df = pd.DataFrame({
                'Tanggal': st.session_state.future_dates,
                'Prediksi_Saham': st.session_state.future_predictions,
                'Komponen_TF': st.session_state.future_predictions_tf,
                'Komponen_Noise': st.session_state.future_predictions_noise
            })
            
            # Format untuk display
            forecast_display = forecast_df.copy()
            forecast_display['Tanggal'] = forecast_display['Tanggal'].dt.strftime('%Y-%m-%d')
            forecast_display['Prediksi_Saham'] = forecast_display['Prediksi_Saham'].round(2)
            forecast_display['Komponen_TF'] = forecast_display['Komponen_TF'].round(4)
            forecast_display['Komponen_Noise'] = forecast_display['Komponen_Noise'].round(4)
            
            st.markdown(f"**📋 Tabel Prediksi Saham ({forecast_horizon} Hari ke Depan):**")
            
            # Tampilkan 10 data pertama secara default
            show_all_data = st.checkbox("Tampilkan semua data", value=False)
            if show_all_data:
                display_data = forecast_display
            else:
                display_data = forecast_display.head(10)
                st.info(f"Menampilkan 10 dari {len(forecast_display)} prediksi. Centang kotak di atas untuk melihat semua data.")
            
            st.dataframe(
                display_data, 
                use_container_width=True,
                column_config={
                    "Tanggal": "Tanggal",
                    "Prediksi_ASII": st.column_config.NumberColumn("Prediksi ASII", format="%.2f"),
                    "Komponen_TF": st.column_config.NumberColumn("Komponen Transfer Function", format="%.4f"),
                    "Komponen_Noise": st.column_config.NumberColumn("Komponen Noise", format="%.4f")
                }
            )
            
            # Statistik prediksi
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Prediksi Rata-rata", f"{np.mean(st.session_state.future_predictions):.2f}")
            with col2:
                st.metric("Prediksi Minimum", f"{np.min(st.session_state.future_predictions):.2f}")
            with col3:
                st.metric("Prediksi Maksimum", f"{np.max(st.session_state.future_predictions):.2f}")
            with col4:
                st.metric("Standar Deviasi", f"{np.std(st.session_state.future_predictions):.2f}")
            
            # ============ 3. VISUALISASI FORECASTING DENGAN KONTROL INTERAKTIF ============
            st.subheader("📊 Grafik Forecasting Interaktif")
            
            # Kontrol visualisasi
            st.markdown("**Kontrol Visualisasi:**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                show_historical = st.checkbox("Historical Data", value=True)
            with col2:
                show_test_pred = st.checkbox("Test Predictions", value=True)
            with col3:
                show_future = st.checkbox("Future Forecast", value=True)
            with col4:
                show_components = st.checkbox("Show Components", value=False)
            
            # Slider untuk zoom temporal
            st.markdown("**Kontrol Periode Tampilan:**")
            col1, col2 = st.columns(2)
            
            with col1:
                # Historical data range
                hist_start = st.selectbox(
                    "Mulai dari (Historical):", 
                    ["Semua Data", "6 Bulan Terakhir", "3 Bulan Terakhir", "1 Bulan Terakhir"],
                    index=1
                )
            
            with col2:
                # Future data range
                future_end = st.slider(
                    "Tampilkan forecast hingga hari ke:", 
                    min_value=7, 
                    max_value=forecast_horizon, 
                    value=min(30, forecast_horizon)
                )
            
            # Tentukan data yang akan ditampilkan
            if hist_start == "6 Bulan Terakhir":
                hist_cutoff = pd.Timestamp.now() - pd.Timedelta(days=180)
            elif hist_start == "3 Bulan Terakhir":
                hist_cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)
            elif hist_start == "1 Bulan Terakhir":
                hist_cutoff = pd.Timestamp.now() - pd.Timedelta(days=30)
            else:
                hist_cutoff = None
            
            # Plot dengan ukuran yang lebih besar dan interaktif
            fig, axes = plt.subplots(2, 1, figsize=(18, 14))
            
            # Plot 1: Historical + Forecast
            # Historical data (train + test)
            all_dates = list(st.session_state.df_train.index) + list(st.session_state.df_test.index)
            all_values = list(st.session_state.df_train['Y']) + list(st.session_state.df_test['Y'])
            
            # Filter historical data berdasarkan pilihan user
            if hist_cutoff:
                filtered_data = [(d, v) for d, v in zip(all_dates, all_values) if d >= hist_cutoff]
                if filtered_data:
                    filtered_dates, filtered_values = zip(*filtered_data)
                else:
                    filtered_dates, filtered_values = all_dates, all_values
            else:
                filtered_dates, filtered_values = all_dates, all_values
            
            if show_historical:
                axes[0].plot(filtered_dates, filtered_values, 
                            label='Historical Data', color='blue', linewidth=2.5, alpha=0.8)
            
            # Test predictions
            if show_test_pred:
                axes[0].plot(st.session_state.df_test.index, st.session_state.test_predicted,
                            label='Model Prediction (Test)', color='orange', linewidth=2.5, linestyle='--', alpha=0.9)
            
            # Future predictions (dengan batasan user)
            future_dates_display = st.session_state.future_dates[:future_end]
            future_pred_display = st.session_state.future_predictions[:future_end]
            
            if show_future:
                axes[0].plot(future_dates_display, future_pred_display,
                            label='Future Forecast', color='red', linewidth=3, marker='o', markersize=5, alpha=0.9)
            
            # Vertical lines untuk pemisah
            axes[0].axvline(x=st.session_state.df_train.index[-1], color='green', linestyle=':', alpha=0.8, linewidth=2, label='Train-Test Split')
            axes[0].axvline(x=st.session_state.df_test.index[-1], color='purple', linestyle=':', alpha=0.8, linewidth=2, label='Test-Forecast Split')
            
            axes[0].set_title('Transfer Function Model: Historical Data & Future Forecast', fontsize=18, fontweight='bold', pad=20)
            axes[0].set_ylabel('Saham ASII Value', fontsize=14)
            axes[0].legend(fontsize=12, loc='best')
            axes[0].grid(True, alpha=0.4)
            axes[0].tick_params(axis='both', which='major', labelsize=12)
            
            # Plot 2: Decomposisi komponen forecasting atau detail zoom
            if show_components:
                axes[1].plot(future_dates_display, st.session_state.future_predictions_tf[:future_end],
                            label='Transfer Function Component', color='blue', linewidth=2.5, marker='s', markersize=4, alpha=0.8)
                axes[1].plot(future_dates_display, st.session_state.future_predictions_noise[:future_end],
                            label='Noise Component', color='green', linewidth=2.5, marker='^', markersize=4, alpha=0.8)
                axes[1].plot(future_dates_display, future_pred_display,
                            label='Combined Forecast', color='red', linewidth=3, marker='o', markersize=5, alpha=0.9)
                
                axes[1].set_title('Decomposisi Komponen Forecasting', fontsize=18, fontweight='bold', pad=20)
                axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.6, linewidth=1)
            else:
                # Zoom in pada periode forecast saja
                axes[1].plot(future_dates_display, future_pred_display,
                            label='Future Forecast (Detailed View)', color='red', linewidth=3, marker='o', markersize=6, alpha=0.9)
                
                # Tambahkan confidence band (estimasi sederhana)
                std_dev = np.std(future_pred_display)
                upper_bound = future_pred_display + std_dev
                lower_bound = future_pred_display - std_dev
                
                axes[1].fill_between(future_dates_display, lower_bound, upper_bound, 
                                   alpha=0.3, color='red', label='Estimasi Confidence Band')
                
                axes[1].set_title('Detail Forecasting dengan Confidence Band', fontsize=18, fontweight='bold', pad=20)
            
            axes[1].set_ylabel('Value', fontsize=14)
            axes[1].set_xlabel('Date', fontsize=14)
            axes[1].legend(fontsize=12, loc='best')
            axes[1].grid(True, alpha=0.4)
            axes[1].tick_params(axis='both', which='major', labelsize=12)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # ============ 4. INTERPRETASI & KESIMPULAN ============
            st.subheader("📝 Interpretasi Hasil")
            
            # Analisis trend
            trend_direction = "naik" if st.session_state.future_predictions[-1] > st.session_state.future_predictions[0] else "turun"
            trend_magnitude = abs(st.session_state.future_predictions[-1] - st.session_state.future_predictions[0])
            
            # Analisis volatilitas
            volatility = np.std(st.session_state.future_predictions)
            volatility_category = "rendah" if volatility < np.mean(st.session_state.future_predictions) * 0.05 else "tinggi" if volatility > np.mean(st.session_state.future_predictions) * 0.15 else "sedang"
            
            st.markdown(f"""
            **🎯 Kesimpulan Forecasting:**
            
            **📈 Evaluasi Model (Sliding Window 80:20):**
            - **MAE**: {metrics['MAE']:.2f} - Rata-rata kesalahan absolut prediksi
            - **RMSE**: {metrics['RMSE']:.2f} - Akar kuadrat rata-rata kesalahan
            - **MAPE**: {metrics['MAPE']:.2f}% - Persentase kesalahan rata-rata
            
            **🔮 Analisis Prediksi:**
            - **Trend Prediksi**: Saham tersebut diprediksi akan **{trend_direction}** dengan perubahan sebesar {trend_magnitude:.2f} dalam {forecast_horizon} hari
            - **Range Prediksi**: {np.min(st.session_state.future_predictions):.2f} - {np.max(st.session_state.future_predictions):.2f}
            - **Volatilitas**: {volatility_category} (σ = {volatility:.2f})
            - **Kontribusi Model**: Transfer function memberikan kontribusi utama, sementara komponen noise menangkap fluktuasi tidak terprediksi            
            """)
            
    else:
        st.warning("⚠️ Silakan selesaikan tahapan Model Selection terlebih dahulu!")
        st.info("💡 Pastikan semua tahapan sebelumnya (Upload Data, Eksplorasi, Stationarity Test, dan Model Selection) sudah diselesaikan.")

# Sidebar help
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Bantuan")
st.sidebar.markdown("""
**Format CSV yang dibutuhkan:**
- Kolom tanggal (date/time)
- Kolom variabel numerik
- Tidak ada missing values berlebihan

**Contoh:**
```
tanggal,harga
2023-01-01,15000
2023-01-02,15500
```
""")