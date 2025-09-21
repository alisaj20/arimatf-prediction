import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import requests
from datetime import datetime, timedelta
import yfinance as yf
warnings.filterwarnings('ignore')

# Import untuk modeling
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import datetime as dt

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Saham LQ45 Indonesia",
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
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .stSelectbox > div > div > select {
        background-color: #ffffff;
    }
    .prediction-table {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    .real-time-indicator {
        background-color: #28a745;
        color: white;
        padding: 0.5rem;
        border-radius: 0.25rem;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Daftar saham LQ45 dengan kode Yahoo Finance
LQ45_STOCKS = {
    'ASII': 'ASII.JK',
    'BBCA': 'BBCA.JK',
    'BMRI': 'BMRI.JK',
    'BBRI': 'BBRI.JK',
    'TLKM': 'TLKM.JK',
    'UNVR': 'UNVR.JK',
    'INDF': 'INDF.JK',
    'ICBP': 'ICBP.JK',
    'KLBF': 'KLBF.JK',
    'GGRM': 'GGRM.JK',
    'SMGR': 'SMGR.JK',
    'PGAS': 'PGAS.JK',
    'ADRO': 'ADRO.JK',
    'ITMG': 'ITMG.JK',
    'PTBA': 'PTBA.JK',
    'INTP': 'INTP.JK',
    'WSKT': 'WSKT.JK',
    'WIKA': 'WIKA.JK',
    'BBTN': 'BBTN.JK',
    'BTPS': 'BTPS.JK'
}

# Fungsi untuk mendapatkan data kurs USD/IDR real-time
@st.cache_data(ttl=300)  # Cache selama 5 menit
def get_realtime_usd_idr():
    try:
        # Menggunakan API Exchange Rate
        url = "https://api.exchangerate-api.com/v4/latest/USD"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if 'rates' in data and 'IDR' in data['rates']:
            current_rate = data['rates']['IDR']
            return current_rate, True
        else:
            # Fallback menggunakan Yahoo Finance
            usd_idr = yf.download('USDIDR=X', period='1d', interval='1d')
            if not usd_idr.empty:
                current_rate = usd_idr['Close'].iloc[-1]
                return current_rate, True
            else:
                return 15800, False  # Default value jika gagal
    except Exception as e:
        st.warning(f"Gagal mengambil kurs real-time: {e}")
        return 15800, False

# Fungsi untuk mendapatkan data saham real-time
@st.cache_data(ttl=300)  # Cache selama 5 menit
def get_stock_data(stock_code, period='1y'):
    try:
        yahoo_code = LQ45_STOCKS.get(stock_code, f"{stock_code}.JK")
        stock_data = yf.download(yahoo_code, period=period)
        
        if not stock_data.empty:
            # Ambil harga Close dan reset index
            stock_data = stock_data['Close'].reset_index()
            stock_data.columns = ['Date', 'Close']
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            return stock_data
        else:
            return None
    except Exception as e:
        st.error(f"Error mengambil data saham {stock_code}: {e}")
        return None

# Fungsi untuk mendapatkan data kurs historis
@st.cache_data(ttl=300)
def get_usd_idr_history(period='1y'):
    try:
        usd_idr = yf.download('USDIDR=X', period=period)
        if not usd_idr.empty:
            usd_idr = usd_idr['Close'].reset_index()
            usd_idr.columns = ['Date', 'Rate']
            usd_idr['Date'] = pd.to_datetime(usd_idr['Date'])
            return usd_idr
        else:
            return None
    except Exception as e:
        st.error(f"Error mengambil data kurs: {e}")
        return None

# Fungsi untuk uji stasioneritas
def adf_test(series, title=''):
    result = adfuller(series.dropna(), autolag='AIC')
    return {
        'title': title,
        'adf_stat': result[0],
        'p_value': result[1],
        'is_stationary': result[1] <= 0.05
    }

# Fungsi prediksi dengan ARIMA Transfer Function
def predict_with_transfer_function(endog, exog, forecast_steps=10, exog_forecast=None):
    try:
        # Model SARIMAX sebagai Transfer Function
        model = SARIMAX(endog, 
                       exog=exog, 
                       order=(1, 1, 1),
                       seasonal_order=(0, 0, 0, 0),
                       trend='n')
        model_fit = model.fit(disp=False)
        
        # Forecast
        if exog_forecast is not None:
            forecast = model_fit.forecast(steps=forecast_steps, exog=exog_forecast)
        else:
            # Gunakan nilai terakhir exog untuk forecast
            last_exog = exog.iloc[-1]
            exog_forecast = np.full(forecast_steps, last_exog)
            forecast = model_fit.forecast(steps=forecast_steps, exog=exog_forecast)
        
        # Confidence interval
        forecast_ci = model_fit.get_forecast(steps=forecast_steps, exog=exog_forecast).conf_int()
        
        return forecast, forecast_ci, model_fit
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None, None, None

# Fungsi untuk menghitung metrik evaluasi
def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }

# Fungsi untuk membuat grafik prediksi
def create_prediction_chart(historical_data, predictions, forecast_ci, stock_name, forecast_dates):
    fig = go.Figure()
    
    # Data historis (60 hari terakhir untuk visualisasi yang lebih baik)
    recent_data = historical_data.tail(60)
    
    # Line historis
    fig.add_trace(go.Scatter(
        x=recent_data.index,
        y=recent_data.values,
        mode='lines',
        name='Data Historis',
        line=dict(color='blue', width=2)
    ))
    
    # Line prediksi
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=predictions,
        mode='lines+markers',
        name='Prediksi',
        line=dict(color='red', width=2, dash='dash'),
        marker=dict(size=6)
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_ci.iloc[:, 1],  # Upper bound
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_ci.iloc[:, 0],  # Lower bound
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.2)',
        name='Confidence Interval (95%)',
        showlegend=True
    ))
    
    fig.update_layout(
        title=f'Prediksi Harga Saham {stock_name}',
        xaxis_title='Tanggal',
        yaxis_title='Harga (IDR)',
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

# Fungsi untuk prediksi kurs otomatis
def predict_usd_idr_trend(usd_data, forecast_days):
    """Prediksi trend kurs USD/IDR berdasarkan data historis"""
    try:
        # Ambil data 30 hari terakhir untuk trend
        recent_rates = usd_data['Rate'].tail(30)
        
        # Hitung moving average dan trend
        ma_short = recent_rates.tail(5).mean()
        ma_long = recent_rates.tail(15).mean()
        
        # Hitung trend linear sederhana
        x = np.arange(len(recent_rates))
        y = recent_rates.values
        trend_slope = np.polyfit(x, y, 1)[0]
        
        # Prediksi kurs untuk forecast_days ke depan
        current_rate = recent_rates.iloc[-1]
        future_rates = [current_rate + trend_slope * i for i in range(1, forecast_days + 1)]
        
        # Tambahkan noise kecil untuk realisme
        noise = np.random.normal(0, current_rate * 0.001, forecast_days)
        future_rates = np.array(future_rates) + noise
        
        return future_rates, trend_slope
    except Exception as e:
        st.warning(f"Gagal memprediksi trend kurs: {e}")
        current_rate = usd_data['Rate'].iloc[-1]
        return np.full(forecast_days, current_rate), 0

# Header utama
st.markdown('<h1 class="main-header">📈 Sistem Prediksi Saham LQ45 Indonesia</h1>', unsafe_allow_html=True)

# Real-time indicator
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f'<div class="real-time-indicator">🔴 LIVE DATA - Updated: {current_time}</div>', unsafe_allow_html=True)

# Sidebar untuk input
st.sidebar.header("⚙️ Pengaturan Prediksi")

# Pilih saham LQ45
selected_stock = st.sidebar.selectbox(
    "Pilih Saham LQ45:",
    list(LQ45_STOCKS.keys()),
    index=0
)

# Parameter prediksi
forecast_days = st.sidebar.slider("Jumlah Hari Prediksi:", 1, 30, 10)

# Opsi periode data
data_period = st.sidebar.selectbox(
    "Periode Data Historis:",
    ['1y', '6mo', '3mo', '1mo'],
    index=0
)

# Tombol untuk refresh data
if st.sidebar.button("🔄 Refresh Data Real-time", type="secondary"):
    st.cache_data.clear()
    st.rerun()

# Tombol untuk menjalankan prediksi
if st.sidebar.button("🔮 Jalankan Prediksi", type="primary"):
    with st.spinner("Mengambil data real-time dan memproses prediksi..."):
        
        # Ambil data real-time
        stock_data = get_stock_data(selected_stock, period=data_period)
        usd_data = get_usd_idr_history(period=data_period)
        current_usd_rate, is_realtime = get_realtime_usd_idr()
        
        if stock_data is not None and usd_data is not None:
            
            # Tampilkan status data real-time
            col1, col2, col3 = st.columns(3)
            with col1:
                status_icon = "🟢" if is_realtime else "🟡"
                st.info(f"{status_icon} Kurs USD/IDR: {current_usd_rate:,.0f}")
            with col2:
                st.info(f"📊 Data Saham: {len(stock_data)} hari")
            with col3:
                st.info(f"💱 Data Kurs: {len(usd_data)} hari")
            
            # Align data berdasarkan tanggal
            stock_data.set_index('Date', inplace=True)
            usd_data.set_index('Date', inplace=True)
            
            # Merge data
            aligned_data = pd.concat([stock_data['Close'], usd_data['Rate']], axis=1, join='inner')
            aligned_data.columns = ['Stock', 'Rate']
            aligned_data = aligned_data.dropna()
            
            if len(aligned_data) > 50:  # Pastikan data cukup
                
                # Split data untuk evaluasi (80% train, 20% test)
                split_idx = int(len(aligned_data) * 0.8)
                train_data = aligned_data.iloc[:split_idx]
                test_data = aligned_data.iloc[split_idx:]
                
                # === BAGIAN 1: DATA HISTORIS ===
                st.header("📊 Data Historis Real-time")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(f"📈 Grafik Harga {selected_stock}")
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Scatter(
                        x=aligned_data.index,
                        y=aligned_data['Stock'],
                        mode='lines',
                        name=selected_stock,
                        line=dict(color='blue', width=1.5)
                    ))
                    fig_hist.update_layout(
                        title=f'Harga Historis {selected_stock}',
                        xaxis_title='Tanggal',
                        yaxis_title='Harga (IDR)',
                        height=400
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    st.subheader("💱 Grafik Kurs USD/IDR")
                    fig_kurs = go.Figure()
                    fig_kurs.add_trace(go.Scatter(
                        x=aligned_data.index,
                        y=aligned_data['Rate'],
                        mode='lines',
                        name='USD/IDR',
                        line=dict(color='green', width=1.5)
                    ))
                    fig_kurs.update_layout(
                        title='Kurs USD/IDR Historis',
                        xaxis_title='Tanggal',
                        yaxis_title='Kurs',
                        height=400
                    )
                    st.plotly_chart(fig_kurs, use_container_width=True)
                
                # Tabel statistik deskriptif
                st.subheader("📋 Statistik Deskriptif")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**{selected_stock}**")
                    stock_stats = aligned_data['Stock'].describe()
                    st.dataframe(stock_stats.round(2))
                
                with col2:
                    st.write("**Kurs USD/IDR**")
                    kurs_stats = aligned_data['Rate'].describe()
                    st.dataframe(kurs_stats.round(2))
                
                # === BAGIAN 2: PREDIKSI ===
                st.header("🔮 Hasil Prediksi dengan ARIMA Transfer Function")
                
                # Prediksi trend kurs otomatis
                future_usd_rates, trend_slope = predict_usd_idr_trend(usd_data, forecast_days)
                
                st.info(f"📈 Trend Kurs USD/IDR: {trend_slope:.2f}/hari | Prediksi otomatis berdasarkan data historis")
                
                # Jalankan prediksi
                forecast, forecast_ci, model_fit = predict_with_transfer_function(
                    aligned_data['Stock'], 
                    aligned_data['Rate'], 
                    forecast_steps=forecast_days,
                    exog_forecast=future_usd_rates
                )
                
                if forecast is not None:
                    # Buat tanggal untuk prediksi
                    last_date = aligned_data.index[-1]
                    forecast_dates = pd.date_range(
                        start=last_date + pd.Timedelta(days=1),
                        periods=forecast_days,
                        freq='D'
                    )
                    
                    # Grafik prediksi
                    st.subheader("📈 Grafik Prediksi")
                    prediction_chart = create_prediction_chart(
                        aligned_data['Stock'], 
                        forecast, 
                        forecast_ci, 
                        selected_stock, 
                        forecast_dates
                    )
                    st.plotly_chart(prediction_chart, use_container_width=True)
                    
                    # Tabel prediksi
                    st.subheader("📋 Tabel Prediksi Detail")
                    
                    # Buat DataFrame untuk tabel prediksi
                    prediction_df = pd.DataFrame({
                        'Tanggal': forecast_dates,
                        'Prediksi Harga': forecast.round(2),
                        'Batas Bawah (95%)': forecast_ci.iloc[:, 0].round(2),
                        'Batas Atas (95%)': forecast_ci.iloc[:, 1].round(2),
                        'Kurs USD/IDR': future_usd_rates.round(2)
                    })
                    
                    # Tambahkan kolom perubahan
                    last_price = aligned_data['Stock'].iloc[-1]
                    prediction_df['Perubahan (%)'] = ((prediction_df['Prediksi Harga'] - last_price) / last_price * 100).round(2)
                    prediction_df['Perubahan Kurs (%)'] = ((prediction_df['Kurs USD/IDR'] - current_usd_rate) / current_usd_rate * 100).round(2)
                    
                    st.dataframe(prediction_df, use_container_width=True)
                    
                    # Ringkasan prediksi
                    st.subheader("📊 Ringkasan Prediksi")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Harga Saat Ini",
                            f"Rp {last_price:,.0f}",
                            delta=None
                        )
                    
                    with col2:
                        pred_tomorrow = forecast.iloc[0]
                        change_tomorrow = ((pred_tomorrow - last_price) / last_price) * 100
                        st.metric(
                            "Prediksi Besok",
                            f"Rp {pred_tomorrow:,.0f}",
                            delta=f"{change_tomorrow:.2f}%"
                        )
                    
                    with col3:
                        pred_end = forecast.iloc[-1]
                        change_end = ((pred_end - last_price) / last_price) * 100
                        st.metric(
                            f"Prediksi {forecast_days} Hari",
                            f"Rp {pred_end:,.0f}",
                            delta=f"{change_end:.2f}%"
                        )
                    
                    with col4:
                        volatility = forecast.std()
                        st.metric(
                            "Volatilitas Prediksi",
                            f"Rp {volatility:,.0f}",
                            delta=None
                        )
                    
                    # === BAGIAN 3: EVALUASI MODEL ===
                    st.header("📊 Evaluasi Model ARIMA Transfer Function")
                    
                    if len(test_data) > 0:
                        # Prediksi pada data test
                        test_forecast, _, _ = predict_with_transfer_function(
                            train_data['Stock'],
                            train_data['Rate'],
                            forecast_steps=len(test_data),
                            exog_forecast=test_data['Rate'].values
                        )
                        
                        if test_forecast is not None:
                            # Hitung metrik evaluasi
                            metrics = calculate_metrics(test_data['Stock'].values, test_forecast)
                            
                            # Tampilkan metrik
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>MAE (Mean Absolute Error)</h3>
                                    <h2 style="color: #1f77b4;">Rp {metrics['MAE']:,.0f}</h2>
                                    <p>Rata-rata kesalahan absolut</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>RMSE (Root Mean Square Error)</h3>
                                    <h2 style="color: #ff7f0e;">Rp {metrics['RMSE']:,.0f}</h2>
                                    <p>Akar rata-rata kesalahan kuadrat</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col3:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>MAPE (Mean Absolute Percentage Error)</h3>
                                    <h2 style="color: #2ca02c;">{metrics['MAPE']:.2f}%</h2>
                                    <p>Rata-rata kesalahan persentase absolut</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Grafik perbandingan aktual vs prediksi
                            st.subheader("📈 Perbandingan Aktual vs Prediksi (Data Test)")
                            
                            comparison_fig = go.Figure()
                            
                            comparison_fig.add_trace(go.Scatter(
                                x=test_data.index,
                                y=test_data['Stock'].values,
                                mode='lines+markers',
                                name='Actual',
                                line=dict(color='blue', width=2),
                                marker=dict(size=4)
                            ))
                            
                            comparison_fig.add_trace(go.Scatter(
                                x=test_data.index,
                                y=test_forecast,
                                mode='lines+markers',
                                name='Predicted',
                                line=dict(color='red', width=2, dash='dash'),
                                marker=dict(size=4)
                            ))
                            
                            comparison_fig.update_layout(
                                title='Perbandingan Nilai Aktual vs Prediksi',
                                xaxis_title='Tanggal',
                                yaxis_title='Harga (IDR)',
                                hovermode='x unified',
                                height=400
                            )
                            
                            st.plotly_chart(comparison_fig, use_container_width=True)
                            
                            # Analisis residual
                            st.subheader("📊 Analisis Residual")
                            
                            residuals = test_data['Stock'].values - test_forecast
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Histogram residual
                                fig_hist_res = go.Figure()
                                fig_hist_res.add_trace(go.Histogram(
                                    x=residuals,
                                    nbinsx=20,
                                    name='Residuals',
                                    marker_color='lightblue'
                                ))
                                fig_hist_res.update_layout(
                                    title='Distribusi Residual',
                                    xaxis_title='Residual',
                                    yaxis_title='Frekuensi',
                                    height=300
                                )
                                st.plotly_chart(fig_hist_res, use_container_width=True)
                            
                            with col2:
                                # Residual vs waktu
                                fig_res_time = go.Figure()
                                fig_res_time.add_trace(go.Scatter(
                                    x=test_data.index,
                                    y=residuals,
                                    mode='markers',
                                    name='Residuals',
                                    marker=dict(color='red', size=4)
                                ))
                                fig_res_time.add_hline(y=0, line_dash="dash", line_color="black")
                                fig_res_time.update_layout(
                                    title='Residual vs Waktu',
                                    xaxis_title='Tanggal',
                                    yaxis_title='Residual',
                                    height=300
                                )
                                st.plotly_chart(fig_res_time, use_container_width=True)
                            
                            # Interpretasi hasil
                            st.subheader("💡 Interpretasi Hasil")
                            
                            # Buat interpretasi berdasarkan MAPE
                            if metrics['MAPE'] < 5:
                                accuracy_level = "Sangat Baik"
                                color = "green"
                                interpretation = "Model menunjukkan akurasi yang sangat tinggi. Prediksi sangat dapat diandalkan."
                            elif metrics['MAPE'] < 10:
                                accuracy_level = "Baik"
                                color = "blue"
                                interpretation = "Model menunjukkan akurasi yang baik. Prediksi dapat digunakan untuk pengambilan keputusan."
                            elif metrics['MAPE'] < 20:
                                accuracy_level = "Cukup"
                                color = "orange"
                                interpretation = "Model menunjukkan akurasi yang cukup. Gunakan prediksi dengan kehati-hatian."
                            else:
                                accuracy_level = "Kurang Baik"
                                color = "red"
                                interpretation = "Model menunjukkan akurasi yang kurang baik. Perlu perbaikan model."
                            
                            st.markdown(f"""
                            **Tingkat Akurasi Model: <span style="color: {color};">{accuracy_level}</span>**
                            
                            **Metrik Evaluasi:**
                            - **MAE**: Rata-rata kesalahan prediksi adalah Rp {metrics['MAE']:,.0f}
                            - **RMSE**: Kesalahan prediksi dengan penalti untuk error besar adalah Rp {metrics['RMSE']:,.0f}
                            - **MAPE**: Model memiliki tingkat kesalahan rata-rata {metrics['MAPE']:.2f}%
                            
                            **Interpretasi:** {interpretation}
                            """, unsafe_allow_html=True)
                            
                            # Rekomendasi trading
                            st.subheader("🎯 Rekomendasi Trading")
                            
                            # Analisis sinyal
                            pred_change = ((pred_end - last_price) / last_price) * 100
                            
                            if pred_change > 5:
                                signal = "BUY"
                                signal_color = "green"
                                recommendation = f"Sinyal BUY - Prediksi kenaikan {pred_change:.2f}% dalam {forecast_days} hari"
                            elif pred_change < -5:
                                signal = "SELL"
                                signal_color = "red"
                                recommendation = f"Sinyal SELL - Prediksi penurunan {pred_change:.2f}% dalam {forecast_days} hari"
                            else:
                                signal = "HOLD"
                                signal_color = "orange"
                                recommendation = f"Sinyal HOLD - Pergerakan sideways {pred_change:.2f}% dalam {forecast_days} hari"
                            
                            st.markdown(f"""
                            <div style="background-color: {signal_color}; color: white; padding: 1rem; border-radius: 0.5rem; text-align: center;">
                                <h2>{signal}</h2>
                                <p style="font-size: 1.2rem;">{recommendation}</p>
                            </div>
                            """, unsafe_allow_html=True)
