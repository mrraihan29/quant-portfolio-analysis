# --- 1. Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime
from typing import List, Tuple, Dict, Optional

# --- 2. Konfigurasi Halaman (Profesional) ---
# Mengatur konfigurasi halaman HARUS menjadi perintah streamlit pertama
st.set_page_config(
    page_title="Optimasi Portofolio MPT",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. Fungsi Logika Optimasi (Diadaptasi dari Notebook/API) ---
# Kita salin logikanya ke sini agar app ini self-contained

def get_portfolio_metrics(weights: np.ndarray, mu: pd.Series, Sigma: pd.DataFrame, risk_free_rate: float) -> Tuple[float, float, float]:
    """Menghitung metrik portofolio (Return, Volatility, Sharpe)"""
    port_return = np.sum(mu * weights)
    port_variance = np.dot(weights.T, np.dot(Sigma, weights))
    port_volatility = np.sqrt(port_variance)
    sharpe_ratio = (port_return - risk_free_rate) / port_volatility
    return port_return, port_volatility, sharpe_ratio

def negative_sharpe_ratio(weights: np.ndarray, mu: pd.Series, Sigma: pd.DataFrame, risk_free_rate: float) -> float:
    """Fungsi objektif untuk minimalkan Negative SR"""
    return -get_portfolio_metrics(weights, mu, Sigma, risk_free_rate)[2]

def portfolio_volatility(weights: np.ndarray, mu: pd.Series, Sigma: pd.DataFrame, risk_free_rate: float) -> float:
    """Fungsi objektif untuk minimalkan Volatility"""
    return get_portfolio_metrics(weights, mu, Sigma, risk_free_rate)[1]

def run_streamlit_optimization(tickers: List[str], years: int, risk_free_rate: float) -> Optional[Dict[str, Dict]]:
    """
    Menjalankan optimasi penuh dan mengembalikan hasil
    atau None jika terjadi error.
    """
    try:
        # 1. Akuisisi Data
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - pd.DateOffset(years=years)).strftime('%Y-%m-%d')
        
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if data.empty:
            st.error(f"Error: Tidak ada data yang diunduh untuk tickers: {tickers}. Periksa simbol.")
            return None
            
        adj_close_df = data['Close']
        if adj_close_df.isnull().values.any():
            adj_close_df = adj_close_df.dropna()
            
    except Exception as e:
        st.error(f"Error saat mengunduh data: {e}")
        return None

    # 2. Hitung Log Returns & Komponen MPT
    try:
        log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()
        if log_returns.empty:
            st.error("Gagal menghitung log returns. Mungkin rentang data terlalu pendek.")
            return None

        correct_order = log_returns.columns.to_list()
        num_assets = len(correct_order)
        mu = log_returns.mean() * 252
        Sigma = log_returns.cov() * 252
    except Exception as e:
        st.error(f"Error saat menghitung komponen MPT: {e}")
        return None

    # 3. Siapkan Optimasi
    constraint = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    initial_guess = np.array(num_assets * [1. / num_assets])

    # 4. Jalankan Optimasi (Max Sharpe & Min Vol)
    try:
        opt_sharpe_result = minimize(
            negative_sharpe_ratio, initial_guess,
            args=(mu, Sigma, risk_free_rate), method='SLSQP',
            bounds=bounds, constraints=constraint
        )
        opt_vol_result = minimize(
            portfolio_volatility, initial_guess,
            args=(mu, Sigma, risk_free_rate), method='SLSQP',
            bounds=bounds, constraints=constraint
        )
    except Exception as e:
        st.error(f"Error saat optimasi: {e}")
        return None

    # 5. Ekstrak & Format Hasil
    if not (opt_sharpe_result.success and opt_vol_result.success):
        st.warning("Optimasi mungkin tidak berhasil menemukan solusi yang konvergen.")

    opt_sharpe_weights = opt_sharpe_result.x
    opt_vol_weights = opt_vol_result.x
    
    sharpe_metrics = get_portfolio_metrics(opt_sharpe_weights, mu, Sigma, risk_free_rate)
    vol_metrics = get_portfolio_metrics(opt_vol_weights, mu, Sigma, risk_free_rate)

    # Siapkan hasil untuk di-return
    results = {
        "max_sharpe": {
            "metrics": {"Return": sharpe_metrics[0], "Volatility": sharpe_metrics[1], "SharpeRatio": sharpe_metrics[2]},
            "weights": {ticker: weight for ticker, weight in zip(correct_order, opt_sharpe_weights)}
        },
        "min_vol": {
            "metrics": {"Return": vol_metrics[0], "Volatility": vol_metrics[1], "SharpeRatio": vol_metrics[2]},
            "weights": {ticker: weight for ticker, weight in zip(correct_order, opt_vol_weights)}
        }
    }
    return results

# --- 4. Tampilan UI Streamlit ---

# Judul Utama
st.title("📈 Dashboard Optimasi Portofolio (Modern Portfolio Theory)")
st.markdown("""
Aplikasi ini menjalankan **Optimasi MPT** untuk menemukan portofolio "terbaik" berdasarkan data historis.
Gunakan panel di sebelah kiri untuk mengkonfigurasi parameter Anda.
""")

# --- Panel Input (Sidebar) ---
st.sidebar.header("⚙️ Parameter Konfigurasi")

# Input Tickers
tickers_input = st.sidebar.text_input(
    "Masukkan Tickers (dipisah koma)", 
    "AAPL,MSFT,GOOGL,AMZN,NVDA,TSLA,META"
)

# Input Tahun
years_input = st.sidebar.slider(
    "Tahun Data Historis", 
    min_value=1, max_value=10, value=5, step=1
)

# Input Risk-Free Rate
rf_input = st.sidebar.slider(
    "Risk-Free Rate (Imbal Hasil Obligasi)", 
    min_value=0.00, max_value=0.10, value=0.02, step=0.005, format="%.3f"
)

# Tombol untuk Menjalankan
run_button = st.sidebar.button("Jalankan Optimasi")


# --- Panel Output (Main Area) ---

if run_button:
    # 1. Proses input
    tickers_list = [ticker.strip().upper() for ticker in tickers_input.split(",")]
    
    if not tickers_list:
        st.warning("Silakan masukkan setidaknya satu ticker.")
    else:
        # 2. Tampilkan spinner saat kalkulasi
        with st.spinner(f"Menjalankan optimasi untuk: {', '.join(tickers_list)}... Ini mungkin memakan waktu beberapa detik."):
            
            # 3. Jalankan logika inti
            optimization_results = run_streamlit_optimization(tickers_list, years_input, rf_input)
        
        # 4. Tampilkan hasil JIKA berhasil
        if optimization_results:
            st.success("Optimasi Berhasil Diselesaikan!")
            
            max_sharpe = optimization_results["max_sharpe"]
            min_vol = optimization_results["min_vol"]
            
            # Buat DataFrame untuk bobot (untuk plotting)
            weights_df = pd.DataFrame({
                "Max Sharpe": max_sharpe["weights"],
                "Min Volatility": min_vol["weights"]
            })
            # Filter bobot yang sangat kecil (noise) agar plot bersih
            weights_df[weights_df < 0.0001] = 0 

            st.header("Hasil Optimasi Portofolio")

            # 5. Tampilkan metrik dalam kolom (tampilan profesional)
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Portofolio Sharpe Ratio Maksimum")
                st.metric(label="Annualized Return", value=f"{max_sharpe['metrics']['Return']:.2%}")
                st.metric(label="Annualized Volatility (Risk)", value=f"{max_sharpe['metrics']['Volatility']:.2%}")
                st.metric(label="Sharpe Ratio", value=f"{max_sharpe['metrics']['SharpeRatio']:.2f}")

            with col2:
                st.subheader("Portofolio Volatilitas Minimum")
                st.metric(label="Annualized Return", value=f"{min_vol['metrics']['Return']:.2%}")
                st.metric(label="Annuald Volatility (Risk)", value=f"{min_vol['metrics']['Volatility']:.2%}")
                st.metric(label="Sharpe Ratio", value=f"{min_vol['metrics']['SharpeRatio']:.2f}")

            # 6. Tampilkan Bar Chart yang Interaktif
            st.header("Alokasi Bobot Portofolio Optimal")
            st.bar_chart(weights_df, height=400)
            
            # 7. Tampilkan tabel bobot
            st.subheader("Detail Bobot (Persentase)")
            st.dataframe(weights_df.map(lambda x: f"{x:.2%}"))

else:
    st.info("Silakan konfigurasikan parameter Anda di sidebar kiri dan tekan 'Jalankan Optimasi'.")