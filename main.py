# --- 1. Imports ---
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime
from typing import List, Dict

# --- 2. FastAPI App Initialization ---
app = FastAPI(
    title="Portfolio Optimization API",
    description="API ini menjalankan MPT untuk menemukan portofolio optimal (Max Sharpe & Min Vol).",
    version="1.0.0"
)

# --- 3. Pydantic Models (Request & Response Types) ---
# Model untuk request: Apa yang kita harapkan dari user
class PortfolioRequest(BaseModel):
    tickers: List[str] = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META']
    years: int = 5
    risk_free_rate: float = 0.02

# Model untuk response: Apa yang akan kita kirim kembali
class PortfolioMetrics(BaseModel):
    Return: float
    Volatility: float
    SharpeRatio: float

class OptimalPortfolio(BaseModel):
    metrics: PortfolioMetrics
    weights: Dict[str, float]

class OptimizationResponse(BaseModel):
    max_sharpe_portfolio: OptimalPortfolio
    min_volatility_portfolio: OptimalPortfolio

# --- 4. Fungsi Pembantu MPT (Dipotong dari Notebook) ---
# Kita gabungkan semua fungsi helper di satu tempat

def get_portfolio_metrics(weights: np.ndarray, mu: pd.Series, Sigma: pd.DataFrame, risk_free_rate: float):
    """Menghitung metrik portofolio (Return, Volatility, Sharpe)"""
    port_return = np.sum(mu * weights)
    port_variance = np.dot(weights.T, np.dot(Sigma, weights))
    port_volatility = np.sqrt(port_variance)
    sharpe_ratio = (port_return - risk_free_rate) / port_volatility
    return port_return, port_volatility, sharpe_ratio

def negative_sharpe_ratio(weights: np.ndarray, mu: pd.Series, Sigma: pd.DataFrame, risk_free_rate: float):
    """Fungsi objektif untuk minimalkan Negative SR"""
    return -get_portfolio_metrics(weights, mu, Sigma, risk_free_rate)[2]

def portfolio_volatility(weights: np.ndarray, mu: pd.Series, Sigma: pd.DataFrame, risk_free_rate: float):
    """Fungsi objektif untuk minimalkan Volatility"""
    return get_portfolio_metrics(weights, mu, Sigma, risk_free_rate)[1]

# --- 5. Fungsi Logika Inti (Core Logic) ---
# Ini adalah "mesin" utama yang menjalankan semua analisis kita

def run_portfolio_optimization(tickers: List[str], years: int, risk_free_rate: float) -> OptimizationResponse:
    
    # 1. Akuisisi Data
    try:
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = (datetime.today() - pd.DateOffset(years=years)).strftime('%Y-%m-%d')
        
        data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
        if data.empty:
            raise ValueError("Tidak ada data yang diunduh. Periksa ticker.")
            
        adj_close_df = data['Close']
        if adj_close_df.isnull().values.any():
            adj_close_df = adj_close_df.dropna()
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error mengunduh data: {e}")

    # 2. Hitung Log Returns & Komponen MPT
    log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()
    
    # Simpan urutan kolom yang benar (alfabetis)
    correct_order = log_returns.columns.to_list()
    num_assets = len(correct_order)
    
    mu = log_returns.mean() * 252
    Sigma = log_returns.cov() * 252

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
        raise HTTPException(status_code=500, detail=f"Error saat optimasi: {e}")

    # 5. Ekstrak & Format Hasil
    if not (opt_sharpe_result.success and opt_vol_result.success):
        raise HTTPException(status_code=500, detail="Optimasi gagal menemukan solusi.")

    # Ekstrak bobot
    opt_sharpe_weights = opt_sharpe_result.x
    opt_vol_weights = opt_vol_result.x

    # Dapatkan metrik
    sharpe_metrics = get_portfolio_metrics(opt_sharpe_weights, mu, Sigma, risk_free_rate)
    vol_metrics = get_portfolio_metrics(opt_vol_weights, mu, Sigma, risk_free_rate)

    # Format output JSON
    max_sharpe_response = OptimalPortfolio(
        metrics=PortfolioMetrics(Return=sharpe_metrics[0], Volatility=sharpe_metrics[1], SharpeRatio=sharpe_metrics[2]),
        weights={ticker: weight for ticker, weight in zip(correct_order, opt_sharpe_weights)}
    )
    
    min_vol_response = OptimalPortfolio(
        metrics=PortfolioMetrics(Return=vol_metrics[0], Volatility=vol_metrics[1], SharpeRatio=vol_metrics[2]),
        weights={ticker: weight for ticker, weight in zip(correct_order, opt_vol_weights)}
    )
    
    return OptimizationResponse(
        max_sharpe_portfolio=max_sharpe_response,
        min_volatility_portfolio=min_vol_response
    )

# --- 6. API Endpoint Definition ---
# Ini adalah "pintu" ke aplikasi kita

@app.post("/optimize", response_model=OptimizationResponse)
async def optimize_portfolio_endpoint(request: PortfolioRequest):
    """
    Menjalankan optimasi MPT berdasarkan daftar ticker dan parameter.
    """
    try:
        results = run_portfolio_optimization(
            tickers=request.tickers,
            years=request.years,
            risk_free_rate=request.risk_free_rate
        )
        return results
    except HTTPException as e:
        # Re-raise HTTPExceptions (seperti 400 atau 500)
        raise e
    except Exception as e:
        # Tangkap error tak terduga lainnya
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# --- 7. Uvicorn run command (untuk development) ---
if __name__ == "__main__":
    import uvicorn
    # Ini memungkinkan kita menjalankan 'python main.py'
    # 'reload=True' akan me-restart server setiap kali kita menyimpan file
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)