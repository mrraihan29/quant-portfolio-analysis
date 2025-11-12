# Analisis Portofolio Kuantitatif & Optimasi Modern Portfolio Theory (MPT)

Repositori ini menyajikan analisis kuantitatif *end-to-end* dalam membangun dan mengoptimalkan portofolio saham berbasis **Modern Portfolio Theory (MPT)**. Proyek ini mencakup tahapan akuisisi data, analisis statistik ilmiah, simulasi numerik, serta optimasi matematis untuk memperoleh portofolio yang paling efisien dalam konteks *risk-return trade-off*.

---

## Fitur Utama

- **Akuisisi Data:**  
  Pengunduhan data harga saham historis melalui **Yahoo Finance** (`yfinance`).

- **Analisis Statistik (EDA):**  
  - Transformasi harga menjadi **Logarithmic Returns** untuk memastikan stasioneritas.  
  - **Uji Augmented Dickey-Fuller (ADF):** membuktikan stasioneritas secara statistik.  
  - **Uji Jarque-Bera (JB):** menguji normalitas distribusi, mengonfirmasi adanya *fat tails* (leptokurtosis).

- **Simulasi Monte Carlo:**  
  Eksekusi **100.000 simulasi portofolio acak** untuk memvisualisasikan *Efficient Frontier*.

- **Optimasi Matematis:**  
  Penerapan **`scipy.optimize`** untuk memperoleh solusi optimal:  
  1. Portofolio **Sharpe Ratio Maksimum** (efisiensi return-to-risk).  
  2. Portofolio **Volatilitas Minimum** (risiko terendah).

---

## Technology Stack

- **Analisis Data:** Python 3.10, Pandas, NumPy  
- **Data Finansial:** yfinance  
- **Statistik & Optimasi:** SciPy, Statsmodels  
- **Visualisasi:** Matplotlib, Seaborn  
- **Lingkungan:** Anaconda  

---

## Struktur Repositori

- `01_Portfolio_Analysis.ipynb` → Notebook Jupyter *end-to-end* (akuisisi data → analisis → optimasi).  
- `.gitignore` → Mengabaikan file lingkungan dan cache.  
- `README.md` → Dokumentasi utama.  
- *(Akan datang)* `main.py` → API FastAPI untuk menyajikan hasil optimasi portofolio.  

---

## Temuan Utama: Portofolio Optimal

Analisis dilakukan terhadap data historis 5 tahun (Nov 2020 – Nov 2025) untuk 7 saham teknologi: **AAPL, AMZN, GOOGL, META, MSFT, NVDA, TSLA**.

### 1. Portofolio Konservatif (Volatilitas Minimum)
- **Tujuan:** Risiko serendah mungkin.  
- **Return Tahunan:** 18.69%  
- **Volatilitas Tahunan:** 23.91%  
- **Alokasi Bobot:**  
  - MSFT: 53.61%  
  - AAPL: 33.19%  
  - GOOGL: 13.20%  

### 2. Portofolio Efisien (Sharpe Ratio Maksimum)
- **Tujuan:** *Return-to-risk* terbaik.  
- **Return Tahunan:** 43.44%  
- **Volatilitas Tahunan:** 40.43%  
- **Alokasi Bobot:**  
  - NVDA: 65.24%  
  - GOOGL: 34.76%  

---

## 🏃 Cara Menjalankan

1. Clone repositori ini:  
   ```bash
   git clone <url-repo>
   cd <nama-folder>
   ```

2. Buat dan aktifkan lingkungan Anaconda:  
   ```bash
   conda create -n quant_portfolio python=3.10 -y
   conda activate quant_portfolio
   ```

3. Instal dependensi:  
   ```bash
   pip install pandas numpy matplotlib seaborn scipy yfinance notebook
   ```

4. Jalankan Jupyter Notebook:  
   ```bash
   jupyter notebook 01_Portfolio_Analysis.ipynb
   ```

---

## Catatan Akademik

Analisis ini menegaskan relevansi **Modern Portfolio Theory (MPT)** dalam konteks saham teknologi.  
- Portofolio konservatif menekankan **preservasi modal** dengan risiko minimal.  
- Portofolio efisien menekankan **maksimalisasi rasio return-to-risk**, sesuai dengan preferensi investor agresif.  

Dengan demikian, hasil penelitian ini menunjukkan bahwa pemilihan portofolio optimal bergantung pada **profil risiko investor**, serta menegaskan keunggulan metode optimasi matematis dibandingkan simulasi Monte Carlo dalam menemukan solusi portofolio yang benar-benar optimal.