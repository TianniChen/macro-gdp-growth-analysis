# 🌍 World Bank Macroeconomic Analysis: What Drives GDP Growth?

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Data-World_Bank_API-0071BC?style=for-the-badge" alt="World Bank">
  <img src="https://img.shields.io/badge/Analysis-Econometrics-success?style=for-the-badge" alt="Analysis">
</p>

---

## 🎯 1. Problem & Core Intent
**The Problem:** While GDP is the standard measure of national success, its drivers are often obscured by noise. Is economic growth a result of controlled inflation, or is it inextricably linked to labor market stability?  
**The Core Intent:** This project isn't just a static report; it's a **dynamic analytical pipeline**. It aims to "stress-test" classic economic theories across 10 global giants, using real-time data to reveal how different economies respond to the same global pressures (inflation and unemployment) over a 15-year horizon.

## 📊 2. Data
* **Source:** [World Bank Open Data API](https://data.worldbank.org/)
* **Access Date:** April 2026
* **Key Fields:** * `NY.GDP.MKTP.KD.ZG`: GDP growth (annual %)
    * `FP.CPI.TOTL.ZG`: Inflation, consumer prices (annual %)
    * `SL.UEM.TOTL.ZS`: Unemployment, total (% of total labor force)

## 🛠 3. Methods
1.  **Automated ETL:** Direct API integration to bypass static CSV limitations.
2.  **Panel Construction:** Reshaping multi-indicator data into a unified "Country-Year" longitudinal structure.
3.  **Visual Analytics:** Leveraging Boxplots for volatility analysis and Dual-Axis charts for individual economy deep-dives.
4.  **Exploratory Modeling:** Using OLS Regression to quantify the predictive weight of inflation and unemployment on growth.

## 💡 4. Key Findings
* **Okun's Law Re-validation:** The analysis confirms a distinct negative correlation between unemployment and GDP growth across the dataset, proving that for most major economies, labor market health remains the primary "engine" for growth.
* **Economic "Personalities":** Boxplot distributions reveal a sharp divide in stability. Developed economies (e.g., Germany, USA) show a "tight" growth cluster, while emerging markets (e.g., India, Brazil) exhibit high-ceiling but high-volatility growth patterns.
* **The Inflation Paradox:** Contrary to simple theory, the scatter analysis shows that inflation doesn't always hinder growth. However, the USA case study highlights that extreme inflation spikes act as a "hard brake" on GDP, leading to visible stagflationary gaps.
* **Model Limitation as a Discovery:** The regression's residual variance suggests that while macro indicators are foundational, "black swan" events (like pandemic shocks) and structural policies often override traditional economic cycles.

## 🚀 5. How to Run
1.  **Clone:** `git clone https://github.com/TianniChen/macro-gdp-growth-analysis.git`
2.  **Install:** `pip install pandas requests matplotlib seaborn scikit-learn`
3.  **Execute:** `python main_analysis.py`

## 🔗 6. Product Link / Demo
* **Interactive Demo:** [⚡ Run in Google Colab](https://colab.research.google.com/github/TianniChen/macro-gdp-growth-analysis/blob/main/Analysis.ipynb)
* **Static Preview:** [📄 View on NBViewer](https://nbviewer.org/github/TianniChen/macro-gdp-growth-analysis/blob/main/Analysis.ipynb)

## ⚠️ 7. Limitations & Next Steps
* **Lag Effects:** Growth often responds to inflation with a 6-12 month delay, which our current OLS model doesn't capture.
* **Next Steps:** Implementing a **Time-Series Forecasting** model (like ARIMA or Prophet) to predict next year's growth based on current macro trends.

---
*Created by Tianni Chen / 2026 Macro Analysis Series*
