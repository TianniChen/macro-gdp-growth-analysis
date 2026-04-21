# 🌍 What Drives GDP Growth?

### A Cross-Country Macroeconomic Analysis Using World Bank Data

[](https://www.python.org/)
[](https://pandas.pydata.org/)
[](https://data.worldbank.org/)
[](https://opensource.org/licenses/MIT)

## 📌 Project Overview

This project investigates the dynamic relationship between core macroeconomic indicators—**GDP Growth, Inflation, and Unemployment**—across 10 major global economies from 2010 to 2024.

By leveraging the **World Bank API**, this analysis moves beyond static datasets to provide a live-data pipeline that explores economic resilience, the validity of **Okun's Law**, and the impact of inflation spikes on national output.

-----

## 🚀 Interactive Demo
Wanna see the code in action right now? No installation required!
[
![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TianniChen/macro-gdp-analysis/blob/main/Analysis.ipynb
)

---
## 🚀 Key Features

  - **Live Data Integration:** Dynamic fetching of indicators directly from World Bank servers.
  - **Automated Wrangling:** Conversion of raw JSON responses into structured Panel Data (Country-Year).
  - **Advanced Visualizations:** 7 distinct plot types covering trends, volatility, and correlations.
  - **Statistical Modeling:** Exploratory Multiple Linear Regression to quantify indicator impacts.
  - **Scalable Architecture:** Easily adaptable to different countries or timeframes.

-----

## 🛠️ Technical Workflow & API Logic

The heart of this project is the integration with the **World Bank (WB) REST API**.

### 📡 Data Acquisition

The script identifies indicators using unique WB codes:

  * `NY.GDP.MKTP.KD.ZG`: GDP growth (annual %)
  * `FP.CPI.TOTL.ZG`: Inflation, consumer prices (%)
  * `SL.UEM.TOTL.ZS`: Unemployment, total (% of labor force)

### 🔄 Data Pipeline

1.  **Request:** Construct API calls with specific ISO codes (e.g., `USA`, `CHN`).
2.  **Parse:** Handle the WB nested JSON structure (Metadata + Data).
3.  **Pivot:** Reshape "long-format" API data into a "wide-format" panel for analysis.
4.  **Impute:** Clean missing values and generate descriptive statistics.

-----

## 📊 Analytical Insights (The Story)

The visualizations follow a specific economic logic:

1.  **Growth Benchmarking:** Comparing GDP trajectories to identify "Growth Leaders" vs. "Stable Markets."
2.  **Volatility Assessment:** Using **Boxplots** to visualize economic resilience—distinguishing between steady growth and high-fluctuation emerging markets.
3.  **Okun’s Law Validation:** A regression-scatter plot testing the historical rule that higher growth reduces unemployment.
4.  **Dual-Axis Dynamics:** Deep-dives into the USA economy to visualize "Stagflation" risks (high inflation paired with dropping GDP).

-----

## 📦 Installation & Usage

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/macro-gdp-analysis.git
    cd macro-gdp-analysis
    ```

2.  **Install dependencies:**

    ```bash
    pip install pandas requests matplotlib seaborn scikit-learn
    ```

3.  **Run the analysis:**

      - For the script: `python main_analysis.py`
      - For the interactive version: Open `Analysis.ipynb` in Jupyter Notebook.

-----

## 📁 Project Structure

```text
├── main_analysis.py       # Main Python script for the full pipeline
├── Analysis.ipynb         # Step-by-step interactive Notebook
├── outputs/               # Automatically generated charts and CSVs
│   ├── gdp_growth_trends.png
│   ├── macro_correlation_heatmap.png
│   └── regression_summary.txt
└── README.md              # Project documentation
```

-----

## 📝 Conclusion & Limitations

  - **Conclusion:** While inflation and unemployment are significant, they only partially explain GDP growth ($R^2$ values suggest other external factors like trade policy and tech innovation are crucial).
  - **Limitations:** API data for the current year may have reporting lags; Linear models assume relationships are constant, which may not hold during global crises.

-----

## 🤝 Contributing

Contributions, issues, and feature requests are welcome\! Feel free to check the [issues page](https://www.google.com/search?q=https://github.com/your-username/macro-gdp-analysis/issues).

**Author:** Tianni Chen  

-----

*Disclaimer: This project is for educational and research purposes. Data is sourced from the World Bank and is subject to their terms of use.*
