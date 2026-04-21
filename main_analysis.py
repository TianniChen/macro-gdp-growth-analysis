"""
What Drives GDP Growth? A Cross-Country Macroeconomic Analysis
This script fetches data from the World Bank API, cleans it, generates multiple
visualizations, and runs an exploratory linear regression model.
"""

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. Configuration & Setup
# ==========================================
pd.set_option("display.max_columns", 50)
sns.set_theme()

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

COUNTRIES = {
    "USA": "United States",
    "CHN": "China",
    "JPN": "Japan",
    "DEU": "Germany",
    "GBR": "United Kingdom",
    "IND": "India",
    "BRA": "Brazil",
    "CAN": "Canada",
    "AUS": "Australia",
    "KOR": "South Korea",
}

INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "gdp_growth",
    "FP.CPI.TOTL.ZG": "inflation",
    "SL.UEM.TOTL.ZS": "unemployment",
}

START_YEAR = 2010
END_YEAR = 2024


# ==========================================
# 2. Data Fetching Functions
# ==========================================
def fetch_world_bank_indicator(country_code: str, indicator_code: str, start_year: int, end_year: int) -> pd.DataFrame:
    """Fetch specific indicator data for a country from the World Bank API."""
    url = (
        f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}"
        f"?format=json&per_page=2000"
    )

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as e:
        raise RuntimeError(
            f"Failed to fetch data for country={country_code}, indicator={indicator_code}. "
            "Please check your internet connection and the World Bank API."
        ) from e

    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        return pd.DataFrame(columns=["country_code", "country", "year", "indicator_code", "value"])

    rows = []
    for item in payload[1]:
        year = item.get("date")
        value = item.get("value")

        if year is None:
            continue

        year = int(year)
        if start_year <= year <= end_year:
            rows.append(
                {
                    "country_code": country_code,
                    "country": COUNTRIES.get(country_code, country_code),
                    "year": year,
                    "indicator_code": indicator_code,
                    "value": value,
                }
            )

    return pd.DataFrame(rows)


def build_long_dataset() -> pd.DataFrame:
    """Aggregate all fetched data into a single long-format DataFrame."""
    all_frames = []
    for country_code in COUNTRIES:
        for indicator_code in INDICATORS:
            df = fetch_world_bank_indicator(country_code, indicator_code, START_YEAR, END_YEAR)
            all_frames.append(df)

    if not all_frames:
        raise ValueError("No data frames were created from the World Bank API requests.")

    long_df = pd.concat(all_frames, ignore_index=True)
    long_df["indicator_name"] = long_df["indicator_code"].map(INDICATORS)
    return long_df


# ==========================================
# 3. Data Processing Pipeline
# ==========================================
print("Fetching data from World Bank API...")
long_df = build_long_dataset()

# Convert long table to wide panel data (Country-Year index)
panel = (
    long_df.pivot_table(
        index=["country_code", "country", "year"],
        columns="indicator_name",
        values="value",
        aggfunc="first",
    )
    .reset_index()
    .sort_values(["country", "year"])
)
panel.columns.name = None

def get_latest_snapshot(panel_df: pd.DataFrame) -> pd.DataFrame:
    """Extract the most recent available data for each country."""
    snapshots = []
    for country in panel_df["country"].unique():
        subset = panel_df[panel_df["country"] == country].sort_values("year")
        valid = subset.dropna(subset=["gdp_growth", "inflation", "unemployment"], how="all")
        if not valid.empty:
            snapshots.append(valid.iloc[-1])
    latest_df = pd.DataFrame(snapshots).reset_index(drop=True)
    return latest_df.sort_values("gdp_growth", ascending=False)

latest_df = get_latest_snapshot(panel)


# ==========================================
# 4. Visualizations
# ==========================================
print("Generating visualizations...")

# Viz 1: GDP Growth Trends
plt.figure(figsize=(11, 6))
for country in panel["country"].unique():
    subset = panel[panel["country"] == country]
    plt.plot(subset["year"], subset["gdp_growth"], marker="o", linewidth=1.5, label=country)
plt.title("GDP Growth Trends by Country (2010-2024)")
plt.xlabel("Year")
plt.ylabel("GDP Growth (%)")
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "gdp_growth_trends.png", dpi=300)
plt.close()

# Viz 2: Latest GDP Growth Comparison Bar Chart
plt.figure(figsize=(10, 6))
ordered = latest_df.sort_values("gdp_growth", ascending=True)
plt.barh(ordered["country"], ordered["gdp_growth"])
plt.title("Latest Available GDP Growth by Country")
plt.xlabel("GDP Growth (%)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "latest_gdp_growth_bar.png", dpi=300)
plt.close()

# Viz 3: GDP Growth vs Inflation Scatter Plot
plot_df = latest_df.dropna(subset=["inflation", "gdp_growth"])
plt.figure(figsize=(8, 6))
plt.scatter(plot_df["inflation"], plot_df["gdp_growth"])
for _, row in plot_df.iterrows():
    plt.annotate(row["country"], (row["inflation"], row["gdp_growth"]), fontsize=8)
plt.title("GDP Growth vs Inflation (Latest Available Year)")
plt.xlabel("Inflation (%)")
plt.ylabel("GDP Growth (%)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "gdp_vs_inflation_scatter.png", dpi=300)
plt.close()

# Viz 4: Correlation Heatmap
corr_df = panel[["gdp_growth", "inflation", "unemployment"]].corr(numeric_only=True)
plt.figure(figsize=(6, 5))
sns.heatmap(corr_df, annot=True, cmap="Blues", fmt=".2f")
plt.title("Correlation Heatmap of Macroeconomic Indicators")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "macro_correlation_heatmap.png", dpi=300)
plt.close()

# Viz 5: GDP Growth Volatility (Boxplot)
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12, 6))
sns.boxplot(x="country", y="gdp_growth", data=panel,
            palette="Set3", hue="country", legend=False)
plt.title("Distribution and Volatility of GDP Growth (2010-2024)")
plt.xlabel("Country")
plt.ylabel("GDP Growth (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "gdp_volatility_boxplot.png", dpi=300)
plt.show()

# Viz 6: Exploring Okun's Law (Regression Plot)
plt.figure(figsize=(8, 6))
plot_df_unemp = panel.dropna(subset=["unemployment", "gdp_growth"])
sns.regplot(x="unemployment", y="gdp_growth", data=plot_df_unemp,
            scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
plt.title("GDP Growth vs Unemployment (All Years)")
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("GDP Growth (%)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "gdp_vs_unemployment_reg.png", dpi=300)
plt.close()

# Viz 7: USA Macro Dynamics (Dual-Axis)
usa_data = panel[panel["country_code"] == "USA"].dropna(subset=["gdp_growth", "inflation"])
fig, ax1 = plt.subplots(figsize=(10, 6))

color1 = 'tab:blue'
ax1.set_xlabel('Year')
ax1.set_ylabel('GDP Growth (%)', color=color1)
ax1.plot(usa_data['year'], usa_data['gdp_growth'], color=color1, marker='o', linewidth=2, label='GDP Growth')
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = 'tab:red'
ax2.set_ylabel('Inflation (%)', color=color2)
ax2.plot(usa_data['year'], usa_data['inflation'], color=color2, marker='s', linestyle='--', linewidth=2, label='Inflation')
ax2.tick_params(axis='y', labelcolor=color2)

fig.suptitle("Macroeconomic Dynamics in the USA (GDP vs Inflation)", fontsize=14)
fig.tight_layout()
plt.savefig(OUTPUT_DIR / "usa_macro_dynamics.png", dpi=300)
plt.close()


# ==========================================
# 5. Exploratory Modeling: Linear Regression
# ==========================================
print("Running exploratory linear regression...")
model_df = panel.dropna(subset=["gdp_growth", "inflation", "unemployment"]).copy()
X = model_df[["inflation", "unemployment"]]
y = model_df["gdp_growth"]

if model_df.empty:
    raise ValueError("No complete rows are available for regression after dropping missing values.")

model = LinearRegression()
model.fit(X, y)
pred = model.predict(X)

rmse = np.sqrt(mean_squared_error(y, pred))
r2 = r2_score(y, pred)

summary_text = textwrap.dedent(f"""
Linear Regression Summary
=========================
Target variable: GDP growth
Features: inflation, unemployment

Coefficients:
- inflation: {model.coef_[0]:.4f}
- unemployment: {model.coef_[1]:.4f}
Intercept: {model.intercept_:.4f}

Model fit:
- R^2: {r2:.4f}
- RMSE: {rmse:.4f}

Sample size: {len(model_df)}
""").strip()

print("\n" + summary_text)


# ==========================================
# 6. Export Results
# ==========================================
long_df.to_csv(OUTPUT_DIR / "macro_raw_long.csv", index=False)
panel.to_csv(OUTPUT_DIR / "macro_panel.csv", index=False)
latest_df.to_csv(OUTPUT_DIR / "latest_snapshot.csv", index=False)

with open(OUTPUT_DIR / "regression_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary_text)

print(f"\n✅ Analysis complete! All charts and data saved to the '{OUTPUT_DIR}' directory.")