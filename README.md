# Experimentation, Segmentation & Demand Forecasting on the JFK–DXB Corridor

End-to-end OTA analytics project for the **JFK (New York) → DXB (Dubai)** travel corridor, combining demand forecasting, funnel diagnostics, pricing experimentation, traveler segmentation, and sentiment intelligence to drive measurable growth decisions.

---

## 🎯 Project Objective

Build a practical decision system to answer:

1. **When should demand be activated?** (forecasting & seasonality)
2. **Where does booking leakage occur?** (funnel diagnostics)
3. **Does bundle pricing outperform hotel-only pricing?** (A/B experimentation)
4. **Which traveler archetypes require differentiated strategy?** (segmentation)
5. **What sentiment themes should shape marketing & merchandising?** (review analytics)

---

## 🧱 Tech Stack

- **Language:** Python
- **Environment:** Jupyter Notebook
- **Core libraries:** pandas, numpy, matplotlib, seaborn, scipy, scikit-learn
- **Methods used:**
  - Time-series benchmarking (naive, seasonal naive, linear trend)
  - Two-proportion z-test, bootstrap CI, SRM diagnostics
  - PCA + KMeans clustering, silhouette & CH diagnostics
  - VADER sentiment scoring + topic keyword extraction
- **Data architecture:** processed data-first pipeline (`data/processed/*`)

---

## 📓 Notebook Coverage

- `notebooks/01_demand_forecasting.ipynb`
- `notebooks/02_booking_funnel.ipynb`
- `notebooks/03_ab_test_pricing.ipynb`
- `notebooks/04_traveler_segmentation.ipynb`
- `notebooks/05_sentiment_marketing.ipynb`

---

## 📊 Key Implementations & Results

## 1) Demand Forecasting (M01)

**Data**
- Source used: `forecast_ready.csv`
- Coverage: **84 months** (`2019-01` to `2025-12`)
- Trend regressors: **5**

**Core findings**
- Peak passenger months: **Jan, Feb, Mar, Oct, Nov, Dec**
- Trough months: **Apr–Sep**
- COVID trough vs pre-COVID avg: **-67.9%**
- Post-COVID (2023+) vs pre-COVID avg: **+28.2%**

**YoY passenger trajectory**
- 2020: **152,334** (**-69.8%** vs 2019)
- 2021: **289,859** (**+90.3%**)
- 2022: **451,026** (**+55.6%**)
- 2023: **548,861** (**+21.7%**)
- 2024: **607,193** (**+10.6%**)
- 2025: **605,877** (**-0.2%**)

**Forecast benchmark (test_periods=6)**
- **Best model:** `seasonal_naive`
- MAE: **4,486.83**
- RMSE: **5,565.30**
- MAPE: **9.40%**

**Action**
- Use **Q1 + Q4 demand windows** for aggressive acquisition and merchandising.
- Plan softer spend in **Apr–Sep**, emphasizing efficiency.

---

## 2) Booking Funnel Analysis (M02)

**Data**
- Source used: `booking_funnel/hotels_enriched.csv`
- Hotels: **202** (NYC **106**, Dubai **96**)
- Bookable share overall: **99.0%**

**Market comparison**
- **Dubai:** 96 listings, 94 bookable, **97.9%** bookability, avg rating **4.43**, avg visibility **55.3**
- **NYC:** 106 listings, 106 bookable, **100.0%** bookability, avg rating **3.97**, avg visibility **53.1**

**Rating distribution**
- Median rating: **Dubai ~4.6**, **NYC ~4.1**

**Trust quadrant (rating × review-volume)**
- **Dubai:** Star Performer 39, Low Signal 34, Hidden Gem 14, Known but Risky 9
- **NYC:** Hidden Gem 28, Star Performer 27, Known but Risky 26, Low Signal 25

**Action**
- Prioritize “Star Performer” and “Hidden Gem” inventory in ranking/placements.
- Apply trust-building UX (reviews, badges, cancellation clarity) to “Known but Risky” inventory.

---

## 3) A/B Test — Bundle vs Hotel-Only Pricing (M03)

**Experiment integrity**
- Total sample: **10,000**
- Control: **5,015**, Treatment: **4,985**
- SRM p-value: **0.7642** (no SRM issue)

**Primary conversion outcome**
- Control CVR: **3.8684%**
- Treatment CVR: **4.4333%**
- Absolute lift: **+0.565 pp**
- Relative lift: **+14.60%**
- z-test p-value: **0.1567**
- 95% CI: **[-0.217 pp, +1.347 pp]**
- Significant at 5%: **No**

**Bootstrap robustness**
- Mean diff: **+0.565 pp**
- 95% CI: **[-0.218 pp, +1.346 pp]**
- Significant: **No**

**Revenue lens**
- Control revenue: **$62,035.31**
- Treatment revenue: **$62,579.74**
- Revenue delta: **+$544.43**
- RPV: **$12.37 → $12.55** (~**+1.49%**)
- AOV: **$319.77 → $283.17**

**Segment directional signals (not statistically significant)**
- Strongest positive directional lift: **Desktop (+1.109 pp; +35.3%)**
- Business fare segment showed negative directional lift

**Action**
- Do **targeted phase-2 testing**, not full rollout:
  - prioritize desktop + economy/premium cohorts
  - redesign treatment for business-fare users
  - increase sample size for power on ~0.5 pp effects

---

## 4) Traveler Segmentation (M04)

**Data & modeling**
- Travelers generated: **2,000**
- Features: **9 curated behavioral features**
- PCA: **9 → 7 components**, **94.6% variance retained**
- K-search range: **2..8**
- Best K by silhouette: **2**
- K=2 metrics: Silhouette **0.2077**, CH **575.5**, Inertia **13225.3**
- Cluster sizes: **{0: 1128, 1: 872}**

**Action**
- Operate at least **2 macro-archetypes** in campaign logic:
  - one likely value/short-lead dominant
  - one likely higher-spend/intent-dense cohort
- Deploy differentiated offers, CRM timing, and merchandising emphasis by cluster.

> Note: Cluster-profile label tables (archetype naming + feature means) were not visible in shared output excerpts.

---

## 5) Sentiment & Marketing Attribution (M05)

**Data quality**
- Hotels analyzed: **202**
- Extracted reviews: **1,005**
- Review split: NYC **530**, Dubai **475**
- Hotels with reviews: **202**
- Avg review length: **782.21 chars**
- Empty reviews: **0.00%**
- VADER available: **True**

**Sentiment distribution**
- Positive: **88.76%**
- Negative: **10.75%**
- Neutral: **0.50%**

**Detected topic families**
- Location
- Service
- Price/Value
- Room Quality
- Food/Dining
- Amenities
- Views/Ambiance

**Action**
- Keep acquisition creative aligned to strongest positive themes.
- Use negative-theme concentrations to prioritize product/listing fixes.
- Close rating–sentiment mismatch items in high-visibility inventory first.

---

## 🧠 Cross-Module Key Findings

1. **Demand is seasonal and shock-sensitive**, but corridor recovery is strong post-2022.
2. **Dubai inventory appears stronger on ratings**, while NYC has broader trust dispersion.
3. **Bundle treatment is directionally positive** on CVR/RPV, but not yet statistically conclusive.
4. **Segmentation confirms non-uniform behavior**, supporting differentiated commercial strategy.
5. **Sentiment is predominantly positive**, with topic-level intelligence usable for conversion-focused messaging.

---

## ✅ Recommended Business Actions

### Immediate (0–30 days)
- Use seasonal calendar triggers for campaign allocation (Q1/Q4 emphasis).
- Ranking boosts for trusted/high-signal listings.
- Launch phase-2 A/B in high-signal cohorts (desktop, economy/premium).

### Near-term (30–90 days)
- Segment-specific landing pages and offer bundles.
- Negative-topic remediation workflow for high-impression properties.
- Add experiment guardrails: CVR, RPV, AOV, and segment-level lift tracking.

### Medium-term (90+ days)
- Uplift modeling for treatment assignment.
- Automated weekly demand + conversion decision dashboards.
- Richer NLP (embeddings/topics) beyond keyword topic buckets.

---

## 📁 Project Structure

```text
OTA-Growth-Analytics-NYC-Dubai/
├─ notebooks/
│  ├─ 01_demand_forecasting.ipynb
│  ├─ 02_booking_funnel.ipynb
│  ├─ 03_ab_test_pricing.ipynb
│  ├─ 04_traveler_segmentation.ipynb
│  └─ 05_sentiment_marketing.ipynb
├─ src/
│  ├─ analysis/
│  ├─ preprocessing/
│  └─ ...
├─ config/
│  └─ settings.py
├─ data/
│  ├─ raw/
│  ├─ seeds/
│  └─ processed/
└─ outputs/
   ├─ figures/
   └─ reports/
```

---

## ⚠️ Notes & Limitations

- Some outputs (deep cluster profile tables and advanced M05 aggregations) were not fully visible in shared excerpts.
- Price-tier completeness is limited (high share of `Unknown`) in hotel source.
- A/B result is promising but underpowered for definitive 5% significance at current observed effect.

---

## 👤 Author

**Dheeraj Choudhary**  
GitHub: [@Dheerajc15](https://github.com/Dheerajc15)