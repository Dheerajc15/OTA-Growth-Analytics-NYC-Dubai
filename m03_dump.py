#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Module 03: A/B Test — Bundle vs Hotel-Only Pricing
=====================================================
Data Source: Simulated Fare Data (numpy) — Data Source #6
Business Q: Does bundling flight+hotel at 12-15% discount lift conversion
            enough to offset the per-booking margin loss?
"""
import sys, os
sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy import stats

from config.settings import FARE_RANGES, AB_TEST_SAMPLE_SIZE, AB_TEST_SEED
from src.analysis.ab_test_engine import (
    generate_ab_test_data, check_sample_ratio_mismatch,
    check_covariate_balance, run_conversion_test,
    run_chi_square_test, bootstrap_conversion_diff,
    analyze_revenue, analyze_segments,
    simulate_sequential_test, compute_required_sample_size,
    project_incremental_revenue,
)

sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.1)
plt.rcParams["figure.figsize"] = (12, 6)

print("✅ Imports loaded")


# In[2]:


# ── Generate A/B test data ──
df = generate_ab_test_data()
df.head(10)


# ## 📋 Test Design
# 
# | Parameter | Value |
# |---|---|
# | **Control (A)** | Hotel-only listing at full price |
# | **Treatment (B)** | Flight+Hotel bundle at 12% discount |
# | **Primary Metric** | Conversion rate (booked / shown) |
# | **Secondary Metrics** | Revenue per visitor, AOV, segment lift |
# | **Sample Size** | 10,000 (5,000 per group) |
# | **Expected Baseline CVR** | ~3.2% (OTA industry average) |
# | **MDE** | 20% relative lift |
# | **Alpha** | 0.05 (two-tailed) |
# | **Power** | 80% |

# In[3]:


# ── Sample Ratio Mismatch check ──
srm = check_sample_ratio_mismatch(df)
pd.DataFrame([srm]).T


# In[5]:


# ── Covariate balance check ──
balance = check_covariate_balance(df)
balance.style.map(
    lambda v: "background-color: #c8e6c9" if v is True else
              "background-color: #ffcdd2" if v is False else "",
    subset=["BALANCED"]
)


# In[7]:


# ── Covariate balance check ──
balance = check_covariate_balance(df)
balance.style.map(
    lambda v: "background-color: #c8e6c9" if v is True else
              "background-color: #ffcdd2" if v is False else "",
    subset=["BALANCED"]
)


# In[8]:


fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Fare class distribution
pd.crosstab(df["GROUP"], df["FARE_CLASS"], normalize="index").plot(
    kind="bar", ax=axes[0], rot=0
)
axes[0].set_title("Fare Class by Group")
axes[0].set_ylabel("Proportion")

# Traveler type distribution
pd.crosstab(df["GROUP"], df["TRAVELER_TYPE"], normalize="index").plot(
    kind="bar", ax=axes[1], rot=0
)
axes[1].set_title("Traveler Type by Group")

# Device distribution
pd.crosstab(df["GROUP"], df["DEVICE"], normalize="index").plot(
    kind="bar", ax=axes[2], rot=0
)
axes[2].set_title("Device by Group")

plt.suptitle("Pre-Test Balance: Group Composition", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("../outputs/figures/m03_pretest_balance.png", dpi=150, bbox_inches="tight")
plt.show()


# In[9]:


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Hotel price distribution by group
for grp, color in [("control", "#1976d2"), ("treatment", "#e64a19")]:
    gdf = df[df["GROUP"] == grp]
    axes[0].hist(gdf["HOTEL_PRICE"], bins=50, alpha=0.5, label=grp.title(), color=color)
axes[0].set_title("Hotel Base Price Distribution")
axes[0].set_xlabel("Price ($)")
axes[0].legend()

# Display price (what user actually sees)
for grp, color in [("control", "#1976d2"), ("treatment", "#e64a19")]:
    gdf = df[df["GROUP"] == grp]
    axes[1].hist(gdf["DISPLAY_PRICE"], bins=50, alpha=0.5, label=grp.title(), color=color)
axes[1].set_title("Display Price (Bundle discount applied)")
axes[1].set_xlabel("Price ($)")
axes[1].legend()

plt.suptitle("Price Distributions: Control vs Treatment", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("../outputs/figures/m03_price_distributions.png", dpi=150, bbox_inches="tight")
plt.show()


# In[10]:


# ── PRIMARY TEST ──
test_result = run_conversion_test(df)


# In[11]:


chi2 = run_chi_square_test(df)
pd.DataFrame([chi2]).T


# In[13]:


bootstrap = bootstrap_conversion_diff(df, n_bootstrap=10000)


# In[14]:


fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(bootstrap["bootstrap_diffs"] * 100, bins=80, color="#7986cb", alpha=0.7, edgecolor="white")
ax.axvline(0, color="red", linestyle="--", linewidth=2, label="No Effect (0)")
ax.axvline(bootstrap["ci_lower"] * 100, color="green", linestyle="--", linewidth=1.5, label=f"95% CI Lower")
ax.axvline(bootstrap["ci_upper"] * 100, color="green", linestyle="--", linewidth=1.5, label=f"95% CI Upper")
ax.axvline(bootstrap["mean_diff"] * 100, color="orange", linewidth=2, label=f"Mean Diff")

ax.set_xlabel("Conversion Rate Difference (percentage points)")
ax.set_ylabel("Frequency")
ax.set_title("Bootstrap Distribution of CVR Difference (Treatment − Control)", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("../outputs/figures/m03_bootstrap_ci.png", dpi=150, bbox_inches="tight")
plt.show()


# In[15]:


revenue = analyze_revenue(df)


# In[16]:


rev_fare = revenue["revenue_by_fare"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# CVR by fare class × group
pivot_cvr = rev_fare.pivot(index="FARE_CLASS", columns="GROUP", values="CVR")
pivot_cvr.plot(kind="bar", ax=axes[0], rot=0, color=["#1976d2", "#e64a19"])
axes[0].set_title("CVR by Fare Class")
axes[0].set_ylabel("Conversion Rate")
axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# RPV by fare class × group
pivot_rpv = rev_fare.pivot(index="FARE_CLASS", columns="GROUP", values="RPV")
pivot_rpv.plot(kind="bar", ax=axes[1], rot=0, color=["#1976d2", "#e64a19"])
axes[1].set_title("Revenue Per Visitor by Fare Class")
axes[1].set_ylabel("RPV ($)")

plt.suptitle("Revenue Breakdown by Fare Class", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("../outputs/figures/m03_revenue_by_fare.png", dpi=150, bbox_inches="tight")
plt.show()


# In[18]:


segments = analyze_segments(df)
segments.style.map(
    lambda v: "background-color: #c8e6c9" if v is True else
              "background-color: #ffcdd2" if v is False else "",
    subset=["SIGNIFICANT"]
)


# In[20]:


# ── Segment Lift Visualization (Improved) ──

fig, ax = plt.subplots(figsize=(14, 7))

# Nicer labels for legend
label_map = {
    "FARE_CLASS": "Fare Class",
    "TRAVELER_TYPE": "Traveler Type",
    "DEVICE": "Device",
}

# Color & marker config
style_map = {
    "FARE_CLASS":     {"marker": "o", "color": "#1976d2"},
    "TRAVELER_TYPE":  {"marker": "D", "color": "#e64a19"},
    "DEVICE":         {"marker": "^", "color": "#388e3c"},
}

# Normalize bubble sizes to a readable range (min 80, max 600)
total_n = segments["N_CONTROL"] + segments["N_TREATMENT"]
size_min, size_max = total_n.min(), total_n.max()
normalized_sizes = 80 + (total_n - size_min) / (size_max - size_min + 1) * 520

for seg_type in ["FARE_CLASS", "TRAVELER_TYPE", "DEVICE"]:
    mask = segments["SEGMENT_TYPE"] == seg_type
    seg = segments[mask]
    sizes = normalized_sizes[mask]
    style = style_map[seg_type]

    # Significant = full opacity; not significant = faded + hollow
    for _, row in seg.iterrows():
        row_size = normalized_sizes.loc[row.name]
        alpha = 1.0 if row["SIGNIFICANT"] else 0.35
        edge_width = 1.5 if row["SIGNIFICANT"] else 2.5

        ax.scatter(
            row["RELATIVE_LIFT_PCT"],
            row["SEGMENT_VALUE"],
            s=row_size,
            marker=style["marker"],
            color=style["color"],
            alpha=alpha,
            edgecolors="white" if row["SIGNIFICANT"] else style["color"],
            linewidths=edge_width,
            zorder=5,
        )

        # Value label to the right of each point
        sig_star = " ✱" if row["SIGNIFICANT"] else ""
        ax.annotate(
            f"{row['RELATIVE_LIFT_PCT']:+.1f}%{sig_star}",
            xy=(row["RELATIVE_LIFT_PCT"], row["SEGMENT_VALUE"]),
            xytext=(12, 0),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold" if row["SIGNIFICANT"] else "normal",
            color=style["color"],
            va="center",
        )

    # Invisible scatter for clean legend entry
    ax.scatter([], [], marker=style["marker"], color=style["color"],
               s=120, label=label_map[seg_type], edgecolors="white", linewidths=1.5)

# Zero-effect reference line
ax.axvline(0, color="#9e9e9e", linestyle="--", linewidth=1.5, alpha=0.7, zorder=1)

# Shading: green for positive lift, red for negative
xlim = ax.get_xlim()
ax.axvspan(0, max(xlim[1], 100), color="#c8e6c9", alpha=0.08, zorder=0)
ax.axvspan(min(xlim[0], -30), 0, color="#ffcdd2", alpha=0.08, zorder=0)

# Axis formatting
ax.set_xlabel("Relative Lift (%)", fontsize=12)
ax.set_title("Bundle Lift by Segment", fontsize=15, fontweight="bold", pad=15)
ax.tick_params(axis="y", labelsize=11)
ax.grid(axis="x", alpha=0.3)

# Legend
legend = ax.legend(
    title="Segment Type", title_fontsize=11, fontsize=10,
    loc="upper left", framealpha=0.9, edgecolor="#cccccc",
)

# Footnote for significance
ax.text(
    0.99, 0.02,
    "✱ = statistically significant (p < 0.05)   |   faded = not significant",
    transform=ax.transAxes, fontsize=9, color="#757575",
    ha="right", va="bottom",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#e0e0e0"),
)

plt.tight_layout()
plt.savefig("../outputs/figures/m03_segment_lift.png", dpi=150, bbox_inches="tight")
plt.show()


# In[21]:


sequential = simulate_sequential_test(df)
sequential.head(20)


# In[22]:


fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(sequential["PCT_COMPLETE"], sequential["P_VALUE"],
        "o-", color="#1976d2", linewidth=2, label="Observed p-value")
ax.plot(sequential["PCT_COMPLETE"], sequential["ALPHA_BOUNDARY"],
        "s--", color="#e64a19", linewidth=1.5, label="O'Brien-Fleming boundary")
ax.axhline(0.05, color="gray", linestyle=":", alpha=0.5, label="α = 0.05")

# Mark early stopping point
stops = sequential[sequential["WOULD_STOP"]]
if not stops.empty:
    first = stops.iloc[0]
    ax.scatter([first["PCT_COMPLETE"]], [first["P_VALUE"]],
               s=200, color="green", zorder=5, marker="*", label="Would stop here")

ax.set_xlabel("% of Data Observed")
ax.set_ylabel("p-value")
ax.set_yscale("log")
ax.set_title("Sequential Monitoring: When Could We Stop Early?", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("../outputs/figures/m03_sequential_test.png", dpi=150, bbox_inches="tight")
plt.show()


# In[23]:


power = compute_required_sample_size()
pd.DataFrame([power]).T


# In[24]:


fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for i, (grp, color, title) in enumerate([
    ("control", "#1976d2", "Control (Hotel-Only)"),
    ("treatment", "#e64a19", "Treatment (Bundle)"),
]):
    gdf = df[df["GROUP"] == grp]
    n_total = len(gdf)
    n_converted = gdf["CONVERTED"].sum()

    stages = ["Impressions", "Converted"]
    values = [n_total, n_converted]
    pcts = [100, n_converted / n_total * 100]

    axes[i].barh(stages[::-1], values[::-1], color=color, alpha=0.8)
    for j, (val, pct) in enumerate(zip(values[::-1], pcts[::-1])):
        axes[i].text(val + 30, j, f"{val:,} ({pct:.1f}%)", va="center", fontsize=12)
    axes[i].set_title(title, fontsize=13, fontweight="bold")
    axes[i].set_xlim(0, n_total * 1.3)

plt.suptitle("Conversion Funnel: Control vs Treatment", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("../outputs/figures/m03_conversion_funnel.png", dpi=150, bbox_inches="tight")
plt.show()


# ## 📊 Executive Summary — M03 A/B Test
# 
# ### Test Result
# - **Bundle pricing significantly increases conversion rate** vs hotel-only
# - The 12% bundle discount drives higher CVR, and the RPV lift confirms the
#   higher volume more than offsets the per-booking discount
# 
# ### Key Findings
# 1. **Primary Metric**: Treatment CVR is significantly higher (p < 0.05)
# 2. **Bootstrap**: 95% CI for the difference excludes zero — robust
# 3. **Revenue**: RPV (Revenue Per Visitor) is higher for bundle group
# 4. **Best Segments**: Economy travelers and mobile users show strongest lift
# 5. **Sequential**: Test could have been stopped early, saving run-time
# 
# ### Recommendation
# 🟢 **SHIP IT** — Roll out bundle pricing to 100% of NYC→Dubai traffic,
# starting with economy fare class where lift is strongest. Monitor RPV
# weekly for 30 days post-rollout.

# In[ ]:




