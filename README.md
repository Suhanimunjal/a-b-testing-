# A/B Testing Analysis: Marketing Campaign Effectiveness

## Business Problem

A company ran a marketing campaign where users were randomly assigned to one of two groups:

- **Ad group (`ad`)**: Users were shown advertisements.
- **Public Service Announcement group (`psa`)**: Users were shown public service announcements instead of ads (control group).

The key business question: **Did the ad campaign significantly increase user conversions compared to the control group?**

If the answer is yes, the business can justify continued ad spend. If not, the ad budget may be better allocated elsewhere.

---

## Dataset

| Column | Description |
|---|---|
| `user id` | Unique identifier for each user |
| `test group` | `ad` (treatment) or `psa` (control) |
| `converted` | Whether the user converted (`True` / `False`) |
| `total ads` | Total number of ads the user was shown |
| `most ads day` | Day of the week when the user saw the most ads |
| `most ads hour` | Hour of the day when the user saw the most ads |

**Source**: `data/raw/marketing_AB.csv`

---

## Methodology

### Why Chi-Square Test (Not a t-Test)?

This is the most important statistical decision in the project — and a common interview question.

**The outcome variable (`converted`) is binary** — each user either converted (`True`) or didn't (`False`). This makes the data *categorical*, not continuous.

| Test | When to Use | Why It Doesn't / Does Apply Here |
|---|---|---|
| **t-test** | Compares *means* of a continuous variable between two groups (e.g., average revenue, time on site) | ❌ You *can* compute a mean of 0/1 values (which gives you a proportion), but the t-test assumes the data is approximately normally distributed. Binary data follows a **Bernoulli distribution**, violating that assumption. While the Central Limit Theorem makes the t-test *asymptotically* valid for very large samples, the chi-square test is the *principled* choice for categorical data. |
| **Chi-square test of independence** | Tests whether two categorical variables are *independent* (i.e., whether group membership is associated with conversion) | ✅ Both variables are categorical: `test group` (ad/psa) and `converted` (True/False). The chi-square test directly answers: "Is there a statistically significant association between seeing ads and converting?" |

**Bottom line**: When both the independent variable (group) and the dependent variable (outcome) are categorical, the chi-square test of independence is the standard, correct test.

### Statistical Framework

1. **Null Hypothesis (H₀)**: There is no association between test group and conversion — the ad campaign had no effect.
2. **Alternative Hypothesis (H₁)**: There is an association — the ad campaign affected conversions.
3. **Significance Level**: α = 0.05 (5% chance of a false positive).
4. **Confidence Interval**: We compute a 95% CI for the difference in conversion rates to quantify the *magnitude* of the effect (not just whether it exists).

---

## Results

> Run `python scripts/ab_test.py` to reproduce the statistical results, or open `notebooks/01_analysis.ipynb` for the full exploratory analysis.

The results include:

- **Chi-square statistic and p-value**: Whether the difference is statistically significant.
- **Conversion rates**: Exact percentages for each group.
- **95% Confidence Interval**: The range of plausible values for the true difference in conversion rates.
- **Business interpretation**: A plain-English summary of what the numbers mean.

---

## Recommendation

The recommendation depends on the statistical results:

- **If p < 0.05**: The ad campaign significantly changed conversion behavior. Examine the confidence interval to quantify the expected lift and decide whether the ROI justifies continued ad spend.
- **If p ≥ 0.05**: There is no statistically significant evidence that the ad campaign changed conversions. Consider reallocating the ad budget or redesigning the campaign creative.

Additionally, the hour-of-day analysis in `notebooks/02_visualization.ipynb` can inform **when** to show ads for maximum impact, regardless of the overall campaign result.

---

## Project Structure

```
ab-testing-portfolio/
├── data/
│   └── raw/
│       └── marketing_AB.csv        # Raw dataset
├── notebooks/
│   ├── 01_analysis.ipynb           # EDA: distributions, group sizes, conversion rates
│   └── 02_visualization.ipynb      # Publication-quality charts with CIs
├── scripts/
│   └── ab_test.py                  # Chi-square test, CI calculation, business summary
└── README.md                       # This file
```

---

## How to Run

```bash
# Install dependencies
pip install pandas numpy scipy matplotlib seaborn

# Run the statistical test
python scripts/ab_test.py

# Open the notebooks
jupyter notebook notebooks/01_analysis.ipynb
```

---

## Interview Talking Points

1. **Why chi-square over t-test?** — Conversion is binary/categorical, not continuous. Chi-square tests association between two categorical variables.
2. **Why a confidence interval in addition to the p-value?** — The p-value tells you *if* there's a difference; the CI tells you *how big* the difference likely is. Business decisions need both.
3. **What assumptions does chi-square require?** — Observations are independent (each user counted once), expected cell counts ≥ 5 (easily met with large samples), and the data is from random sampling.
4. **What would you do differently with more data?** — Segment by hour/day for targeted recommendations, run a logistic regression to control for `total ads` as a covariate, or use a Bayesian approach for richer probability statements.
