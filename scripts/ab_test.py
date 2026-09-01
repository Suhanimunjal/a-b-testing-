"""
A/B Test: Chi-Square Test of Independence
==========================================
Tests whether the marketing ad campaign significantly changed conversion
behavior compared to the PSA (public service announcement) control group.

Statistical approach:
- Chi-square test of independence (not a t-test) because both variables
  are categorical: test group (ad/psa) and converted (True/False).
- 95% confidence interval for the difference in proportions using the
  Wald method (normal approximation), which is valid for large samples.

Usage:
    python scripts/ab_test.py
"""

import pandas as pd
import numpy as np
from scipy import stats


def load_data(filepath: str = "data/raw/marketing_AB.csv") -> pd.DataFrame:
    """Load the marketing A/B test dataset."""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]:,} users, {df.shape[1]} columns\n")
    return df


def compute_conversion_rates(df: pd.DataFrame) -> dict:
    """
    Calculate conversion rate for each test group.

    Returns a dict with group names as keys and dicts of
    {n, conversions, rate} as values.
    """
    results = {}
    for group in ["ad", "psa"]:
        group_data = df[df["test group"] == group]
        n = len(group_data)
        conversions = group_data["converted"].sum()
        rate = conversions / n
        results[group] = {"n": n, "conversions": conversions, "rate": rate}

    return results


def run_chi_square_test(df: pd.DataFrame) -> tuple:
    """
    Perform a Chi-square test of independence on a 2x2 contingency table.

    Why chi-square and not a t-test?
    --------------------------------
    The dependent variable ('converted') is binary (True/False), making it
    categorical. A t-test compares means of a *continuous* variable and
    assumes approximate normality — neither condition is met here.

    The chi-square test directly asks: "Are test group and conversion
    status independent?" — which is exactly our research question.

    Returns:
        (chi2_statistic, p_value, contingency_table)
    """
    # Build the contingency table: rows = test group, cols = converted
    # This is the foundation of the chi-square test — it counts how many
    # users in each group converted vs. didn't convert.
    contingency_table = pd.crosstab(df["test group"], df["converted"])

    print("Contingency Table (observed counts):")
    print(contingency_table)
    print()

    # scipy.stats.chi2_contingency performs the chi-square test.
    # It computes expected frequencies under H₀ (independence),
    # then measures how far observed counts deviate from expected.
    # correction=False disables Yates' continuity correction — standard
    # practice for large samples where the correction is unnecessarily
    # conservative.
    chi2, p_value, dof, expected = stats.chi2_contingency(
        contingency_table, correction=False
    )

    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"Degrees of freedom:   {dof}")
    print(f"P-value:              {p_value:.6f}")
    print()

    return chi2, p_value, contingency_table


def compute_confidence_interval(
    rates: dict, confidence: float = 0.95
) -> tuple:
    """
    Calculate the confidence interval for the difference in conversion rates
    (ad group − psa group) using the Wald method (normal approximation).

    Why the Wald method?
    --------------------
    For large samples (n > 30 in each group, which we have), the sampling
    distribution of a proportion is approximately normal by the Central
    Limit Theorem. The Wald interval uses this approximation to construct
    a CI for the difference p_ad - p_psa.

    Parameters:
        rates: dict from compute_conversion_rates
        confidence: confidence level (default 0.95 for a 95% CI)

    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    p_ad = rates["ad"]["rate"]
    p_psa = rates["psa"]["rate"]
    n_ad = rates["ad"]["n"]
    n_psa = rates["psa"]["n"]

    # Point estimate: difference in conversion rates
    diff = p_ad - p_psa

    # Standard error of the difference in two independent proportions.
    # SE = sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)
    # This formula assumes the two groups are independent (they are —
    # users are randomly assigned to one group only).
    se = np.sqrt(
        (p_ad * (1 - p_ad) / n_ad) + (p_psa * (1 - p_psa) / n_psa)
    )

    # Z-score for the desired confidence level (1.96 for 95%)
    z = stats.norm.ppf((1 + confidence) / 2)

    lower = diff - z * se
    upper = diff + z * se

    return diff, lower, upper


def print_business_summary(
    rates: dict, p_value: float, ci: tuple, alpha: float = 0.05
) -> None:
    """
    Print a plain-English summary of the A/B test results.

    This is the "so what?" — translating statistics into a business
    recommendation that a non-technical stakeholder can act on.
    """
    p_ad = rates["ad"]["rate"]
    p_psa = rates["psa"]["rate"]
    n_ad = rates["ad"]["n"]
    n_psa = rates["psa"]["n"]
    diff, lower, upper = ci

    print("=" * 70)
    print("BUSINESS SUMMARY")
    print("=" * 70)
    print()
    print(f"  Ad group:    {p_ad:.2%} conversion rate  ({rates['ad']['conversions']:,} / {n_ad:,} users)")
    print(f"  PSA group:   {p_psa:.2%} conversion rate  ({rates['psa']['conversions']:,} / {n_psa:,} users)")
    print(f"  Difference:  {diff:+.2%} (ad − psa)")
    print()

    if p_value < alpha:
        print(f"  ✅ STATISTICALLY SIGNIFICANT (p = {p_value:.6f}, α = {alpha})")
        print()

        if diff > 0:
            print(
                f"  The ad campaign significantly INCREASED conversions from "
                f"{p_psa:.2%} to {p_ad:.2%}."
            )
        else:
            print(
                f"  The ad campaign significantly DECREASED conversions from "
                f"{p_psa:.2%} to {p_ad:.2%}."
            )

        print(
            f"  With 95% confidence, the true difference in conversion rates "
            f"is between {lower:+.2%} and {upper:+.2%}."
        )
        print()
        print(
            "  RECOMMENDATION: The data supports that the ad campaign had a "
            "real effect on conversions."
        )
        if diff > 0:
            print(
                "  Consider continuing or scaling the campaign, and analyze "
                "hour-of-day patterns to optimize timing."
            )
        else:
            print(
                "  However, the effect is NEGATIVE — the ads may be hurting "
                "conversions. Consider redesigning the creative."
            )
    else:
        print(f"  ❌ NOT STATISTICALLY SIGNIFICANT (p = {p_value:.6f}, α = {alpha})")
        print()
        print(
            "  There is NO statistically significant evidence that the ad "
            "campaign changed conversion behavior."
        )
        print(
            f"  The observed difference of {diff:+.2%} could plausibly be due "
            f"to random chance."
        )
        print(
            f"  95% CI for the true difference: [{lower:+.2%}, {upper:+.2%}]"
        )
        print()
        print(
            "  RECOMMENDATION: The ad campaign does not appear to be effective. "
            "Consider reallocating the ad budget or redesigning the campaign."
        )

    print()
    print("=" * 70)


def main():
    # ------------------------------------------------------------------
    # Step 1: Load the data
    # ------------------------------------------------------------------
    df = load_data()

    # ------------------------------------------------------------------
    # Step 2: Compute conversion rates for each group
    # ------------------------------------------------------------------
    print("CONVERSION RATES BY GROUP")
    print("-" * 40)
    rates = compute_conversion_rates(df)
    for group, info in rates.items():
        print(
            f"  {group.upper():>3} group: "
            f"{info['rate']:.4%}  "
            f"({info['conversions']:,} conversions / {info['n']:,} users)"
        )
    print()

    # ------------------------------------------------------------------
    # Step 3: Run the chi-square test of independence
    # ------------------------------------------------------------------
    print("CHI-SQUARE TEST OF INDEPENDENCE")
    print("-" * 40)
    chi2, p_value, _ = run_chi_square_test(df)

    # ------------------------------------------------------------------
    # Step 4: Compute 95% confidence interval for the difference
    # ------------------------------------------------------------------
    print("95% CONFIDENCE INTERVAL FOR DIFFERENCE (ad − psa)")
    print("-" * 40)
    ci = compute_confidence_interval(rates)
    diff, lower, upper = ci
    print(f"  Point estimate:  {diff:+.4%}")
    print(f"  95% CI:          [{lower:+.4%}, {upper:+.4%}]")
    print()

    # ------------------------------------------------------------------
    # Step 5: Print the business summary
    # ------------------------------------------------------------------
    print_business_summary(rates, p_value, ci)


if __name__ == "__main__":
    main()
