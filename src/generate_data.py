"""Generate a reproducible synthetic dataset of B2B marketing leads."""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_LEADS = 5_000


def main() -> None:
    rng = np.random.default_rng(SEED)
    data = pd.DataFrame(
        {
            "Time_on_Website": np.clip(rng.gamma(shape=2.2, scale=110, size=N_LEADS), 5, 1_800),
            "Email_Open_Rate": rng.beta(a=2.2, b=3.0, size=N_LEADS),
            "Webinar_Attendance": rng.binomial(1, 0.28, size=N_LEADS),
            "Company_Size": rng.choice(
                ["Startup", "Mid-Market", "Enterprise"], N_LEADS, p=[0.42, 0.38, 0.20]
            ),
            "Industry": rng.choice(
                ["Tech", "Healthcare", "Finance", "Retail"], N_LEADS,
                p=[0.35, 0.25, 0.22, 0.18],
            ),
            "Lead_Source": rng.choice(
                ["Organic", "Paid Ads", "Referral"], N_LEADS, p=[0.45, 0.35, 0.20]
            ),
        }
    )

    # Engagement and referral signals raise conversion likelihood. Calibrate the
    # intercept so the population mean is exactly 15%, then sample outcomes.
    score = (
        0.0018 * (data["Time_on_Website"].to_numpy() - 240)
        + 1.35 * (data["Email_Open_Rate"].to_numpy() - 0.42)
        + 0.85 * data["Webinar_Attendance"].to_numpy()
        + 0.35 * (data["Company_Size"].to_numpy() == "Enterprise")
        + 0.18 * (data["Company_Size"].to_numpy() == "Mid-Market")
        + 0.2 * (data["Industry"].to_numpy() == "Tech")
        + 0.42 * (data["Lead_Source"].to_numpy() == "Referral")
        - 0.12 * (data["Lead_Source"].to_numpy() == "Paid Ads")
    )
    low, high = -15.0, 5.0
    for _ in range(80):
        intercept = (low + high) / 2
        mean_rate = np.mean(1 / (1 + np.exp(-(score + intercept))))
        if mean_rate < 0.15:
            low = intercept
        else:
            high = intercept
    probabilities = 1 / (1 + np.exp(-(score + (low + high) / 2)))
    data["Converted"] = rng.binomial(1, probabilities).astype(int)

    output_path = Path(__file__).resolve().parents[1] / "leads_data.csv"
    data.to_csv(output_path, index=False)
    print(f"Saved {len(data):,} leads to {output_path} ({data['Converted'].mean():.1%} converted).")


if __name__ == "__main__":
    main()
