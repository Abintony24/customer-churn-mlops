import pandas as pd
from scipy.stats import ks_2samp

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"


def check_data_drift():

    print("Starting Data Drift Monitoring...")

    df = pd.read_csv(DATA_PATH)

    # Prepare reference and current datasets
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )
    df = df.dropna(subset=["TotalCharges"])

    reference_data = df.sample(frac=0.7, random_state=42)
    current_data = df.drop(reference_data.index).copy()

    # Simulate drift in MonthlyCharges
    current_data["MonthlyCharges"] *= 1.5

    drift_detected = False
    threshold = 0.05

    numerical_columns = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    for column in numerical_columns:

        stat, p_value = ks_2samp(
            reference_data[column],
            current_data[column]
        )

        if p_value < threshold:
            print(
                f"Drift detected in {column}: "
                f"p-value = {p_value:.4f}"
            )
            drift_detected = True
        else:
            print(
                f"{column} stable: p-value = {p_value:.4f}"
            )

    print(f"\nDrift detected: {drift_detected}")

    return drift_detected


if __name__ == "__main__":
    check_data_drift()