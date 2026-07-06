import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/munirahsofeaMRT244075/agile-data-science-pma/refs/heads/main/data./telco.csv"

def load_data():
    return pd.read_csv(DATA_URL)

def test_no_duplicate_rows():
    df = load_data()
    assert df.duplicated().sum() == 0, "Dataset contains duplicate rows"

def test_total_charges_no_blanks():
    df = load_data()
    blanks = (df['TotalCharges'].astype(str).str.strip() == '').sum()
    assert blanks == 0, f"TotalCharges has {blanks} blank values"

def test_churn_values_valid():
    df = load_data()
    valid_values = {'Yes', 'No'}
    assert set(df['Churn'].unique()).issubset(valid_values), "Churn column has unexpected values"
