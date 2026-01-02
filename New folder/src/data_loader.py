import pandas as pd

def load_data():
    df = pd.read_csv(
        "data/electricity_consumption_dataset.csv",
        parse_dates=["datetime"]
    )
    df.set_index("datetime", inplace=True)
    return df
