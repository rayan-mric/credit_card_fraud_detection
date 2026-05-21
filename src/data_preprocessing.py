import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    # remove duplicates
    df = df.drop_duplicates()

    # scale features
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])
    df["Time"] = scaler.fit_transform(df[["Time"]])

    return df

if __name__ == "__main__":
    df = load_data("data/raw/creditcard.csv")
    df = preprocess(df)

    df.to_csv("data/processed/cleaned.csv", index=False)
    print("Preprocessing complete")