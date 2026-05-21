import pandas as pd
from sklearn.ensemble import IsolationForest

df = pd.read_csv("data/processed/cleaned.csv")

X = df.drop("Class", axis=1)

model = IsolationForest(contamination=0.01, random_state=42)
df["anomaly"] = model.fit_predict(X)

# convert anomaly labels
df["anomaly"] = df["anomaly"].map({1: 0, -1: 1})

df.to_csv("outputs/predictions_anomaly.csv", index=False)

print("Anomaly detection completed")