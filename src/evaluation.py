import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("outputs/predictions_anomaly.csv")

print("Confusion Matrix:")
print(confusion_matrix(df["Class"], df["anomaly"]))

print("\nClassification Report:")
print(classification_report(df["Class"], df["anomaly"]))