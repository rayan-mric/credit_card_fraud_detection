import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/processed/cleaned.csv")

# class distribution
plt.figure()
sns.countplot(x="Class", data=df)
plt.title("Fraud vs Non-Fraud Distribution")
plt.savefig("../outputs/charts/class_distribution.png")

print("Charts saved successfully")