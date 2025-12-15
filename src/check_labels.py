import pandas as pd

df = pd.read_csv("data/raw/two-sides.csv")

df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
df = df.dropna(subset=["Y"])
df["Y_bin"] = (df["Y"] > 0).astype(int)

print("Total rows:", len(df))
print("Unique raw Y values:", df["Y"].unique()[:20])
print("Positive count (Y>0):", df["Y_bin"].sum())
print("Negative count (Y==0):", len(df) - df["Y_bin"].sum())
