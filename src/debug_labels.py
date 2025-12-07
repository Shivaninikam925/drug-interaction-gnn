import pandas as pd

df = pd.read_csv("data/raw/two-sides.csv")
print(df["Y"].unique())