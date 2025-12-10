import pandas as pd

INPUT = "data/raw/two-sides.csv"
OUTPUT = "data/raw/two-sides-clean.csv"

def clean(x):
    try:
        s = str(x).strip()
    except:
        return None

    if s == "" or s.lower() in ["nan", "none", "null"]:
        return None
    return s

print("Reading CSV...")
df = pd.read_csv(INPUT)

print("Cleaning SMILES columns...")
df["X1"] = df["X1"].apply(clean)
df["X2"] = df["X2"].apply(clean)

before = len(df)
df = df[df["X1"].notnull() & df["X2"].notnull()].copy()
after = len(df)

print(f"Rows before cleaning: {before}")
print(f"Rows after cleaning : {after}")
print(f"Removed rows: {before - after}")

print("Saving clean file:", OUTPUT)
df.to_csv(OUTPUT, index=False)

print("DONE. File cleaned successfully.")
