import pandas as pd

fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

fake_df["label"] = "Fake"
true_df["label"] = "True"

df = pd.concat([fake_df, true_df]).sample(frac=1).reset_index(drop=True)

df = df[["text", "label"]].dropna()

print(df.head())

df.to_csv("news.csv", index=False)