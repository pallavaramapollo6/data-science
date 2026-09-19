import pandas as pd
df = pd.read_csv("D:\\Notes\\DS\\data\\sales.csv")
# Explore
print(df)

# Create Profit column
df["Profit"] = df["Sales"] - df["Cost"]
print(df)

# Total sales
print("Total Sales:", df["Sales"].sum())

# Average sales
print("Average Sales:", df["Sales"].mean())

# Region-wise sales
print(df.groupby("Region")["Sales"].sum())

# Highest sales
print(df.sort_values("Sales", ascending=False))