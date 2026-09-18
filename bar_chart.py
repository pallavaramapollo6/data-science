import pandas as pd
import matplotlib.pyplot as plt
data_file = "D:\\Notes\\DS\\data\\sales.xlsx"
df = pd.read_excel(data_file)
df.groupby("Region")["Sales"].sum().plot(kind="bar")
plt.show()
df.groupby("Region")["Sales"].sum().plot(kind="barh")
plt.show()
df.to_csv("output.csv", index=False)
df.to_excel("output.xlsx", index=False)
