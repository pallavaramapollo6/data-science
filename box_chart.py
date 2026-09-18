#Three lines to make our compiler able to draw:
import pandas as pd
import matplotlib.pyplot as plt
csv_file = "D:\\Notes\\DS\\data\\sales.csv"
df = pd.read_csv(csv_file)
df.plot(kind="box")
plt.show()
