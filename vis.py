import pandas as pd
from s2_bands import S2Bands
import matplotlib.pyplot as plt

band_columns = ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11"]#"B12"]
df = pd.read_csv(r"data\processed\S2A_MSIL2A_20220207T002711_N0400_R016_T54HWE_20220207T023040\csvs\ag.csv")
rows = df.iloc[0:10]

for ind, row1 in rows.iterrows():
    row_np = row1[band_columns].to_numpy()
    plt.plot(row_np)
    plt.show()
    print(row_np)
    print(band_columns)
