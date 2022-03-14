import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd



# Data path 
path = "../data/projects.csv"
# Pandas dataframe
data = pd.read_csv(path, encoding="ISO-8859-1")
data = data.dropna()
data_categorical = data.select_dtypes(include='object')
data_categorical = data_categorical[:1000]


# indices_to_keep = data.isfinite([np.nan, np.inf, -np.inf]).any(1)
# data =  data[indices_to_keep].astype(np.float64)


data = data[:1000]


from prince import MCA
famd = MCA(n_components =2, n_iter = 3, random_state = 101)
c = famd.fit(data_categorical)
d =c.transform(data_categorical)

print(d)
c.plot_coordinates(data_categorical,figsize=(15, 10) )

