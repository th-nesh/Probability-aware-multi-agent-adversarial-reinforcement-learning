
import pandas as pd

import matplotlib.pyplot as  plt
from matplotlib import pyplot
import pandas as pd
import numpy as np

finaloutput = "dist_csvs/error_dist_move_all.csv"
dataset_final = pd.read_csv(finaloutput, delimiter= ",")
dataset_final.columns = ['data']
#print(dataset_final)
#dataset_final = dataset_final.astype({"yolo_center_x":"float","yolo_width":"float","ref_center_x":"float","ref_width":"float"})

#plt.hist(dataset_final["center_error"], bins = 5)
#plt.show()
#print(dataset_final)
X= dataset_final["data"]
print(np.mean(X))
bins = np.linspace(-5,5,6)
counts, bins = np.histogram(X, bins = bins, range = [-5, 10], density = False)
binscenters = np.array([0.5 * (bins[i] + bins[i+1]) for i in range(len(bins)-1)])
counts = counts/len(X)
print(counts)
xspace = np.linspace(-5, 10, 100000)
plt.bar(binscenters, counts, color='navy', label=r'Histogram entries')
#plt.plot(xspace, exponential(xspace, *popt_exponential), color='darkorange', linewidth=2.5, label=r'Fitted function')
plt.xlabel("Deviation in pixels")
plt.ylabel("No.Of Observations")
plt.show()

a = [2,3,4,5]

