import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

# Distributie normala cu media 0 si deviatie standard 1, 1000 samples
x = stats.norm.rvs(0, 1, size=10000)
# Distributie uniforma intre -1 si 1, 1000 samples . Primul parametru fiind limita inferioara a intervalului,
# al doilea parametru fiind "marimea" intervalului, aka [-1,-1+2] = [-1,1]
y = stats.uniform.rvs(-1, 2, size=10000)
# Compunerea prin insumare a celor 2 distributii
z = x + y
# Afisarea aproximarii densitatii probabilitatilor, mediei, intervalului etc. variabilelor x,y,z
az.plot_posterior({'x': x, 'y': y, 'z': z})
plt.show()
