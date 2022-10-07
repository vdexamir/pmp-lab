import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

# Distributia pentru primul mecaninc
m1 = stats.expon(scale=1 / 4).rvs(size=10000)
# Distributia pentru al doilea mecanic
m2 = stats.expon(scale=1 / 6).rvs(size=10000)

m = m1

az.plot_posterior({'m1': m1, 'm2': m2, 'm': m})
plt.show()

[30, 30, 30, 30]
[20, 20, 20, 20]

