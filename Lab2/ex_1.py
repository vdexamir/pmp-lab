import random
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from scipy import stats

np.random.seed(1)

# Constante
size = 10000
λ1 = 1/4
λ2 = 1/6

# Distributia pentru primul mecaninc
m1 = stats.expon(scale=λ1).rvs(size=size)
# Distributia pentru al doilea mecanic
m2 = stats.expon(scale=λ2).rvs(size=size)
# Distributia pentru cei doi mecanici
m = []

# Daca avem 10000 de clienti verificam in medie la ce mecanic ajunge, in functie de rezultat aplicam distributia
# corespunzatoare fiecarui mecanic
for i in range(size):
    client = random.randint(0, 100)
    if client < 40:
        distribution = stats.expon(scale=λ1).rvs(1)[0]
    else:
        distribution = stats.expon(scale=λ2).rvs(1)[0]

    m.append(distribution)

az.plot_posterior({'m1': m1, 'm2': m2, 'm': m})
plt.show()

print("media = ", np.mean(m))
print("deviatia standard = ", np.std(m))



