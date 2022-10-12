import random
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from scipy import stats
from functools import reduce

np.random.seed(1)

# Constante
size = 10000
α1 = 4
λ1 = 1 / 3
α2 = 4
λ2 = 1 / 2
α3 = 5
λ3 = 1 / 2
α4 = 5
λ4 = 1 / 3
λ = 1 / 4
p1 = 0.25
p2 = 0.25
p3 = 0.30
p4 = 1 - p1 - p2 - p3

# Calculam distributiile pentru fiecare dintre cele 4 servere, inclusiv latenta dintre client si server
server1 = stats.gamma(α1, scale=λ1).rvs(size=size)
server2 = stats.gamma(α2, scale=λ2).rvs(size=size)
server3 = stats.gamma(α3, scale=λ3).rvs(size=size)
server4 = stats.gamma(α4, scale=λ4).rvs(size=size)
latency = stats.expon(scale=λ).rvs(size=size)

server_redirect = random.choices((1, 2, 3, 4), weights=(p1, p2, p3, p4), k=size)
responses_time = []


def compute_response_time(index):
    match server_redirect[index]:
        case 1:
            return server1[index] + server_redirect[index]
        case 2:
            return server2[index] + server_redirect[index]
        case 3:
            return server3[index] + server_redirect[index]
        case _:
            return server4[index] + server_redirect[index]


# Calculam timpul de raspuns pentru 10000 de requesturi in medie in functie de alegeri random ale serverelor
for i in range(size):
    responses_time.append(compute_response_time(index=i))

more_than_3_milliseconds = reduce(lambda total, value: total + value, responses_time)

az.plot_posterior({'responses time': responses_time})
plt.show()

print("p: X > 3ms = ", more_than_3_milliseconds / size)
print("media = ", np.mean(responses_time))
print("deviatia standard = ", np.std(responses_time))
