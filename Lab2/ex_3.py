import numpy as np
import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)

# Constants
n = 10
size = 100
p1 = 50
p2 = 30

# Pentru 100 de seturi calculam rezultate posibile obtinute in urma a 10 aruncari in functie de probabilitatile
# monezilor

ss, sb, bs, bb = [], [], [], []
for result in range(size):
    ssT, sbT, bsT, bbT = 0, 0, 0, 0
    for throws in range(n):
        good_coin = np.random.randint(0, 100)
        bad_coin = np.random.randint(0, 100)
        if good_coin < p1 and bad_coin < p2:
            ssT += 1
        elif good_coin < p1 and bad_coin > p2:
            sbT += 1
        elif good_coin > p1 and bad_coin < p2:
            bsT += 1
        elif good_coin > p1 and bad_coin > p2:
            bbT += 1
    ss.append(ssT)
    sb.append(sbT)
    bs.append(bsT)
    bb.append(bbT)

az.plot_posterior({'ss': ss, 'sb': sb, 'bs': bs, 'bb': bb})
plt.show()
