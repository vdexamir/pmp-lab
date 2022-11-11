import arviz as az
import matplotlib.pyplot as plt

import numpy as np
import pymc3 as pm
import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv('Prices.csv')

    price = data['Price'].values
    speed = data['Speed'].values
    hard_drive = data['HardDrive'].values
    ram = data['Ram'].values
    premium = data['Premium'].values

    fig, axes = plt.subplots(2, 2, sharex=False, figsize=(10, 8))
    axes[0, 0].scatter(speed, price, alpha=0.6)
    axes[0, 1].scatter(hard_drive, price, alpha=0.6)
    axes[1, 0].scatter(ram, price, alpha=0.6)
    axes[1, 1].scatter(premium, price, alpha=0.6)
    axes[0, 0].set_ylabel("Price")
    axes[0, 0].set_xlabel("Speed")
    axes[0, 1].set_xlabel("HardDrive")
    axes[1, 0].set_xlabel("Ram")
    axes[1, 1].set_xlabel("Premium")
    plt.savefig('price_correlations.png')

    # Ex.1 - Folosind distribuţii a priori slab informative asupra parametrilor α, β1, β2 şi σ, folosiţi PyMC3 pentru
    # asimula un eşantion suficient de mare (construi modelul) din distribuţia a posteriori.

    prices_model = pm.Model()
    with prices_model:
        alpha = pm.Normal('alpha', mu=0, sd=10)
        beta_1 = pm.Normal('beta_1', mu=0, sd=10)
        beta_2 = pm.Normal("beta_2", mu=0, sd=5)
        sigma = pm.HalfNormal('sigma', sd=10)
        mu = pm.Deterministic("mu", alpha + beta_1 * speed + beta_2 * hard_drive)
        price_like = pm.Normal('price_like', mu=mu, sigma=sigma, observed=price)
        trace = pm.sample(20000, tune=20000, cores=4)
        prm = pm.sample_posterior_predictive(trace, samples=100, model=prices_model)

    az.plot_trace(prm)
    plt.savefig("prm.png")
    plt.show()

    # 2. Obţineţi estimări de 95% pentru HDI ale parametrilor β1 şi β2.

