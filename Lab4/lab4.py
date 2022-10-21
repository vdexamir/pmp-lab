import matplotlib.pyplot as plt
import pandas as pd
import pymc3 as pm
import arviz as az


def main():
    # ex1

    model = pm.Model()

    # Constante
    # presupunem alpha 1 pentru a putea defini modelul
    alpha = 1
    λ = 20
    std = 0.5
    normal = 1

    with model:
        customers = pm.Poisson('C', mu=λ)
        order = pm.Normal('O', normal, sigma=std)
        preparation = pm.Exponential('P', lam=alpha)
        trace = pm.sample(2000, chains=1)

    dictionary = {
      'customers': trace['C'].tolist(),
      'order': trace['O'].tolist(),
      'preparation': trace['P'].tolist()
    }

    df = pd.DataFrame(dictionary)
    az.plot_posterior(trace)
    plt.show()


if __name__ == '__main__':
    main()
