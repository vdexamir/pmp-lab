import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import pymc3 as pm

if __name__ == '__main__':
    # 1 Generaţi 500 de date dintr-o mixtură de trei distribuţii Gaussiene. În
    clusters = 3
    n_cluster = [100, 200, 300]
    n_total = sum(n_cluster)
    means = [5, 0, 2]
    std_devs = [2, 2, 2]
    mix = np.random.normal(
        np.repeat(means, n_cluster),
        np.repeat(std_devs, n_cluster)
    )
    az.plot_kde(np.array(mix))
    plt.show()
    plt.savefig("./ex_1.png")

    # 2 Calibraţi pe acest set de date un model de mixtură de distribuţii Gaussiene cu 2, 3, respectiv 4 componente.
    components = [2, 3, 4]
    models = []
    results = []
    for component in components:
        with pm.Model() as model:
            p = pm.Dirichlet('p', a=np.ones(component))
            means = pm.Normal('means',
                              mu=np.linspace(mix.min(), mix.max(), component),
                              sd=10, shape=component,
                              transform=pm.distributions.transforms.ordered
                              )
            sd = pm.HalfNormal('sd', sd=10)
            y = pm.NormalMixture('y', w=p, mu=means, sd=sd, observed=mix)

            result = pm.sample(1000, return_inferencedata=True, target_accept=0.95)

            results.append(result)
            models.append(model)

    # 3 Comparaţi cele 3 modele folosind metodele WAIC şi LOO.
    value = dict(zip([str(c) for c in components], results))

    waic_compare = az.compare(value, method='BB-pseudo-BMA', ic="waic", scale="deviance")
    loo_compare = az.compare(value, method='BB-pseudo-BMA', ic="loo", scale="deviance")

    print(waic_compare)
    print(loo_compare)
