import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

if __name__ == "__main__":
    az.style.use('arviz-darkgrid')

    # 1. Pe modelul polinomial din curs, în codul care generează datele (din fişierul date.csv ), schimbaţi
    # order=2 cu o altă valoare, de exemplu order=5 .

    dummy_data = np.loadtxt('./date.csv')
    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]
    order = 5
    x_1p = np.vstack([x_1 ** i for i in range(1, order + 1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()
    plt.scatter(x_1s[0], y_1s)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    plt.savefig('./dummy_data.png')

    # a) Faceţi apoi inferenţa cu model_p şi reprezentaţi grafic această curbă.

    with pm.Model() as model_p:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=10, shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_p = pm.sample(2000, return_inferencedata=True, target_accept=0.95)

    α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
    β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_post = α_p_post + np.dot(β_p_post, x_1s)

    plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {order}')
    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()
    plt.savefig('./model_p.png')

    # b) Repetaţi, dar folosind o distribuţie pentru beta cu sd=100 în loc de sd=10 . În ce fel sunt curbele
    # diferite? Încercaţi acest lucru şi cu sd=np.array([10, 0.1, 0.1, 0.1, 0.1])

    with pm.Model() as model_p_sd_100:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=100, shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_p_sd_100 = pm.sample(2000, return_inferencedata=True, target_accept=0.95)

    α_p_sd_100_post = idata_p_sd_100.posterior['α'].mean(("chain", "draw")).values
    β_p_sd_100_post = idata_p_sd_100.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_sd_100_post = α_p_sd_100_post + np.dot(β_p_sd_100_post, x_1s)

    plt.plot(x_1s[0][idx], y_p_sd_100_post[idx], 'C2', label=f'model order {order}')
    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()
    plt.savefig('./model_p_sd_100.png')

    with pm.Model() as model_p_np_array:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_p_np_array = pm.sample(2000, return_inferencedata=True, target_accept=0.95)

    α_p_np_array_post = idata_p_np_array.posterior['α'].mean(("chain", "draw")).values
    β_p_np_array_post = idata_p_np_array.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_np_array_post = α_p_np_array_post + np.dot(β_p_np_array_post, x_1s)

    plt.plot(x_1s[0][idx], y_p_np_array_post[idx], 'C2', label=f'model order {order}')
    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()
    plt.savefig('./model_p_np_array.png')

    # 2. Repetaţi exerciţiul precedent, dar creşteţi numărul de date la 500 de puncte.
    dummy_data = np.loadtxt('./date_500.csv')
    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]
    order = 5
    x_1p = np.vstack([x_1 ** i for i in range(1, order + 1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()
    plt.scatter(x_1s[0], y_1s)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    plt.savefig('./dummy_data_500.png')

    # a) Faceţi apoi inferenţa cu model_p şi reprezentaţi grafic această curbă.
    with pm.Model() as model_p_500:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=10, shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_p_500 = pm.sample(2000, return_inferencedata=True, target_accept=0.95)

    α_p_500_post = idata_p_500.posterior['α'].mean(("chain", "draw")).values
    β_p_500_post = idata_p_500.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_500_post = α_p_500_post + np.dot(β_p_500_post, x_1s)

    plt.plot(x_1s[0][idx], y_p_500_post[idx], 'C2', label=f'model order {order}')
    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()
    plt.savefig('./model_p_500.png')

    # b) Repetaţi, dar folosind o distribuţie pentru beta cu sd=100 în loc de sd=10 . În ce fel sunt curbele
    # diferite? Încercaţi acest lucru şi cu sd=np.array([10, 0.1, 0.1, 0.1, 0.1])

    with pm.Model() as model_p_500_sd_100:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=100, shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_p_500_sd_100 = pm.sample(2000, return_inferencedata=True, target_accept=0.95)

    α_p_500_sd_100_post = idata_p_500_sd_100.posterior['α'].mean(("chain", "draw")).values
    β_p_500_sd_100_post = idata_p_500_sd_100.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_500_sd_100_post = α_p_500_sd_100_post + np.dot(β_p_500_sd_100_post, x_1s)

    plt.plot(x_1s[0][idx], y_p_500_sd_100_post[idx], 'C2', label=f'model order {order}')
    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()
    plt.savefig('./model_p_500_sd_100.png')

    with pm.Model() as model_p_500_np_array:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=order)
        ε = pm.HalfNormal('ε', 5)
        μ = α + pm.math.dot(β, x_1s)
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_p_500_np_array = pm.sample(2000, return_inferencedata=True, target_accept=0.95)

    α_p_500_np_array_post = idata_p_500_np_array.posterior['α'].mean(("chain", "draw")).values
    β_p_500_np_array_post = idata_p_500_np_array.posterior['β'].mean(("chain", "draw")).values
    idx = np.argsort(x_1s[0])
    y_p_500_np_array_post = α_p_500_np_array_post + np.dot(β_p_500_np_array_post, x_1s)

    plt.plot(x_1s[0][idx], y_p_500_np_array_post[idx], 'C2', label=f'model order {order}')
    plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
    plt.legend()
    plt.savefig('./model_p_500_np_array.png')

# 3.Faceţi inferenţa cu un model cubic (order=3 ), calculaţi WAIC şi LOO, reprezentaţi grafic rezultatele
# şi comparaţi-le cu modelele liniare şi pătratice.

    dummy_data = np.loadtxt("date_500.csv")
    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]
    order = 3
    x_1p = np.vstack([x_1 ** i for i in range(1, order + 1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True))
    x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()
    plt.scatter(x_1s[0], y_1s)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    with pm.Model() as model_c:
        α = pm.Normal('α', mu=0, sd=1)
        β = pm.Normal('β', mu=0, sd=10)
        ε = pm.HalfNormal('ε', 5)
        μ = α + β * x_1s[0]
        y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
        idata_c = pm.sample(2000, return_inferencedata=True, target_accept=0.95)

    waic = az.waic(idata_c, scale="deviance")
    loo = az.loo(idata_c, scale="deviance")
    cmp_df = az.compare({'model_c': idata_c, 'model_p': idata_p, 'model_p_500': idata_p_500}, method='BB-pseudo-BMA', ic="waic", scale="deviance")
    az.plot_compare(cmp_df, insample_dev=False, plot_ic_diff=True, figsize=(35, 1))
