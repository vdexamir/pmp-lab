import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az


def main():
    model = pm.Model()

    with model:
        cutremur = pm.Bernoulli('C', 0.0005)
        incendiu_p = pm.Deterministic('I_p', pm.math.switch(cutremur, 0.03, 0.01))
        incendiu = pm.Bernoulli('I', p=incendiu_p)
        alarma_p = pm.Deterministic('A_p',
                                    pm.math.switch(cutremur,
                                                   pm.math.switch(cutremur, pm.math.switch(incendiu, 0.98, 0.95), 0.02),
                                                   0.0001))
        alarma = pm.Bernoulli('A', p=alarma_p)
        trace = pm.sample(20000)

    dictionary = {
        'cutremur': trace['C'].tolist(),
        'incendiu': trace['I'].tolist(),
        'alarma': trace['A'].tolist()
    }

    df = pd.DataFrame(dictionary)

    p_cutremur = df[((df['cutremur'] == 1) & (df['alarma'] == 1))].shape[0] / df[df['alarma'] == 1].shape[0]
    print('P Cutremur = ', p_cutremur)

    p_incendiu = df[((df['incendiu'] == 1) & (df['alarma'] == 0))].shape[0] / df[df['alarma'] == 0].shape[0]
    print('P Incendiu = ', p_incendiu)


if __name__ == '__main__':
    main()
