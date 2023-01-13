import arviz as az
import matplotlib.pyplot as plt
import pymc3 as pm

if __name__ == '__main__':
    # Ex1 - Pentru fiecare model, identificaţi numărul de lanţuri, mărimea totală a eşantionului generat şi vizual-
    # izaţi distribuţia a posteriori.

    data_centered = az.load_arviz_data("centered_eight")
    data_non_centered = az.load_arviz_data("non_centered_eight")

    print(data_centered)
    print("-------------------------------------------")
    print(data_non_centered)

    # Numarul de lanturi
    chains_centered = len(data_centered["posterior"])
    chains_non_centered = len(data_non_centered["posterior"])

    print(f"Number of chains - centered: {chains_centered}")
    print(f"Number of chains - non-centered: {chains_non_centered}")
    print("-------------------------------------------")

    # Marimea totala a esantionului
    sample_length_centered = az.ess(data_centered).sizes.get("school")
    sample_length_non_centered = az.ess(data_non_centered).sizes.get("school")

    print(f"Total sample length - centered: {sample_length_centered}")
    print(f"Total sample length - non-centered: {sample_length_non_centered}")
    print("-------------------------------------------")

    # Distributia a posteriori
    az.plot_posterior(data_centered)
    az.plot_posterior(data_non_centered)
    plt.show()

    # Ex2 - Folosiţi ArviZ pentru a compara cele două modele, după criteriile Rˆ (Rhat) şi autocorelaţie. Concentraţi-
    # vă pe parametrii mu şi tau

    params = ["mu", "tau"]

    # Rhat
    rhat_centered = az.rhat(data_centered, var_names=params)
    rhat_non_centered = az.rhat(data_non_centered, var_names=params)

    print(rhat_centered)
    print(rhat_non_centered)
    print("-------------------------------------------")

    # Autocorelatie
    az.plot_autocorr(data_centered, var_names=params)
    az.plot_autocorr(data_non_centered, var_names=params)
    plt.show()

    # Ex3 - Număraţi numărul de divergenţe din fiecare model (cu sample_stats.diverging.sum() ), iar apoi
    # identificaţi unde acestea tind să se concentreze în spaţiul parametrilor (mu şi tau ). Puteţi folosi mod-
    # elul din curs, cu az.plot pair sau az.plot parallel

    az.plot_parallel(data_centered, var_names=params)
    az.plot_parallel(data_non_centered, var_names=params)
    plt.show()
