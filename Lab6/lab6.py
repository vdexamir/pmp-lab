import matplotlib.pyplot as plt
import pandas as pd
import pymc3 as pm
from sklearn import preprocessing


def read_data():
    # Read csv data from "data.csv" in a data frame
    # ppvt = rezultatele testelor cognitive ale copiilor de trei ani
    # educ_cat = educaţia mamei la momentul naşterii
    # momage = vârsta mamei

    df = pd.read_csv("data.csv")

    return df


def reshape(data_frame):
    return data_frame.values.reshape(-1, 1)


def print_dependencies_mother_age_graphic(mother_age, ppvt):
    # 1. Reprezentaţi grafic datele care dau dependenţa rezultatului testului de vârsta mamei
    ma = reshape(mother_age)
    iq = reshape(ppvt)

    # Plot
    plt.scatter(ma, iq, color='r')
    plt.title('Mom age dependencies result')
    plt.xlabel('Mother Age')
    plt.ylabel('IQ')
    plt.xticks(())
    plt.yticks(())
    plt.show()


def define_model(mother_age, ppvt):
    # 2. Definiţi modelul Bayesian de regresie liniară (folosind PyMC3) care sa descrie contextul de mai sus.

    with pm.Model():
        α = pm.Normal('α', mu=0, sd=50)
        β = pm.Normal('β', mu=0, sd=10)
        ε = pm.HalfCauchy('ε', 5)
        μ = pm.Deterministic('μ', α + β * mother_age)
        prediction = pm.Normal('prediction', mu=μ, sd=ε, observed=ppvt)
        trace = pm.sample(2000, tune=2000, return_inferencedata=True)

    return trace


def main():
    df = read_data()

    mother_age = df["momage"]
    ppvt = df["ppvt"]

    print_dependencies_mother_age_graphic(mother_age, ppvt)

    trace = define_model(reshape(mother_age), reshape(ppvt))

    regression = trace.posterior.stack(samples={"momage", "ppvt"})
    print(regression)


if __name__ == '__main__':
    main()
