import numpy as np
from numpy.random import uniform as U

# This script simply picks random values of the parameters and checks if they
# correspond to inflationary or not inflationary strategies.

RUNS = 10 ** 8
EPS = np.finfo(float).eps

# Feasibility intervals
# Using EPS instead of zero because we draw from (0, 1),
# while Numpy's uniform draws from [0, 1)
PHI = (EPS, 1)
KALPHA = (EPS, 1/2)
KBETA = (EPS, 1/2)
VALPHA = (EPS, 1)
VBETA = (EPS, 1)

def mu(phi, kalpha, kbeta, valpha, vbeta):
    return phi * np.log(1 + 2/3 * np.sqrt(2 * kalpha * valpha)) + \
        (1 - phi) * np.log(1 - 2/3 * np.sqrt(2 * kbeta * vbeta))

def kappa(phi, kalpha, kbeta, valpha, vbeta):
    return phi * 2/3 * np.sqrt(2 * kalpha * valpha) - \
        (1 - phi) * 2/3 * np.sqrt(2 * kbeta * vbeta)

def show(infl, total = RUNS):
    perc = infl / total * 100
    print(f"infl {perc:.0f}%\tnon infl {100 - perc:.0f}%")

def random_values():
    print(f"runs: {RUNS}")
    print(f"error: {EPS}")
    print("---")

    infl = 0

    for t in range(RUNS):
        phi = U(*PHI)
        kalpha = U(*KALPHA)
        kbeta = U(*KBETA)
        valpha = U(*VALPHA)
        vbeta = U(*VBETA)

        m = mu(phi, kalpha, kbeta, valpha, vbeta)
        k = kappa(phi, kalpha, kbeta, valpha, vbeta)

        # if m < 0 and k > 0:
            # print(f"[mu={m}, kappa={k}] ({phi}, {kalpha}, {kbeta}, {valpha}, {vbeta})")

        infl += m > 0

        if (t + 1) % 100_000 == 0:
            show(infl, t)

    show(infl)

def approach():
    phi = 15/16
    kalpha = 1/2
    kbeta = 0
    valpha = 0
    vbeta = 1

    for e in np.logspace(.1, 0, endpoint=False):
        e = e - 1
        m = mu(phi, kalpha - e, kbeta + e, valpha + e, vbeta - e)
        print("error", e, "μ", m) 

if __name__ == "__main__":
    # random_values()
    approach()
