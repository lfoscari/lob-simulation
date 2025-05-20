import numpy as np
from numpy.random import uniform as U

# This script simply picks random values of the parameters and checks if they
# correspond to inflationary or not inflationary strategies.

POWER = 1

# Feasibility intervals
PHI = (0, 1)
# KALPHA = (0, 1/2)
# KBETA = (0, 1/2)
KALPHA = (0, 1)
KBETA = (0, 1)
VALPHA = (0, 1)
VBETA = (0, 1)

def mu(phi, kalpha, kbeta, valpha, vbeta):
    return phi * np.log(1 + 2/3 * np.sqrt(2) * (kalpha * valpha)**POWER) + \
        (1 - phi) * np.log(1 - 2/3 * np.sqrt(2) * (kbeta * vbeta)**POWER)

def kappa(phi, kalpha, kbeta, valpha, vbeta):
    return phi * 2/3 * np.sqrt(2) * (kalpha * valpha)**POWER - \
        (1 - phi) * 2/3 * np.sqrt(2) * (kbeta * vbeta)**POWER

def random_values():
    t = 1
    
    pos_mu = 0
    pos_k = 0
    edge_case = 0
    # mu_history = []

    while True:
        phi = U(*PHI)
        kalpha = U(*KALPHA)
        kbeta = U(*KBETA)
        valpha = U(*VALPHA)
        vbeta = U(*VBETA)

        m = mu(phi, kalpha, kbeta, valpha, vbeta)
        # mu_history.append(m)
        pos_mu += m > 0

        k = kappa(phi, kalpha, kbeta, valpha, vbeta)
        pos_k += k > 0

        if m < 0 and k > 0:
          # print(f"[mu={m}, kappa={k}] ({phi}, {kalpha}, {kbeta}, {valpha}, {vbeta})")
          edge_case += 1
          
        if t % 100_000 == 0:
            print(f"[{t}]")
            print(f"μ > 0: {np.round(pos_mu / t * 100)}%")
            print(f"κ > 0: {np.round(pos_k / t * 100)}%")
            print(f"μ < 0 ∧ κ > 0: {edge_case / t * 100}%")
            # print(f"μ (avg): {np.average(mu_history)}")

        t += 1

if __name__ == "__main__":
    random_values()
