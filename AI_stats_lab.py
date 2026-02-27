"""
Prob and Stats Lab – Discrete Probability Distributions

Follow the instructions in each function carefully.
DO NOT change function names.
Use random_state=42 where required.
"""

import numpy as np
import math


# =========================================================
# QUESTION 1 – Card Experiment
# =========================================================

def card_experiment():
    # ----- Theoretical probabilities -----
    P_A = 4 / 52
    P_B = 4 / 52
    P_B_given_A = 3 / 51
    P_AB = (4 / 52) * (3 / 51)

    # ----- Simulation -----
    rng = np.random.default_rng(seed=42)

    n_sim = 200_000
    aces = np.array([1]*4 + [0]*48)

    count_A = 0
    count_A_and_B = 0

    for _ in range(n_sim):
        draw = rng.choice(aces, size=2, replace=False)
        if draw[0] == 1:
            count_A += 1
            if draw[1] == 1:
                count_A_and_B += 1

    empirical_P_A = count_A / n_sim
    empirical_P_B_given_A = count_A_and_B / count_A

    absolute_error = abs(empirical_P_B_given_A - P_B_given_A)

    return (
        P_A,
        P_B,
        P_B_given_A,
        P_AB,
        empirical_P_A,
        empirical_P_B_given_A,
        absolute_error
    )


# =========================================================
# QUESTION 2 – Bernoulli
# =========================================================

def bernoulli_lightbulb(p=0.05):
    theoretical_P_X_1 = p
    theoretical_P_X_0 = 1 - p

    rng = np.random.default_rng(seed=42)
    samples = rng.binomial(1, p, 100_000)

    empirical_P_X_1 = np.mean(samples)

    absolute_error = abs(empirical_P_X_1 - theoretical_P_X_1)

    return (
        theoretical_P_X_1,
        theoretical_P_X_0,
        empirical_P_X_1,
        absolute_error
    )


# =========================================================
# QUESTION 3 – Binomial
# =========================================================

def binomial_bulbs(n=10, p=0.05):
    # Theoretical
    theoretical_P_0 = (1 - p) ** n

    theoretical_P_2 = math.comb(n, 2) * (p**2) * ((1 - p) ** (n - 2))

    theoretical_P_ge_1 = 1 - theoretical_P_0

    # Simulation
    rng = np.random.default_rng(seed=42)
    samples = rng.binomial(n, p, 100_000)

    empirical_P_ge_1 = np.mean(samples >= 1)

    absolute_error = abs(empirical_P_ge_1 - theoretical_P_ge_1)

    return (
        theoretical_P_0,
        theoretical_P_2,
        theoretical_P_ge_1,
        empirical_P_ge_1,
        absolute_error
    )


# =========================================================
# QUESTION 4 – Geometric
# =========================================================

def geometric_die():
    p = 1 / 6

    theoretical_P_1 = p
    theoretical_P_3 = (5/6)**2 * p
    theoretical_P_gt_4 = (5/6)**4

    rng = np.random.default_rng(seed=42)
    samples = rng.geometric(p, 200_000)

    empirical_P_gt_4 = np.mean(samples > 4)

    absolute_error = abs(empirical_P_gt_4 - theoretical_P_gt_4)

    return (
        theoretical_P_1,
        theoretical_P_3,
        theoretical_P_gt_4,
        empirical_P_gt_4,
        absolute_error
    )


# =========================================================
# QUESTION 5 – Poisson
# =========================================================

def poisson_customers(lam=12):
    theoretical_P_0 = math.exp(-lam)
    theoretical_P_15 = math.exp(-lam) * lam**15 / math.factorial(15)

    theoretical_P_ge_18 = 1 - sum(
        math.exp(-lam) * lam**k / math.factorial(k)
        for k in range(18)
    )

    rng = np.random.default_rng(seed=42)
    samples = rng.poisson(lam, 100_000)

    empirical_P_ge_18 = np.mean(samples >= 18)

    absolute_error = abs(empirical_P_ge_18 - theoretical_P_ge_18)

    return (
        theoretical_P_0,
        theoretical_P_15,
        theoretical_P_ge_18,
        empirical_P_ge_18,
        absolute_error
    )
