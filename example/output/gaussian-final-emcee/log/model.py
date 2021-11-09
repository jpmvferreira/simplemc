## gaussian.py
# emcee model file to estimate the mean and standard deviation of a gaussian distribution


# imports
from math import log, pi
import numpy as np

# define the natural logarithm of the likelihood
def ln_likelihood(θ, value):
    N = len(value)
    mu, sigma = θ

    sum = 0
    for i in range(0, N):
        sum += (-1/2) * ((value[i] - mu) / sigma)**2

    sum += -N*log(sigma) - (1/2)*log(2*pi)

    return sum

# define the natural logarithm of the priors
def ln_prior(θ):
    mu, sigma = θ

    # flat priors
    if -10 < mu < 10 and 1 < sigma < 10:
        return 0.0

    return -np.inf

# define the probability using the prior and likelihood
def ln_probability(θ, value):
    prior = ln_prior(θ)
    if not np.isfinite(prior):
        return -np.inf
    return prior + ln_likelihood(θ, value)
