//// gaussian.stan
// stan model file to estimate the mean and standard deviation of a gaussian distribution

// data provided to Stan
// must match the name of the columns in data/gaussian.csv!
data {
  int N1;          // number of observations
  real value[N1];  // array of observations
}

// model parameters
// these will be sampled and optimized
parameters {
  real mu;
  real<lower=0> sigma;
}

// likelihood and priors considered
model {
  // likelihood
  value ~ normal(mu, sigma);

  // priors
  mu ~ normal(2, 5);
  sigma ~ normal(3, 1);
}
