## About
A CLI that simplifies the usage of Markov Chain Monte Carlo methods.

Currently, it implements [Stan](https://mc-stan.org/), where the model must be provided in the Stan programming language, or [emcee](https://emcee.readthedocs.io/en/stable/), where the model must be written in Python.

The output of the chains can then be easily analyzed using this same package.

Although developed with cosmology in mind, this package is completely agnostic to the data and model it uses.


## Table of contents
- [Installation](#installation)
  - [Dependencies](#dependencies)
  - [Stable version](#stable-version)
  - [Development version](#development-version)
- [Quick start](#quick-start)
  - [smc-stan](#smc-stan)
  - [smc-emcee](#smc-emcee)
  - [smc-analyze](#smc-analyze)
- [Credits](#credits)
- [Contributing](#contributing)
- [Release cycle](#release-cycle)
- [License](#license)


## Installation
To avoid conflicts the usage of virtual environments is recommended.

### Dependencies
This package requires the usage of Python version 3, as well as the following packages:

- [Numpy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [emcee](https://emcee.readthedocs.io/)
- [getdist](https://getdist.readthedocs.io/)
- [tqdm](https://tqdm.github.io/)
- [h5py](https://www.h5py.org/)
- [arviz](https://arviz-devs.github.io/arviz/)
- [Pandas](https://pandas.pydata.org/)
- [pystan](https://pystan.readthedocs.io/)
- [pyyaml](https://pyyaml.org/)


Dependencies are automatically resolved by `pip`.

### Stable version
A stable version is not yet available, look below for instructions on how to install this package.

### Development version
If you wish to use this package updated to the latests commits, you can install this package directly from the development branch on this repository using `pip`:
```console
$ pip install -e git+https://github.com/jpmvferreira/simplifiedmc.git@dev#egg=simplifiedmc
```

If instead you wish to make changes to the source code, start by cloning the development branch locally followed by an installation using `pip` with the `-e`, `--editable` flag:
```console
$ git clone -b dev https://github.com/jpmvferreira/simplifiedmc
$ pip install -e simplifiedmc
```


## Quick start
This package provides the following programs:
- `smc-stan` wraps the [Stan](https://mc-stan.org/) programming language, that uses the No-U-Turn-Sample (NUTS) a variant of the Hamiltonian Monte Carlo (HMC).
- `smc-emcee` wraps the [emcee](https://emcee.readthedocs.io/en/stable/) python package, that makes use of the Goodman & Weare’s Affine Invariant (MCMC) Ensemble sampler.
- `smc-analyze` analyzes the output from the two previous programs.

These tools have quite a bit to offer and we will show you how to use each flag, for each program, to provide context on how to use them. Don't forget that there's always the help dialog (`-h` or `--help`) to list all available flags, organized by categories, with a brief description of what they do.

To make getting started easy there's a folder named `example` on this repository, which includes the files used during this short tutorial.

Start by cloning this repository locally (you can skip this step if you install the development version):
```console
$ git clone https://github.com/jpmvferreira/simplifiedmc
```

And now change directory to this folder, where all the action will take place:
```console
$ cd simplifiedmc/example
```

An additional note, the format for any data file provided must be the following:
```csv
# this is a comment
# comments are obviously ignored, empty line as well
# there must be no space between comas

column1,column2
1,2
2,3
3,4
```

### smc-stan
In this guide we will be estimating the mean and standard deviation using `smc-stan`, assuming a gaussian distribution, of a series of observations which are present in `data/gaussian.csv`.

To do so we need to create a model for stan. Because stan is a programming language, models must be specifically written in said language. If you have never programmed in stan it's not that hard and there are plenty of tutorials available online and plenty of [documentation](https://mc-stan.org/users/documentation/).

For now this is not required as we already created the model file over at `model/gaussian.stan`, which looks like the following:
```stan
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
```

**Note:** The variable names in the data section in the previous file, as mentioned in the comments, must match the columns in the .csv file. If later you add multiple sources of data (which you can easily) the number of observations must always be ordered (e.g.: N1, N2, ...).

In order to start constraining our model with the available data, we need to provide the required information: the name of our parameters, the initial conditions, the number of steps to sample the posterior distribution and the number of warmup steps.

There is also some optional arguments: the labels for each parameter to show on the plots/tables (Latex supported!), the markers to show on the corner plot and the number of chains to run (default is all available hardware threads).

Now how do you go about and configure that? You can either choose to write it down in a file using YAML syntax or provide those arguments directly to the CLI. If you happen to provide a given configuration argument in the YAML file and in the CLI, the CLI will have priority. This provides great flexibility as to have a base configuration but allowing for fine tuning the parameters without having to rewrite files.

Now let's provide those parameters in a file, which is available over at `config/gaussian-stan.yml`:
```yml
## guassian-stan.yml
# smc-stan configuration file to estimate the mean and standard deviation of a gaussian distribution

# The names of the parameters
# Must match the names defined in the stan model file!
names: [mu, sigma]

# Initial condition for each parameter, for each chain.
# The most relevant initial conditions implemented in this program are:
# - gauss(mu, sigma)
# - uniform(a, b)
# - float(a)
initial:
  mu: gauss(0, 10)
  sigma: uniform(0, 10)

# Number of steps to sample the posterior distribution, after the warmup.
samples: 250

# Number of steps to warmup each chain, which will be discarded.
warmup: 150
```

And now we can execute our program by calling the model (`-m`, `--model`), the data (`-d`, `--data`) and the configuration file (`-y`, `--yml`):
```console
$ smc-stan --model model/gaussian.stan --data data/gaussian.csv --yml config/gaussian-stan.yml
```

However this is exactly the same as using:
```console
$ smc-stan --model model/gaussian.stan --data data/gaussian.csv --names "['mu', 'sigma']" --initial "{'mu': 'gauss(0, 10)', 'sigma': 'uniform(0, 10'}" --samples 250 --warmup 150
```

Which will print a whole lot of stuff to your terminal, organized into sections that start with `[*]`, and show you the time series as well as the corner plot. Let's ignore the output for now.

One thing that you probably saw is that the corner plots and the time-series don't have fancy Latex labels and there is no markers with the expected value. We can fix that by setting the labels (`-l`, `--labels`) and the markers (`--markers`):

```console
$ smc-stan --model model/gaussian.stan --data data/gaussian.csv --names "['mu', 'sigma']" --initial "{'mu': 'gauss(0, 10)', 'sigma': 'uniform(0, 10)'}" --samples 250 --warmup 150 --labels "['\mu', '\sigma']" --markers "{'mu': 2, 'sigma': 3}"
```

Now we take the previous example and, since there's a lot of stuff going on, so let's redirect the output to a given output folder (`-o`, `--output`) and don't show the plots on screen (`-ns`, `--no-show`):
```console
$ smc-stan --model model/gaussian.stan --data data/gaussian.csv --names "['mu', 'sigma']" --initial "{'mu': 'gauss(0, 10)', 'sigma': 'uniform(0, 10)'}" --samples 250 --warmup 150 --labels "['\mu', '\sigma']" --markers "{'mu': 2, 'sigma': 3}" --output output/gaussian-stan --no-show
```

Now you see the reason why we made configuration files in the first place, this can get quite big, quite fast, even for simple models. Luckily, as we've shown before, we can place all of the flags above in a configuration file just like before, which is located over at `config/gaussian-fancy-stan.yml`. To do exactly the same thing as the big terminal command above you can instead write:
```console
$ smc-stan --model model/gaussian.stan --data data/gaussian.csv --yml config/gaussian-fancy-stan.yml --output output/gaussian-stan --no-show
```

On a high level you only need to know the following: The Latex table with the 1 and 2 sigma confidence interval is at `/output/gaussian-stan/CIs.tex`, the corner plot is over at `/output/gaussian-stan/plot/corner.png` and the time series, which can be used to check for convergence, is over at `/output/gaussian-stan/time-series.png`.

The folder `/output/gaussian-stan/log` includes further information regarding the convergence, as well as a backup of the model and configuration used.

You can also save the chain to the output folder (`-sc`, `--save-chain`) and optionally use compression to make it smaller, using gzip (`-g`, `--gzip`) or lzf (`--lzf`), but you shouldn't worry about that because these chains have a very small footprint.

As we've said before if you wish to do so you can overwrite a given configuration argument provided in the configuration file by setting the corresponding flag in the CLI. In this next example we take the `/config/gaussian-fancy-stan.yml` and set the labels μ → α and σ → β:
```console
$ smc-stan --model model/gaussian.stan --data data/gaussian.csv --yml config/gaussian-fancy-stan.yml --labels "['\alpha', '\beta']" --output output/gaussian-stan --no-show
```

Last but not least we can change the number of chains we wish to run (`-c`, `--chains`). This is important because if you have more chains than you have hardware threads, then some chains will run sequentially. If on the other hand you set a number of chains equal to or lower than the total number of available hardware threads then they will run in parallel.

Each chain is independent and will do a number of steps equal to the number of steps to sample the posterior distribution and the warmup. This means that you can get major speedups in your code (as much as the number of hardware threads in your system!), being the only disadvantage that each chain will have to perform the warmup.

A version of the example so far with a custom number of chains is available at `config/gaussian-final-stan.yml`. This is a complete configuration file with all the available arguments, except for the output of course, that must always be provided in the CLI.

### smc-emcee
Here we will be doing the same thing as befores, estimating the mean and standard deviation, assuming a gaussian distribution, of a series of observations which are present in `data/gaussian.csv`, but this time using `smc-emcee`. For more information visit the official [emcee documentation](emcee.readthedocs.io/).

As we did before, the first thing to do is define our model. Because `emcee` is written in Python a model is simply a Python function that returns the probability of a given parameter values knowing the provided data. Here are the contents of the file over at `model/gaussian.py`:
```python
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
    if -10 < mu < 10 and 0 < sigma < 10:
        return 0.0

    return -np.inf

# define the probability using the prior and likelihood
def ln_probability(θ, value):
    prior = ln_prior(θ)
    if not np.isfinite(prior):
        return -np.inf
    return prior + ln_likelihood(θ, value)
```

In `smc-emcee`, besides the model and the data, the following arguments are required: the names of the parameters, the initial conditions, the percentage that we use as a criteria to consider that convergence is met (it uses the autocorrelation time to do so) and the number of steps to sample the posterior distribution.

In short this means that the sampler will run until the autocorrelation time is changing by less than the user provided percentage (5% as given good results so far) and will then sample the posterior distribution by the number of steps the user as provided (usually in the scale of 10⁵).

A simple YAML file with the required configuration arguments, for this specific example, which is located at `config/guassian-emcee.yml`, is the following:
```yaml
# The names of the parameters
# Must match the names defined in the stan model file!
names: [mu, sigma]

# Initial condition for each parameter, for each walker.
# Must be inside the model restrictions, otherwise it might crash.
# The most relevant initial conditions implemented in this program are:
# - gauss(mu, sigma)
# - uniform(a, b)
# - float(a)
initial:
  mu: uniform(-10, 10)
  sigma: uniform(0, 10)

# Autocorreation time must change less than this percentage to consider convergence.
percentage: 50

# Number of steps to sample the posterior distribution, after the convergence is met.
samples: 1000
```

Which we can now use to run our program accordingly:
```console
$ smc-emcee --model model/gaussian.py --data data/gaussian.csv --yml config/gaussian-emcee.yml
```

However there are a few parameter which you might want to fine-tune related to the sampler: the maximum number of steps, the number of steps to check if convergence as been met and the number of walkers.

The maximum number of steps (`-M`, `--maxsteps`) can be used to ensure you don't wait forever. The number of steps to check for convergence (`-c`, `--check`) is quite slow, so don't set this value too low, as it will take more time checking if it has converged rather than sampling the distribution, but also setting this too high means that you will take longer to check if convergence as been met. The number of walkers (`-w`, `--walkers`) is related to the ensemble sampler, setting this higher will cover a higher region of parameter space, but is more computationally expensive.

Because this is an ensemble sampler it doesn't run independent chains, however, you can make use of multi threaded systems by changing the number of Python processes it spawns (`--processes`).

A complete YAML configuration file for this example can be found at `config/guassian-final-emcee.yml` and is not shown here for brevity. Obviously you can also configure the program in the command line without relying on this configuration file.

Finally we're left with the output options. We can specify the output folder (`-o`, `--output`), save the chain to disk (`-sc`, `--save-chain`) and optionally provide compression using gzip (`-g`, `--gzip`) or lzf (`--lzf`).

Because the way `emcee` handles saving the chain it is recommended to use `--tmp` if in your machine `/tmp/` is mounted in RAM, or `--shm` if `/dev/shm` is mounted in RAM to store the chain temporarily in those places.

You can also thin the samples (`-t`, `--thin`) as a function of the autocorrelation time, show the time series plot (`-t`, `--time-series`) which is memory intensive in this case, and optionally don't show the plots on screen (`-ns`, `--no-show`) and don't show the fancy progress bar (`-np`, `--no-progress`).


### smc-analyze
To get some output we've executed the following commands:
```console
$ smc-stan --model model/gaussian.stan --data data/gaussian.csv --yml config/gaussian-final-stan.yml --output output/gaussian-final-stan --save-chain
(supressed output)
$ smc-emcee --model model/gaussian.py --data data/gaussian.csv --yml config/gaussian-final-emcee.yml --output output/gaussian-final-emcee --save-chain --tmp
```

Which stored the sampler information in their respective directories. If you wish to plot them together in the same corner plot, do as follows:
```console
$ smc-analyze -i output/gaussian-final-emcee output/gaussian-final-stan
```
![](./example/analyzed/emcee-stan.png)

**Note:** You can only use this if both output share the parameter space, which is this case, they do.

This doesn't look very pretty, beside the fact that we haven't provided emcee to converge of course, so let's add the markers showing the expected value (`-m`, `--markers`), a legend (`--legend`) for each plot and change the alpha of both the filled region (`--filled-alpha`) and the contours (`--countor-alpha`):
```console
$ smc-analyze -i output/gaussian-final-emcee output/gaussian-final-stan --markers "{'mu': 2, 'sigma': 3}" --legend "['stan', 'emcee']" --contour-alpha 0.25 --filled-alpha 0.9
```
![](./example/analyzed/emcee-stan-fancy.png)


This looks better and now we can choose to save it (`-o`, `--output`) and not to have it display on the screen (`-ns`, `--no-show`).

We could have also specified the labels (`-l`, `--labels`), however this defaults to whatever was used to run the sampler, so make sure to have these consistent between runs, otherwise use this argument to overwrite it.


## Credits
The contents in this repository were developed by myself, you can contact me in the following ways:
- Personal email: [jose@jpferreira.me](mailto:jose@jpferreira.me) - [[PGP key](https://pastebin.com/REkhQKg2)]
- Institutional email: [joseferreira@alunos.fc.ul.pt](mailto:joseferreira@alunos.fc.ul.pt) - [[PGP key](https://pastebin.com/rfBpi8jc)]


## Contributing
Any discussion, suggestions, pull requests or bug reports are always welcome. If you wish to submit you code, pull requests should be targeted towards the dev branch, otherwise, feel free to use this issue section in this repository, or even send me an email.


## Release cycle
All versions will have the format X.Y.Z, and the first one will be 1.0.0, which is release as soon as I think the program is good enough to be shared.

Each time that there is an update which does not modify the program behavior (e.g.: documentation, packaging, fixing bugs) it will increment Z (e.g.: 1.0.0 -> 1.0.1).

Each time that there is an update which modifies the program behavior (e.g.: adding features) it will increment Y and reset Z (e.g.: 1.0.1 -> 1.1.0).

Each time that there is an update which is not backwards compatible (e.g.: removing features, changing how the program is meant to be used) it will increment X and reset both Y and Z (e.g.: 1.1.2 -> 2.0.0).

In this repository you will find branches for the stable version (master) and the development version (dev). All modifications are done in the dev branch and, after being tested, are merged in the master branch. After a version bump the new version will be released.


## License
All of the contents provided in this repository are available under the MIT license.

For further information, refer to the file [LICENSE.md](./LICENSE.md) provided in this repository.
