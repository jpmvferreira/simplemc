from setuptools import setup

with open(f"README.md") as f:
    long_description = f.read()

setup(name="simplemc",
      version="0.dev",
      description="A CLI that simplifies the usage of MCMC methos, using different algorithms, on different models, with different datasets.",
      long_description=long_description,
      long_description_content_type="text/markdown",
      install_requires=[
        "numpy",
        "matplotlib",
        "emcee",
        "getdist",
        "tqdm",
        "h5py",
        "arviz",
        "pandas",
        "pystan",
        "pyyaml",
      ],
      url="https://github.com/jpmvferreira/simplemc",
      author="José Ferreira",
      author_email="jose@jpferreira.me",
      license="MIT",
      scripts=["bin/smc-stan", "bin/smc-emcee", "bin/smc-analyze"],
      zip_safe=False)
