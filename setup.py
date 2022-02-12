from setuptools import setup

with open(f"README.md") as f:
    long_description = f.read()

setup(name="simplifiedmc",
      version="1.0.1",
      description="A CLI that simplifies the usage of MCMC methods.",
      scripts=["bin/smc-stan", "bin/smc-emcee", "bin/smc-analyze"],
      packages=["simplifiedmc"],
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
      url="https://github.com/jpmvferreira/simplifiedmc",
      author="José Ferreira",
      author_email="jose@jpferreira.me",
      license="MIT",
      zip_safe=False)
