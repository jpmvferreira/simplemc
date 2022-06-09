# imports
from datetime import datetime, timezone
from multiprocessing import cpu_count
from random import gauss, uniform
import matplotlib.pyplot as plt
import numpy as np
import arviz as az
import yaml
import sys
import os

# local import
import simplifiedmc as smc


# fetch the arguments from CLI and from the configuration file and check for incompatible or missing arguments
def load(args):
    # required arguments
    model = args.model
    data = args.data
    output = args.output

    # global config file
    if args.config:
        config = args.config
    else:
        os.system("echo 'none: none' > /tmp/dummy.yml")  # workaround because loading an empty file crashes
        config = "/tmp/dummy.yml"                        # .

    # get config arguments from file or CLI
    with open(config, "r") as file:
        yml_loaded = yaml.full_load(file)

        # sampler configuration
        names = eval(args.names) if args.names else yml_loaded.get("names")
        labels = eval(args.labels) if args.labels else yml_loaded.get("labels")
        initial = eval(args.initial) if args.initial else yml_loaded.get("initial")
        markers = eval(args.markers) if args.markers else yml_loaded.get("markers")
        samples = args.samples if args.samples else yml_loaded.get("samples")
        warmup = args.warmup if args.warmup else yml_loaded.get("warmup")
        chains = args.chains if args.chains else yml_loaded.get("chains")

        # model selection criteria
        PSIS_LOO_CV = args.PSIS_LOO_CV if args.PSIS_LOO_CV else yml_loaded.get("PSIS-LOO-CV")
        WAIC = args.WAIC if args.WAIC else yml_loaded.get("WAIC")
        AIC = args.AIC if args.AIC else yml_loaded.get("AIC")
        BIC = args.BIC if args.BIC else yml_loaded.get("BIC")
        DIC = args.DIC if args.DIC else yml_loaded.get("DIC")

        # output configuration
        overwrite = args.overwrite if args.overwrite else yml_loaded.get("overwrite")
        savechain = args.save_chain if args.save_chain else yml_loaded.get("save-chain")
        compress = args.compress if args.compress else yml_loaded.get("compress")
        hide_plots = args.hide_plots if args.hide_plots else yml_loaded.get("hide-plots")

    # perform checks and set defaults
    if not names:
        raise Exception("Parameters names must be provided either in CLI or configuration file")
    if not labels:
        labels = names
    if not initial:
        raise Exception("Initial confitions must be provided either in CLI or configuration file")
    if not samples:
        raise Exception("The number of steps to sample the posterior distribution, after the warmup, must be provided either in CLI or configuration file")
    if not warmup:
        raise Exception("The number of steps to warmup each chain must be provided either in CLI or configuration file")

    if not chains:
        chains = cpu_count()

    if not markers:
        markers = {}
    for name in names:
        try:
            markers[name]
        except KeyError:
            markers[name] = None

    # check if sizes match
    if not ( len(names) == len(labels) == len(initial) ):
        raise Exception(f"number of dimensions missmatch: len(names) = {len(names)}, len(labels) = {len(labels)}, len(initial) = {len(initial)}")

    # evaluate initial conditions to Python function(s), for each chain
    # we're returning both init and initial because the later is required to output the configuration used
    init = []
    for i in range(0, chains):
        init.append({})
        for name in names:
            init[i][name] = eval(initial[name])

    # number of parameters, useful later
    ndim = len(names)

    return model, data, output, config, names, labels, initial, markers, samples, warmup, chains, PSIS_LOO_CV, WAIC, AIC, BIC, DIC, overwrite, savechain, compress, hide_plots, init, ndim


# save configuration used to file
def save(file, names, labels, initial, markers, samples, warmup, chains, PSIS_LOO_CV, WAIC, AIC, BIC, DIC, overwrite, savechain, compress, hide_plots):
    with open(file, "w") as file:
        file.write("## config.yml\n")
        file.write("# backup of the configuration arguments used for this run\n")
        file.write("\n")

        yaml.dump({"names": names}, file)
        file.write("\n")
        yaml.dump({"labels": labels}, file)
        file.write("\n")
        yaml.dump({"initial": initial}, file, sort_keys=False)
        file.write("\n")
        yaml.dump({"markers": markers}, file, sort_keys=False)
        file.write("\n")
        yaml.dump({"samples": samples}, file)
        file.write("\n")
        yaml.dump({"warmup": warmup}, file)
        file.write("\n")
        yaml.dump({"chains": chains}, file)
        file.write("\n")

        yaml.dump({"PSIS-LOO-CV": PSIS_LOO_CV}, file)
        file.write("\n")
        yaml.dump({"WAIC": WAIC}, file)
        file.write("\n")
        yaml.dump({"AIC": AIC}, file)
        file.write("\n")
        yaml.dump({"BIC": BIC}, file)
        file.write("\n")
        yaml.dump({"DIC": DIC}, file)
        file.write("\n")

        yaml.dump({"overwrite": overwrite}, file)
        file.write("\n")
        yaml.dump({"save-chain": savechain}, file)
        file.write("\n")
        yaml.dump({"compress": compress}, file)
        file.write("\n")
        yaml.dump({"hide-plots": hide_plots}, file)
        file.write("\n")

    return

# get model selection criteria
# (can be easily adapted to emcee and moved to shared.py)
def criteria(fit, PSIS_LOO_CV, WAIC, AIC, BIC, DIC, file=sys.stdout, plotkhat=None, noshow=False):
    if file != sys.stdout:
        file = open(file, "w")
        print("## criteria.log", file=file)
        print("# model selection criteria\n", file=file)

    if PSIS_LOO_CV:
        loo = az.loo(fit, pointwise=True)
        value, error = loo[0:2]

        if plotkhat:
            az.plot_khat(loo, show_bins=True)
            if plotkhat:
                plt.savefig(plotkhat, transparent=True)
            if not noshow:
                plt.show()
            plt.close()

        print(f"PSIS-LOO-CV: {value} ± {error}", file=file)
    else:
        print("PSIS-LOO-CV: Not calculated", file=file)

    if WAIC:
        value, error = az.waic(fit)[0:2]
        print(f"WAIC: {value} ± {error}", file=file)
    else:
        print("WAIC: Not calculated", file=file)

    if AIC:
        print("AIC: Not implemented (yet!)")  # to-do
    else:
        print("AIC: Not calculated", file=file)

    if BIC:
        print("BIC: Not implemented (yet!)")  # to-do
    else:
        print("BIC: Not calculated", file=file)

    if DIC:
        print("DIC: Not implemented (yet!)")  # to-do
    else:
        print("DIC: Not calculated", file=file)

    if file != sys.stdout:
        file.close()

    return


# convert fit to a numpy array of size [steps, chains, ndim], with all of the computed steps
def getsteps(fit, names, samples, warmup, chains, ndim):
    totalsteps = np.empty([samples+warmup, chains, ndim])
    for i in range(ndim):
        for j in range(chains):
            totalsteps[:, j, i] = fit[names[i]][0][j::chains]

    return totalsteps


# flatten total steps (i.e. remove chain information) and remove warmup to a numpy array of size [steps, ndim]
def getflatsamples(samples, warmup, chains, ndim, totalsteps):
    flatsamples = np.empty([samples*chains, ndim])
    for i in range(ndim):
        start = 0
        for j in range(chains):
            flatsamples[start::chains, i] = totalsteps[warmup:, j, i]
            start += 1

    return flatsamples


# plot time series
def timeseries(totalsteps, names, labels, markers, samples, warmup, chains, ndim, output=None, noshow=False):
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    steps = np.arange(samples+warmup)

    for i in range(ndim):
        ax = axes[i]
        for j in range(chains):
            ax.plot(steps, totalsteps[:, j, i], alpha=0.75)
        ax.set_xlim(0, samples+warmup)
        ax.set_ylabel("$" + labels[i] + "$")
        ax.axvline(x=warmup, linestyle="--", color="black", alpha=0.5)
        if markers[names[i]]:
            ax.axhline(y=markers[names[i]], linestyle="--", color="black", alpha=0.5)
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.grid()

    axes[-1].set_xlabel("step number")
    if output:
        plt.savefig(output, transparent=True)
    if not noshow:
        plt.show()
    plt.close()

    return


# print run information
def runlog(timeelapsed, file=sys.stdout):
    if file != sys.stdout:
        file = open(file, "w")
        print("## run.log", file=file)
        print("# information regarding this run\n", file=file)

    print("# program version", file=file)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    version = f"dev on {date} (UTC)" if smc.__version__ == "0.0.0" else smc.__version__
    print(f"{version}", file=file)
    print("", file=file)

    print("# execution time in the format hours:minutes:seconds", file=file)
    print(f"{timeelapsed}", file=file)
    print("", file=file)

    print("# finish date", file=file)  # put this in UTC!
    date = os.popen("date").read()[:-1]
    print(f"{date}", file=file)

    if file != sys.stdout:
        file.close()

    return
