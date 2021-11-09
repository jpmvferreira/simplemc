# imports
from multiprocessing import cpu_count
from random import gauss, uniform
import matplotlib.pyplot as plt
import numpy as np
import yaml
import sys
import os


# fetch the arguments from CLI and from the configuration file and check for incompatible or missing arguments
def load(args):
    # required argumentss
    model = args.model
    data = args.data

    # config file
    if args.yml:
        yml = args.yml
    else:
        os.system("echo 'none: none' > /tmp/dummy.yml")
        yml = "/tmp/dummy.yml"

    # get config arguments from file or CLI
    with open(yml, "r") as file:
        yml_loaded = yaml.full_load(file)
        names = eval(args.names) if args.names else yml_loaded.get("names")
        labels = eval(args.labels) if args.labels else yml_loaded.get("labels")
        initial = eval(args.initial) if args.initial else yml_loaded.get("initial")
        markers = eval(args.markers) if args.markers else yml_loaded.get("markers")
        samples = args.samples if args.samples else yml_loaded.get("samples")
        warmup = args.warmup if args.warmup else yml_loaded.get("warmup")
        chains = args.chains if args.chains else yml_loaded.get("chains")

    # output arguments
    output = args.output
    savechain = args.save_chain
    gzip = args.gzip
    lzf = args.lzf
    noshow = args.no_show

    # check if everything is provided
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

    # set default values if not provided
    if not chains:
        chains = cpu_count()

    if not markers:
        markers = {}
    for name in names:
        try:
            markers[name]
        except KeyError:
            markers[name] = None

    if gzip == []:
        gzip = 4
    elif gzip:
        gzip = int(gzip[0])

    # check gzip values
    if gzip:
        if gzip < 0 or gzip > 9:
            raise Exception(f"Value of gzip must be between 0 and 9 (inclusive), provided value was {gzip}")

    # check if sizes match
    if not ( len(names) == len(labels) == len(initial) ):
        raise Exception(f"number of dimensions missmatch: len(names) = {len(names)}, len(labels) = {len(labels)}, len(initial) = {len(initial)}")

    # check for single compression algorithm
    if gzip and lzf:
        raise Exception("--gzip and --lzf are mutually exclusive, pick one compression algorithm")

    # if noshow is provided, output must also be provided
    if noshow and not output:
        raise Exception("Toggling -n, --noshow requires to provide an output folder, otherwise output will not be shown nor saved.")

    # evaluate initial conditions to Python function(s), for each chain
    # we're returning both init and initial because the latest is required to output the configuration used
    init = []
    for i in range(0, chains):
        init.append({})
        for name in names:
            init[i][name] = eval(initial[name])

    # number of parameters, useful later
    ndim = len(names)

    return model, data, yml, names, labels, initial, markers, samples, warmup, chains, output, savechain, gzip, lzf, noshow, init, ndim


# save configuration used to file
def save(yml, names, labels, initial, markers, samples, warmup, chains, outputyml, output, savechain, gzip, lzf, noshow):
    # orderly save the configuration options
    with open(yml, "w") as file:
        file.write("## config.yml\n")
        file.write("# backup of all the configuration arguments used for this specific run\n")
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

    # orderly save the output arguments
    with open(outputyml, "w") as file:
        file.write("## output.yml\n")
        file.write("# backup of all the output arguments used for this specific run\n")
        file.write("\n")

        yaml.dump({"output": output}, file)
        file.write("\n")

        yaml.dump({"save-chain": savechain}, file)
        file.write("\n")

        yaml.dump({"gzip": gzip}, file)
        file.write("\n")

        yaml.dump({"lzf": lzf}, file)
        file.write("\n")

        yaml.dump({"noshow": noshow}, file)
        file.write("\n")

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
        plt.savefig(output)
    if not noshow:
        plt.show()
    plt.close()

    return


# print run information
def runlog(timeelapsed, file=sys.stdout):
    if file != sys.stdout:
        file = open(file, "w")
        print("## run.log", file=file)
        print("# information regarding the execution of this program", file=file)
        print("# the execution time has the format hours:minutes:seconds", file=file)
        print("", file=file)

    print(f"time: {timeelapsed}", file=file)

    if file != sys.stdout:
        file.close()

    return
