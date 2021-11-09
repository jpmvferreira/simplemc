# imports
from multiprocessing import cpu_count
import matplotlib.patches as mpatches
from random import gauss, uniform
import matplotlib.pyplot as plt
import numpy as np
import yaml
import sys


# fetch the arguments from CLI and the configuration file
# check for incompatible or missing arguments
# set default arguments value
def load(args):
    # required arguments
    model = args.model
    data = args.data

    # config file
    if args.yml:
        yml = args.yml
    else:
        os.system("echo 'none: none' > /tmp/dummy.yml")
        yml = "/tmp/dummy.yml"

    # get config arguments from file and overwrite if provided in the CLI
    with open(yml, "r") as file:
        yml_loaded = yaml.full_load(file)

        names = eval(args.names) if args.names else yml_loaded.get("names")
        labels = eval(args.labels) if args.labels else yml_loaded.get("labels")
        initial = eval(args.initial) if args.initial else yml_loaded.get("initial")
        markers = eval(args.markers) if args.markers else yml_loaded.get("markers")
        percentage = args.percentage / 100 if args.percentage else yml_loaded.get("percentage") / 100
        samples = args.samples if args.samples else yml_loaded.get("samples")
        check = args.check if args.check else yml_loaded.get("check")
        maxsteps = args.maxsteps if args.maxsteps else yml_loaded.get("maxsteps")
        walkers = args.walkers if args.walkers else yml_loaded.get("walkers")
        processes = args.processes if args.processes else yml_loaded.get("processes")

    # output arguments
    output = args.output
    savechain = args.save_chain
    gzip = args.gzip
    lzf = args.lzf
    tmp = args.tmp
    shm = args.shm
    thin = args.thin
    timeseries = args.time_series
    noshow = args.no_show
    noprogress = args.no_progress

    # check if everything that is required is provided
    if not names:
        raise Exception("Parameters names must either be provided in CLI or in the configuration file")
    if not labels:
        labels = names
    if not initial:
        raise Exception("Initial confitions must either be provided in CLI or in the configuration file")
    if not percentage:
        raise Exception("The percentage to consider that convergence is met must either be provided in the CLI or in the configuration file")
    if not samples:
        raise Exception("The number of samples to compute when converge is met must either be provided in CLI or in the configuration file")

    # set defaults
    if not check:
        check = 1000

    if not maxsteps:
        maxsteps = 100000

    if not walkers:
        walkers = 32

    if not processes:
        processes = cpu_count()

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

    # exit if output is not provided and noshow is
    if not output and noshow:
        raise Exception("Flag -n, --noshow is provided without providing -o, --output. This means that the output will not be shown nor saved")

    # exit if savechain is provided but output is not
    if not output and savechain:
        raise Exception("Flag --save-chain is set, but output wasn't provided (using --output or -o)")

    # exit if both tmp and shm are provided
    if tmp and shm:
        raise Exception("Flags --tmp and --shm are mutually exclusive, pick the one which is mounted as tmpfs in your system")

    # exit if tmp or shm is provided, but chain is not
    if (tmp or shm) and not savechain:
        raise Exception("Flag --tmp requires the usage of --save-chain")

    # exit if the number of steps to compute is lower than the maximum number of steps
    if maxsteps < samples:
        raise Exception("The maximum number of steps (-M, --maxsteps) must always be larger than the number of steps to compute (-s, --samples)")

    # check for single compression algorithm
    if gzip and lzf:
        raise Exception("--gzip and --lzf are mutually exclusive, pick one compression algorithm")

    # evaluate initial conditions to Python functions, and turn into an emcee compatible numpy array
    # we're returning both init and initial because the latest is required to output the configuration used
    init = np.empty([walkers, len(names)])
    for i in range(0, walkers):
        for j in range(0, len(names)):
            init[i][j] = eval(initial[names[j]])

    # auxiliary varialbel required here and there
    ndim = len(names)

    return model, data, yml, names, labels, initial, markers, percentage, samples, check, maxsteps, walkers, processes, output, savechain, gzip, lzf, tmp, shm, thin, timeseries, noshow, noprogress, init, ndim


# save configuration and output arguments to files
def save(yml, names, labels, initial, markers, percentage, samples, check, maxsteps, walkers, processes, outputyml, output, savechain, gzip, lzf, tmp, shm, thin, timeseries, noshow, noprogress):
    # orderly save the configuration arguments
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

        yaml.dump({"percentage": percentage * 100}, file)
        file.write("\n")

        yaml.dump({"samples": samples}, file)
        file.write("\n")

        yaml.dump({"check": check}, file)
        file.write("\n")

        yaml.dump({"maxsteps": maxsteps}, file)
        file.write("\n")

        yaml.dump({"walkers": walkers}, file)
        file.write("\n")

        yaml.dump({"processes": processes}, file)
        file.write("\n")

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

        yaml.dump({"tmp": tmp}, file)
        file.write("\n")

        yaml.dump({"shm": shm}, file)
        file.write("\n")

        yaml.dump({"thin": thin}, file)
        file.write("\n")

        yaml.dump({"timeseries": timeseries}, file)
        file.write("\n")

        yaml.dump({"noshow": noshow}, file)
        file.write("\n")

        yaml.dump({"noprogress": noprogress}, file)
        file.write("\n")

    return

def autocorrelation(correlation, samples, check, index, laststep, delta, output=None, noshow=False):
    if type(correlation) != tuple:
        correlation = (correlation,)

    for autocorr in correlation:
        n = check * np.arange(1, index + 1)
        y = autocorr[:index]
        xmin = 0
        xmax = n.max() + check
        ymin = y.min() - 0.1 * (y.max() - y.min())
        ymax = y.max() + 0.1 * (y.max() - y.min())

        plt.plot(n, y, marker=".")

        region = mpatches.Rectangle((laststep-samples, autocorr[index-1] - delta), samples, 2*delta, color="red", alpha=0.2, label="convergence region")
        plt.gca().add_patch(region)

    plt.grid()
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.legend(handles=[region])
    plt.xlabel("number of steps")
    plt.ylabel(r"mean $\hat{\tau}$")
    if output:
        plt.savefig(output)
    if not noshow:
        plt.show()
    plt.close()

    return


# plot the time series
def timeseries(steps, labels, ndim, discard, output=None, noshow=False):
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)

    for i in range(ndim):
        ax = axes[i]
        ax.plot(steps[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(steps))
        ax.set_ylabel("$" + labels[i] + "$")
        ax.axvline(x=discard, linestyle="--", color="red")
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    if output:
        plt.savefig(output)
    if not noshow:
        plt.show()
    plt.close()

    return


# print run information
def runlog(timeelapsed, samples, discard, converged, file=sys.stdout):
    if file != sys.stdout:
        file = open(file, "w")

    print("## run.log", file=file)
    print("# information regarding the execution of this program", file=file)
    print("# the execution time has the format hours:minutes:seconds", file=file)
    print("", file=file)

    print(f"time: {timeelapsed}", file=file)
    print(f"converged: {converged > samples}", file=file)
    print(f"samples: {samples}", file=file)
    print(f"discard: {discard}", file=file)

    if file != sys.stdout:
        file.close()

    return
