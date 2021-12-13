# imports
import matplotlib.pyplot as plt
from getdist import plots
import sys
import os


# corner plot
def corner(mcsamples, markers, output=None, noshow=False, filled_alpha=None, contour_alpha=0.5):
    colors = ["#006FED", "#E03424", "#008000", "#9c5500", "#9224e0", "#ed00e6", "#f2e400", "#00f2e4", "#6fd95f"]
    contour_colors = ["#006FED", "#E03424", "#008000", "#9c5500", "#9224e0", "#ed00e6", "#f2e400", "#00f2e4", "#6fd95f"]

    g = plots.get_subplot_plotter()
    g.settings.alpha_factor_contour_lines=contour_alpha

    if filled_alpha:
        alpha = []
        for i in range(len(mcsamples)):
            alpha.append({"alpha": filled_alpha})
        g.triangle_plot(mcsamples, filled=True, markers=markers, colors=colors, contour_colors=contour_colors, contour_args=alpha)
    else:
        g.triangle_plot(mcsamples, filled=True, markers=markers, contour_colors=contour_colors, colors=colors)

    if output:
        plt.savefig(output, transparent=True)
    if not noshow:
        plt.show()
    plt.close()

    return


# print system information
def syslog(file=sys.stdout):
    if file != sys.stdout:
        file = open(file, "w")
        print("## sys.log", file=file)
        print("# information regarding the system and the date in which this run was executed", file=file)
        print("", file=file)

    print("$ date", file=file)
    date = os.popen("date").read()[:-1]
    print(f"{date}", file=file)
    print("", file=file)

    print("$ uname -a", file=file)
    uname = os.popen("uname -a").read()[:-1]
    print(f"{uname}", file=file)
    print("", file=file)

    print("$ lscpu", file=file)
    lscpu = os.popen("lscpu").read()[:-1]
    print(f"{lscpu}", file=file)
    print("", file=file)

    if file != sys.stdout:
        file.close()

    return


# print confidence intervals in a latex table
def CIs(mcsamples, file=sys.stdout):
    if file != sys.stdout:
        file = open(file, "w")

    print("## CIs.tex", file=file)
    print("# latex table for the 1 and 2 sigma distribution of each parameter", file=file)
    print("", file=file)

    print(mcsamples.getTable(limit=1).tableTex().replace("\n\n", "\n"), file=file)
    print("", file=file)
    print(mcsamples.getTable().tableTex().replace("\n\n", "\n"), file=file)

    if file != sys.stdout:
        file.close()

    return
