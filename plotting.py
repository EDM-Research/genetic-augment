import numpy as np
from matplotlib import pyplot as plt


def plot_logbook(logbook):
    def range_plot(ax, x, lower, upper, middle, title):
        ax.fill_between(gen, lower, upper, color="gray", alpha=0.25)
        ax.plot(x, lower, label="min", color="black")
        ax.plot(x, upper, label="max", color="black")
        ax.plot(x, middle, label="avg", color="black")

        ax.set_title(title)
        ax.set_xlabel("generation")

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    gen = logbook.select("gen")

    # fitness
    fit_min = np.array(logbook.chapters["fitness"].select("min"))
    fit_avg = np.array(logbook.chapters["fitness"].select("avg"))
    fit_max = np.array(logbook.chapters["fitness"].select("max"))

    range_plot(ax[0], gen, fit_min[:,0], fit_max[:,0], fit_avg[:,0], "Variance")
    range_plot(ax[1], gen, fit_min[:,1], fit_max[:,1], fit_avg[:,1], "Distance")

    # length
    size_min = logbook.chapters["size"].select("min")
    size_avg = logbook.chapters["size"].select("avg")
    size_max = logbook.chapters["size"].select("max")

    range_plot(ax[2], gen, size_min, size_max, size_avg, "Length")

    plt.show()


def plot_frontier(population, hof, selected):
    f, ax = plt.subplots(1,1, figsize=(7, 4))

    wvalues = np.array([i.fitness.wvalues for i in population if i not in hof])
    if len(wvalues) > 0:
        ax.scatter(wvalues[:,0], wvalues[:,1], label="population", color="gray")

    wvalues = np.array([i.fitness.wvalues for i in hof if i not in selected])
    if len(wvalues) > 0:
        ax.scatter(wvalues[:,0], wvalues[:,1], label="frontier", color="black")

    wvalues = np.array([selected.fitness.wvalues])
    if len(wvalues) > 0:
        ax.scatter(wvalues[:,0], wvalues[:,1], label="selected", color="red")

    ax.set_xlabel("Variance")
    ax.set_ylabel("Distance")

    f.legend()

    f.tight_layout()

    plt.show()