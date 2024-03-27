from abc import ABC, abstractmethod

import numpy as np
import convex_composite.function as fun
import convex_composite.optimizer as optim
import collections

import matplotlib.pyplot as plt

from pathlib import Path

class Experiment(ABC):
    def __init__(self, name, n, config, optimizer_configs, num_runs):
        self.name = name
        self.config = config
        self.optimizer_configs = optimizer_configs
        self.num_runs = num_runs
        self.n = n

        #assigned by subclass
        self.linear_transform = None
        self.diffable = None
        self.proxable = None
        np.random.seed(config["seed"])

    @abstractmethod
    def compute_optimum(self, x_init):
        pass

    @abstractmethod
    def get_filename(self):
        pass

    def run(self, overwrite_file=False):
        filename = self.get_filename()
        suffix = ".npz"

        if overwrite_file or not Path(filename + suffix).is_file():
            self.compute_optimum(np.zeros(self.n))

            self.objective_values = dict()
            self.calls_linesearch = dict()
            self.calls_operator = dict()

            for optimizer_config in self.optimizer_configs:
                self.objective_values[optimizer_config["name"]] = [[] for i in range(self.num_runs)]
                self.calls_linesearch[optimizer_config["name"]] = [[] for i in range(self.num_runs)]
                self.calls_operator[optimizer_config["name"]] = [[] for i in range(self.num_runs)]

            fun.counting_enabled = True

            for run in range(self.num_runs):
                x_init = eval(self.config["init_proc"] + "(" + str(self.n) + ")")
                problem = optim.CompositeOptimizationProblem(x_init, self.diffable, self.proxable)

                for optimizer_config in self.optimizer_configs:
                    print(optimizer_config["name"])
                    self.linear_transform.reset_num_calls()

                    def callback(k, i, gamma, x, res):
                        fun.counting_enabled = False
                        objective_value = problem.eval_objective(x)
                        fun.counting_enabled = True

                        self.objective_values[optimizer_config["name"]][run].append(objective_value)
                        self.calls_linesearch[optimizer_config["name"]][run].append(i)
                        self.calls_operator[optimizer_config["name"]][run].append(self.linear_transform.get_num_calls())


                        if k % 300 == 0:
                            print(k, i, self.linear_transform.get_num_calls(), gamma, objective_value, objective_value - self.opt,
                                  res, np.linalg.norm(problem.diffable.eval_gradient(x)))

                        #calls = k * optimizer_config["class"].evals_per_iteration + i * optimizer_config["class"].evals_per_linesearch
                        if objective_value - self.opt < self.config["tol"]: #or calls > self.config["maxcalls"]:
                            return True

                        return False

                    optimizer = optimizer_config["class"](optimizer_config["parameters"], problem, callback)
                    if "gamma_init" not in self.config:
                        optimizer.init_gamma()
                    else:
                        optimizer_config["parameters"].gamma_init = self.config["gamma_init"]

                    optimizer.run()

            np.savez(filename, opt=self.opt,
                     objective_values=self.objective_values,
                     calls_linesearch=self.calls_linesearch,
                     calls_operator=self.calls_operator
            )

        else:
            cache = np.load(filename + '.npz', allow_pickle=True)
            self.opt = cache["opt"]
            self.calls_operator = cache.get("calls_operator").item()
            self.calls_linesearch = cache["calls_linesearch"].item()
            self.objective_values = cache["objective_values"].item()




    def linspace_values(self, x, y, interval):
        values = np.arange(0, x[-1], interval) * 0.

        j = 0
        for i in range(0, x[-1], interval):
            while True:
                if x[j] > i:
                    break
                j = j + 1

            # linearly interpolate the values at j-1 and j to obtain the value at i
            values[int(i / interval)] = (
                    y[j - 1]
                    + (i - x[j - 1])
                    * (y[j] - y[j - 1]) / (x[j] - x[j - 1])
            )
        return values

    def plot_mean_stdev(self, xvals, yvals, label, marker, color, refval=0., plotstdev=True, markevery=20,
                        plotevery=250):

        # compute new array with linspaced xvals with shortest length
        xvals_linspace = np.arange(0, xvals[0][-1], plotevery)
        for i in range(1, len(xvals)):
            arange = np.arange(0, xvals[i][-1], plotevery)
            if len(xvals_linspace) > len(arange):
                xvals_linspace = arange

        yvals_mean = np.zeros(len(xvals_linspace))

        for i in range(len(xvals)):
            y_values_interp = self.linspace_values(xvals[i],
                                                   yvals[i], plotevery)
            yvals_mean += y_values_interp[0:len(xvals_linspace)]

        yvals_mean = yvals_mean / len(xvals)

        plt.semilogy(xvals_linspace, yvals_mean - refval,
                     label=label,
                     marker=marker,
                     markevery=markevery,
                     color=color)

        if len(xvals) > 1 and plotstdev:
            yvals_stdev = np.zeros(len(xvals_linspace))

            for i in range(len(xvals)):
                y_values_interp = self.linspace_values(xvals[i],
                                                       yvals[i], plotevery)

                yvals_stdev += (yvals_mean - y_values_interp[0:len(xvals_linspace)]) ** 2

            yvals_stdev = np.sqrt(yvals_stdev / len(xvals))

            plt.fill_between(xvals_linspace,
                             yvals_mean - refval - yvals_stdev,
                             yvals_mean - refval + yvals_stdev,
                             alpha=0.5, facecolor=color,
                             edgecolor='white')


    def plot(self, markevery, plotevery, calls_to_lin_trans=True):
        for optimizer_config in self.optimizer_configs:
            if self.num_runs == 1:
                calls = self.calls_linesearch[optimizer_config["name"]][0]

                iters = np.arange(0, len(calls))

                if calls_to_lin_trans:
                    xvals = (np.array(iters) * optimizer_config["class"].evals_per_iteration
                                + np.array(calls) * optimizer_config["class"].evals_per_linesearch)

                    plt.semilogy(xvals[xvals <= self.config["maxcalls"]], np.array(self.objective_values[optimizer_config["name"]][0])[xvals <= self.config["maxcalls"]] - self.opt,
                             label=optimizer_config["label"],
                             marker=optimizer_config["marker"],
                             markevery=markevery,
                             color=optimizer_config["color"])
                    plt.xlabel("#calls to $A,A^\\top$")
                else:
                    plt.semilogy(iters,
                                 np.array(self.objective_values[optimizer_config["name"]][0]) - self.opt,
                                 label=optimizer_config["label"],
                                 marker=optimizer_config["marker"],
                                 markevery=markevery,
                                 color=optimizer_config["color"])
                    plt.xlabel("iteration $k$")


            else:
                xvals = []
                for calls in self.calls_linesearch[optimizer_config["name"]]:
                    iters = np.arange(0, len(calls))
                    xvals.append((np.array(iters) * optimizer_config["class"].evals_per_iteration
                             + np.array(calls) * optimizer_config["class"].evals_per_linesearch))

                self.plot_mean_stdev(xvals, self.objective_values[optimizer_config["name"]],
                                 label = optimizer_config["label"],
                                 marker = optimizer_config["marker"],
                                 color = optimizer_config["color"],
                                 refval = self.opt,
                                 plotstdev= True,
                                 markevery = markevery,
                                 plotevery = plotevery)

                plt.xlabel("#calls to $A,A^\\top$")

            plt.ylabel("$\\varphi(x) - \\varphi(x^\\star)$")


        plt.tight_layout()
        plt.legend()
