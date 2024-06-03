from abc import ABC, abstractmethod

import numpy as np
import convex_optim.function as fun
import convex_optim.optimizer_universal as optim_univ
import convex_optim.optimizer_base as optim_base


import matplotlib.pyplot as plt

from pathlib import Path

from math import floor, log10

def fexp10(f):
    return int(floor(log10(abs(f)))) if f != 0 else 0

def fman10(f):
    return f/10**fexp10(f)


evals_per_iteration = {
    optim_univ.NesterovUniversalProximalGradientMethod: 1,
    optim_univ.NesterovUniversalFastProximalGradientMethod: 3,
    optim_univ.AdaptiveProximalGradientMethod: 2,
    optim_univ.AutoConditionedFastGradientMethod: 2
}

evals_per_linesearch = {
    optim_univ.NesterovUniversalProximalGradientMethod: 1,
    optim_univ.NesterovUniversalFastProximalGradientMethod: 1,
    optim_univ.AdaptiveProximalGradientMethod: 0,
    optim_univ.AutoConditionedFastGradientMethod: 0
}


class Benchmark(ABC):
    def __init__(self, name, config, optimizer_configs, num_runs):
        self.name = name
        self.config = config
        self.optimizer_configs = optimizer_configs
        self.num_runs = num_runs

        np.random.seed(self.config["seed"])

        self.dim = self.setup()

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def get_fmin(self):
        pass

    @abstractmethod
    def get_filename(self):
        pass

    @abstractmethod
    def setup_problem(self):
        pass

    def init_gamma(self, problem, params):
        x_init = problem.x_init
        grad_init = problem.diffable.eval_gradient(x_init)
        gamma = params.gamma_init

        gamma_prev = params.gamma_init
        x = problem.proxable.eval_prox(x_init - gamma * grad_init, gamma)
        grad_x = problem.diffable.eval_gradient(x)
        L = np.linalg.norm(grad_x - grad_init) / np.linalg.norm(x - x_init)
        if L < 1e-15:
            gamma = gamma / 2
        else:
            gamma = 1 / L
        # print(gamma_prev, gamma)
        if gamma_prev / gamma > 10. or L == 0:
            x = problem.proxable.eval_prox(x_init - gamma * grad_init, gamma)
            grad_x = problem.diffable.eval_gradient(x)
            L = np.linalg.norm(grad_x - grad_init) / np.linalg.norm(x - x_init)
            if L < 1e-15:
                gamma = gamma / 2
            else:
                gamma = 1 / L

        params.gamma_init = gamma

    def run(self, overwrite_file=False):
        filename = self.get_filename()
        suffix = ".npz"

        if overwrite_file or not Path(filename + suffix).is_file():
            (self.fmin, xopt) = self.get_fmin()

            self.objective_values = dict()
            self.calls_linesearch = dict()
        else:
            cache = np.load(filename + '.npz', allow_pickle=True)
            self.fmin = cache["fmin"]
            xopt = cache["xopt"]

            self.calls_linesearch = cache["calls_linesearch"].item()
            self.objective_values = cache["objective_values"].item()


        update_file = False

        for optimizer_config in self.optimizer_configs:
            if optimizer_config["name"] in self.objective_values:
                continue

            update_file = True
            np.random.seed(self.config["seed"])

            self.objective_values[optimizer_config["name"]] = [[] for _ in range(self.num_runs)]
            self.calls_linesearch[optimizer_config["name"]] = [[] for _ in range(self.num_runs)]

            for run in range(self.num_runs):
                x_init = eval(self.config["init_proc"] + "(" + str(self.dim) + ")")
                problem = self.setup_problem(x_init)


                def callback(x, status):
                    objective_value = problem.eval_objective(x)

                    self.objective_values[optimizer_config["name"]][run].append(objective_value)
                    self.calls_linesearch[optimizer_config["name"]][run].append(status.cumsum_backtracks)

                    if status.nit % self.config["verbose"] == 0:
                        print("    nit", status.nit, "backtracks", status.cumsum_backtracks, "objective", objective_value, "res", status.res,
                              "gamma", status.gamma, "|x-x*|", np.linalg.norm(x - xopt, 1) / self.dim)

                    if objective_value - self.fmin < self.config["ftol"]:
                        return True

                    return False

                if "gamma_init" not in self.config:
                    self.init_gamma(problem, optimizer_config["parameters"])
                else:
                    optimizer_config["parameters"].gamma_init = self.config["gamma_init"]
                optimizer = optimizer_config["class"](optimizer_config["parameters"], problem, callback)

                print(optimizer_config["name"] + str(optimizer.params.__dict__))

                optimizer.run()

        if update_file:
            np.savez(filename,
                 fmin=self.fmin,
                 xopt=xopt,
                 objective_values=self.objective_values,
                 calls_linesearch=self.calls_linesearch,
                 #calls_operator=self.calls_operator
            )


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

    def plot_suboptimality(self, markevery, plotevery):
        self.plot(self.objective_values, self.fmin, "$\\varphi(x) - \\varphi(x^\\star)$", markevery, plotevery)


    def plot(self, yvals, refval, ylabel, markevery, plotevery):
        for optimizer_config in self.optimizer_configs:
            if self.num_runs == 1:
                calls = self.calls_linesearch[optimizer_config["name"]][0]

                iters = np.arange(0, len(calls))

                xvals = (np.array(iters) * evals_per_iteration[optimizer_config["class"]]
                            + np.array(calls) * evals_per_linesearch[optimizer_config["class"]])

                plt.semilogy(xvals[xvals <= self.config["maxcalls"]], np.array(yvals[optimizer_config["name"]][0])[xvals <= self.config["maxcalls"]] - refval,
                             label=optimizer_config["label"],
                             marker=optimizer_config["marker"],
                             markevery=markevery,
                             color=optimizer_config["color"])
                plt.xlabel("number of calls to $A,A^\\top$")

            else:
                xvals = []
                for calls in self.calls_linesearch[optimizer_config["name"]]:
                    iters = np.arange(0, len(calls))
                    xvals.append((np.array(iters) * evals_per_iteration[optimizer_config["class"]]
                             + np.array(calls) * evals_per_linesearch[optimizer_config["class"]]))

                self.plot_mean_stdev(xvals, yvals[optimizer_config["name"]],
                                     label = optimizer_config["label"],
                                     marker = optimizer_config["marker"],
                                     color = optimizer_config["color"],
                                     refval = refval,
                                     plotstdev= True,
                                     markevery = markevery,
                                     plotevery = plotevery)

                plt.xlabel("number of calls to $A,A^\\top$")

            plt.ylabel(ylabel)


        plt.tight_layout()
        plt.legend()

