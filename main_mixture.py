import convex_optim.function as fun
import convex_optim.optimizer_universal as optim_univ
import convex_optim.optimizer_base as optim_base
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import benchmarks


from pathlib import Path

class Mixture(benchmarks.Benchmark):
    def setup(self):
        self.n = config["n"]

        np.random.seed(config["seed"])

        diffables = ()
        for m, power, weight in zip(config["num_cols"], config["powers"], config["weights"]):
            diffables += (
                             fun.AffineCompositeLoss(
                                 fun.FunctionTransform(fun.NormPower(power, power), rho=weight),
                                 fun.LinearTransform(2. * np.random.randn(m, self.n) - 1.),
                                2. * np.random.rand(m) - 1.
                             ),
            )


        self.diffable = fun.AdditiveComposite(diffables)

        if config["proxable"] == "IndSimplex":
            self.proxable = fun.IndicatorSimplex()
        elif config["proxable"] == "Ind2NormBall":
            self.proxable = fun.Indicator2NormBall(config["radius"])

        return self.n

    def get_fmin(self):
        problem = optim_base.CompositeOptimizationProblem(np.zeros(self.n), self.diffable, self.proxable)

        self.fmin = np.Inf
        self.xopt = np.zeros(self.n)

        def callback(x, status):
            objective_value = problem.eval_objective(x)
            if objective_value < self.fmin:
                self.fmin = objective_value
                self.xopt = x

            self.x_opt = x
            if status.nit % 300 == 0:
                print(status.nit, status.cumsum_backtracks, status.gamma, objective_value, status.res, np.linalg.norm(problem.diffable.eval_gradient(x)), np.linalg.norm(x, 2),
                      np.sum(x + 1e-12 >= 0) / self.n, np.sum(x) - 1.)

            return False

        optimizer = self.config["reference_optimizer"](self.config["reference_params"], problem, callback)
        if "gamma_init" not in self.config:
            self.init_gamma(problem, config["reference_params"])
        else:
            self.config["reference_params"].gamma_init = self.config["gamma_init"]

        optimizer.run()

        return self.fmin, self.xopt

    def setup_problem(self, x_init):
        return optim_base.CompositeOptimizationProblem(x_init, self.diffable, self.proxable)


    def get_filename(self):
        return ("results/" + self.name + "_config_"
            + self.config["name"]
            + "_num_runs_" + str(self.num_runs)
            + "_seed_" + str(self.config["seed"])
            + "_radius_" + str(self.config["radius"])
            + "_proxable_" + str(self.config["proxable"]))






np.random.seed(5)

name = "mixture_of_probabilities"
num_runs = 1
configs = [
    {
        "name": "mix_of_probs4000",
        "n": 4000,
        "markevery": 200,
        "plotevery": 20,
        "seed": 50,
        "num_cols": [400, 300, 400, 100, 100, 300],
        "powers": [1.8, 1.7, 1.6, 1.5, 1.5, 1.5],
        "weights": [1., 1., 1., 1., 1.],
        "proxable": "Ind2NormBall",
        "radius": 0.1,
        "maxit": 6000,
        "maxcalls": 8000,
        "epsilon": 1e-12,
        "alpha": 0,
        "ftol": 1e-15,
        "init_proc": "np.zeros",
        "reference_optimizer": optim_univ.AdaptiveProximalGradientMethod,
        "reference_params": optim_univ.Parameters(epsilon=1e-14,
                                                  q=2.0,
                                                  maxit=20000,
                                                  tol=1e-15),
        "verbose": 200
    },
    {
        "name": "mix_of_probs3000",
        "n": 3000,
        "markevery": 200,
        "plotevery": 20,
        "seed": 50,
        "num_cols": [400, 300, 400, 100, 100, 300],
        "powers": [1.8, 1.7, 1.6, 1.5, 1.5, 1.5],
        "weights": [1., 1., 1., 1., 1.],
        "proxable": "Ind2NormBall",
        "radius": 0.1,
        "maxit": 6000,
        "maxcalls": 8000,
        "epsilon": 1e-12,
        "alpha": 0,
        "ftol": 1e-15,
        "init_proc": "np.zeros",
        "reference_optimizer": optim_univ.AdaptiveProximalGradientMethod,
        "reference_params": optim_univ.Parameters(epsilon=1e-14,
                                                  q=2.0,
                                                  maxit=20000,
                                                  tol=1e-15),
        "verbose": 200
    },
    {
        "name": "mix_of_probs2000",
        "n": 2000,
        "markevery": 200,
        "plotevery": 20,
        "seed": 50,
        "num_cols": [400, 300, 400, 100, 100, 300],
        "powers": [1.8, 1.7, 1.6, 1.5, 1.5, 1.5],
        "weights": [1., 1., 1., 1., 1.],
        "proxable": "Ind2NormBall",
        "radius": 0.1,
        "maxit": 6000,
        "maxcalls": 8000,
        "epsilon": 1e-12,
        "alpha": 0,
        "ftol": 1e-15,
        "init_proc": "np.zeros",
        "reference_optimizer": optim_univ.AdaptiveProximalGradientMethod,
        "reference_params": optim_univ.Parameters(epsilon=1e-14,
                                             q=2.0,
                                             maxit=20000,
                                             tol=1e-15),
        "verbose": 200
    },
    {
        "name": "mix_of_probs1000",
        "n": 1000,
        "markevery": 200,
        "plotevery": 20,
        "seed": 50,
        "num_cols": [400, 300, 400, 100, 100, 300],
        "powers": [1.8, 1.7, 1.6, 1.5, 1.5, 1.5],
        "weights": [1., 1., 1., 1., 1.],
        "proxable": "Ind2NormBall",
        "radius": 0.1,
        "maxit": 6000,
        "maxcalls": 8000,
        "epsilon": 1e-12,
        "alpha": 0,
        "ftol": 1e-15,
        "init_proc": "np.zeros",
        "reference_optimizer": optim_univ.AdaptiveProximalGradientMethod,
        "reference_params": optim_univ.Parameters(epsilon=1e-14,
                                             q=2.0,
                                             maxit=20000,
                                             tol=1e-15),
        "verbose": 200
    }
]

for config in configs:
    optimizer_configs = [
        {
            "marker": "^",
            "color": "orange",
            "name": "AC-FGM",
            "label": "AC-FGM",
            "class": optim_univ.AutoConditionedFastGradientMethod,
            "parameters": optim_univ.Parameters(epsilon=config["epsilon"],
                                                maxit=config["maxit"],
                                                tol=0,
                                                alpha=config["alpha"])
        },
        {
            "marker": "*",
            "color": "blue",
            "name": "AdaPG1.2",
            "label": "AdaPG$^{1.2, 0.6}$",
            "class": optim_univ.AdaptiveProximalGradientMethod,
            "parameters": optim_univ.Parameters(q=1.2,
                                                maxit=config["maxit"],
                                                tol=1e-16)
        },
        {
            "marker": "o",
            "color": "purple",
            "name": "AdaPG1.5",
            "label": "AdaPG$^{1.5, 0.75}$",
            "class": optim_univ.AdaptiveProximalGradientMethod,
            "parameters": optim_univ.Parameters(q=1.5,
                                                maxit=config["maxit"],
                                                tol=1e-16)
        },
        {
            "marker": "x",
            "color": "darkgreen",
            "name": "AdaPG2.0",
            "label": "AdaPG$^{2, 1}$",
            "class": optim_univ.AdaptiveProximalGradientMethod,
            "parameters": optim_univ.Parameters(q=2.0,
                                                maxit=config["maxit"],
                                                tol=1e-16)
        },
        {
            "marker": "D",
            "color": "black",
            "name": "F-NUPG",
            "label": "F-NUPG",
            "class": optim_univ.NesterovUniversalFastProximalGradientMethod,
            "parameters": optim_univ.Parameters(epsilon=config["epsilon"],
                                                maxit=config["maxit"],
                                                tol=1e-16)
        },
        {
            "marker": "X",
            "color": "red",
            "name": "NUPG",
            "label": "NUPG",
            "class": optim_univ.NesterovUniversalProximalGradientMethod,
            "parameters": optim_univ.Parameters(epsilon=config["epsilon"],
                                                maxit=config["maxit"],
                                                tol=1e-16)
        }
    ]

    experiment = Mixture(name, config, optimizer_configs, num_runs)
    experiment.run()
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    fig, ax = plt.subplots(figsize=(4.5, 4))
    fig.suptitle("$n=" + str(config["n"]) + "$, $r=" + str(config["radius"]) + "$", fontsize=12)
    ax.grid(True)
    experiment.plot_suboptimality(markevery=config["markevery"], plotevery=config["plotevery"])

    filename = experiment.get_filename()
    suffix = ".pdf"
    plt.savefig(filename + suffix, bbox_inches='tight')
    #plt.show()
    plt.close(fig)

