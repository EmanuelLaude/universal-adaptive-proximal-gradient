import numpy as np
import convex_optim.function as fun
import convex_optim.optimizer_universal as optim_univ
import convex_optim.optimizer_base as optim_base

import matplotlib.pyplot as plt
import matplotlib

import benchmarks

from libsvmdata import fetch_libsvm
from sklearn.datasets import load_svmlight_file

class ClassificationLibsvm(benchmarks.Benchmark):
    def setup(self):
        np.random.seed(config["seed"])
        self.num_runs = num_runs
        self.config = config
        self.optimizer_configs = optimizer_configs


        if self.config["name"] == "mushrooms":
            data = load_svmlight_file("datasets/mushrooms")
            self.X, targets = data[0].toarray(), data[1]

        elif self.config["name"] == "leukemia":
            data = load_svmlight_file("datasets/leu.bz2")
            self.X, targets = data[0].toarray(), data[1]
        else:
            self.X, targets = fetch_libsvm(self.config["name"])


        self.labels = np.copy(targets)
        self.labels[targets == min(targets)] = -1
        self.labels[targets == max(targets)] = +1
        if not isinstance(self.X, np.ndarray):
            self.X = self.X.toarray()

        if self.config["add_bias"]:
            self.X = np.column_stack((self.X, np.ones(self.X.shape[0])))

        self.A = -self.X * self.labels[:, None]



        self.loss = None
        if self.config["loss"] == "power_hinge":
            self.loss = fun.PowerHingeLoss(config["power"])
        elif self.config["loss"] == "logistic":
            self.loss = fun.LogisticLoss()



        self.linear_transform = fun.LinearTransform(self.A)
        [m, self.n] = self.A.shape

        self.diffable = fun.FunctionTransform(
            fun.AffineCompositeLoss(self.loss, self.linear_transform),
            rho = 1 / m
        )

        if self.config["regularizer"] == "ellp":
            regularizer = fun.FunctionTransform(
                fun.NormPower(self.config["power"], self.config["power"]),
                rho=config["lamb"]
            )
            self.diffable = fun.AdditiveComposite((self.diffable, regularizer))
            self.proxable = fun.Zero()
        elif self.config["regularizer"] == "ell1":
            self.proxable = fun.FunctionTransform(
                fun.OneNorm(),
                rho=config["lamb"]
            )
        elif self.config["regularizer"] == "ell2":
            self.proxable = fun.FunctionTransform(
                fun.NormPower(2, 2),
                rho=config["lamb"]
            )

        return self.n


    def get_filename(self):
        return ("results/" + self.name
            + "_loss_" + self.config["loss"]
            + "_regularizer_" + self.config["regularizer"]
            + "_name_" + self.config["name"]
            + "_num_runs_" + str(self.num_runs)
            + "_seed_" + str(self.config["seed"])
            + "_lamb_" + str(self.config["lamb"])
            + "_power_" + str(self.config["power"]))


    def setup_problem(self, x_init):
        return optim_base.CompositeOptimizationProblem(x_init, self.diffable, self.proxable)


    def get_fmin(self):
        problem = optim_base.CompositeOptimizationProblem(np.zeros(self.n), self.diffable, self.proxable)

        self.fmin = np.Inf
        self.xopt = np.zeros(self.n)

        [m, _] = self.A.shape

        def callback(x, status):
            prediction = np.dot(self.X, x)
            train_error = np.sum(np.sign(prediction) != self.labels)

            objective_value = problem.eval_objective(x)
            if objective_value < self.fmin:
                self.fmin = objective_value
                self.xopt = x

            if status.nit % 500 == 0:
                print(status.nit, status.cumsum_backtracks, status.gamma, objective_value, status.res, train_error, train_error / m * 100, np.linalg.norm(problem.diffable.eval_gradient(x)))

            return False

        if "gamma_init" not in self.config:
            self.init_gamma(problem, self.config["reference_params"])
        else:
            self.config["reference_params"].gamma_init = self.config["gamma_init"]
        optimizer = self.config["reference_optimizer"](self.config["reference_params"], problem, callback)

        optimizer.run()

        return self.fmin, self.xopt



# dataset, lamb, gamma_init, gt_optimizer, gt_maxit = "covtype.binary", 0.001, 1e-6, optim_univ.AutoConditionedFastGradientMethod, 2000
# dataset, lamb, gamma_init, gt_optimizer, gt_maxit = "rcv1.binary", 0.001, 10., optim.AdaPGM, 2000
# dataset, lamb, gamma_init, gt_optimizer, gt_maxit = "real-sim", 0.001, 10., optim.AdaPGM, 2000

# dataset, lamb, gamma_init, gt_optimizer, gt_maxit = "w8a", 0.005, 1, optim.AdaPGM, 5000
# dataset, lamb, gamma_init, gt_optimizer, gt_maxit  = "a9a", 0.005, 1., optim.AdaPGM, 2000
# dataset, lamb, gt_optimizer, gt_maxit = "madelon", 0.01, optim.AdaPGM, 2000
# dataset, lamb, gamma_init, gt_optimizer, gt_maxit = "australian_scale", 0.01, 1, optim.AdaPGM, 2000
# dataset, lamb, gt_optimizer, gt_maxit = "australian_scale", 0.001, optim.AdaPGM, 2000
# dataset, lamb, gt_optimizer, gt_maxit = "diabetes_scale", 0.001, optim.AdaPGM, 2000
# dataset, lamb, gt_optimizer, gt_maxit = "ijcnn1", 0.001, optim.AdaPGM, 2000


name = "classification_libsvm"
num_runs = 1

# gamma_init undefined triggers call to optimizer.init_gamma()

configs = [
######
# power hinge with ell1
#####
    ####
    # phishing
    #####
    {
        "loss": "power_hinge",
        "regularizer": "ell1",
        "markevery": 100,
        "plotevery": 20,
        "seed": 50,
        "name": "phishing",
        "lamb": 0.01,
        "power": 1.5,
        "reference_optimizer": optim_univ.AutoConditionedFastGradientMethod,
        "reference_params": optim_univ.Parameters(epsilon=1e-14,
                                             maxit=25000,
                                             tol=1e-17),
        "maxit": 10000,
        "maxcalls": 4000,
        "epsilon": 1e-12,
        "alpha": 0,
        "ftol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros",
        "verbose": 200
    },
    {
        "loss": "power_hinge",
        "regularizer": "ell1",
        "markevery": 100,
        "plotevery": 20,
        "seed": 50,
        "name": "phishing",
        "lamb": 0.005,
        "power": 1.5,
        "reference_optimizer": optim_univ.AutoConditionedFastGradientMethod,
        "reference_params": optim_univ.Parameters(epsilon=1e-14,
                                             maxit=25000,
                                             tol=1e-17),
        "maxit": 10000,
        "maxcalls": 4000,
        "epsilon": 1e-12,
        "alpha": 0,
        "ftol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros",
        "verbose": 200
    },
    {
        "loss": "power_hinge",
        "regularizer": "ell1",
        "markevery": 200,
        "plotevery": 20,
        "seed": 50,
        "name": "phishing",
        "lamb": 0.001,
        "power": 1.5,
        "reference_optimizer": optim_univ.AdaptiveProximalGradientMethod,
        "reference_params": optim_univ.Parameters(epsilon=1e-14,
                                             maxit=25000,
                                             tol=1e-17),
        "maxit": 10000,
        "maxcalls": 7000,
        "epsilon": 1e-12,
        "alpha": 0,
        "ftol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros",
        "verbose": 200
    },
    ####
    # a9a
    ####
    {
        "loss": "power_hinge",
        "regularizer": "ell1",
        "markevery": 100,
        "plotevery": 20,
        "seed": 50,
        "name": "a9a",
        "lamb": 0.01,
        "power": 1.5,
        "reference_optimizer": optim_univ.AdaptiveProximalGradientMethod,
        "reference_params": optim_univ.Parameters(epsilon=1e-14,
                                             maxit=25000,
                                             tol=1e-17),
        "maxit": 10000,
        "maxcalls": 4000,
        "epsilon": 1e-12,
        "alpha": 0,
        "ftol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros",
        "verbose": 200
    },
    {
        "loss": "power_hinge",
        "regularizer": "ell1",
        "markevery": 100,
        "plotevery": 20,
        "seed": 50,
        "name": "a9a",
        "lamb": 0.005,
        "power": 1.5,
        "reference_optimizer": optim_univ.AdaptiveProximalGradientMethod,
        "reference_params": optim_univ.Parameters(epsilon=1e-14,
                                             maxit=25000,
                                             tol=1e-17),
        "maxit": 10000,
        "maxcalls": 4000,
        "epsilon": 1e-12,
        "alpha": 0,
        "ftol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros",
        "verbose": 200
    },
    {
        "loss": "power_hinge",
        "regularizer": "ell1",
        "markevery": 100,
        "plotevery": 20,
        "seed": 50,
        "name": "a9a",
        "lamb": 0.001,
        "power": 1.5,
        "reference_optimizer": optim_univ.AdaptiveProximalGradientMethod,
        "reference_params": optim_univ.Parameters(epsilon=1e-14,
                                             maxit=25000,
                                             tol=1e-17),
        "maxit": 10000,
        "maxcalls": 7000,
        "epsilon": 1e-12,
        "alpha": 0,
        "ftol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros",
        "verbose": 200
    },
    #####
    # w8a
    #####
    {
        "loss": "power_hinge",
        "regularizer": "ell1",
        "markevery": 200,
        "plotevery": 20,
        "seed": 50,
        "name": "w8a",
        "lamb": 0.003,
        "power": 1.5,
        "reference_optimizer": optim_univ.AdaptiveProximalGradientMethod,
        "reference_params": optim_univ.Parameters(epsilon=1e-14,
                                             maxit=10000,
                                             tol=1e-17),
        "maxit": 4000,
        "maxcalls": 8000,
        "epsilon": 1e-12,
        "alpha": 0,
        "ftol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros",
        "verbose": 200
    },
    {
        "loss": "power_hinge",
        "regularizer": "ell1",
        "markevery": 200,
        "plotevery": 20,
        "seed": 50,
        "name": "w8a",
        "lamb": 0.005,
        "power": 1.5,
        "reference_optimizer": optim_univ.AdaptiveProximalGradientMethod,
        "reference_params": optim_univ.Parameters(epsilon=1e-14,
                                             maxit=10000,
                                             tol=1e-17),
        "maxit": 4000,
        "maxcalls": 8000,
        "epsilon": 1e-12,
        "alpha": 0,
        "ftol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros",
        "verbose": 200
    },
    {
        "loss": "power_hinge",
        "regularizer": "ell1",
        "markevery": 200,
        "plotevery": 20,
        "seed": 50,
        "name": "w8a",
        "lamb": 0.001,
        "power": 1.5,
        "reference_optimizer": optim_univ.AutoConditionedFastGradientMethod,
        "reference_params": optim_univ.Parameters(epsilon=1e-14,
                                             maxit=10000,
                                             tol=1e-17),
        "maxit": 4000,
        "maxcalls": 8000,
        "epsilon": 1e-12,
        "alpha": 0,
        "ftol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros",
        "verbose": 200
    },
    ####
    # covtype.binary
    ####
    {
        "loss": "power_hinge",
        "regularizer": "ell1",
        "markevery": 40,
        "plotevery": 20,
        "seed": 50,
        "name": "covtype.binary",
        "lamb": 20,
        "power": 1.5,
        "reference_optimizer": optim_univ.AdaptiveProximalGradientMethod,
        "reference_params": optim_univ.Parameters(epsilon=1e-14,
                                             maxit=25000,
                                             tol=1e-17),
        "gamma_init": 1e-7,
        "maxit": 15000,
        "maxcalls": 15000,
        "epsilon": 1e-12,
        "alpha": 0,
        "ftol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros",
        "verbose": 100
    },
    {
        "loss": "power_hinge",
        "regularizer": "ell1",
        "markevery": 100,
        "plotevery": 20,
        "seed": 50,
        "name": "covtype.binary",
        "lamb": 10,
        "power": 1.5,
        "gamma_init": 1e-7,
        "reference_optimizer": optim_univ.AdaptiveProximalGradientMethod,
        "reference_params": optim_univ.Parameters(epsilon=1e-14,
                                             maxit=25000,
                                             tol=1e-17),
        "maxit": 15000,
        "maxcalls": 15000,
        "epsilon": 1e-12,
        "alpha": 0,
        "ftol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros",
        "verbose": 100
    },
    {
        "loss": "power_hinge",
        "regularizer": "ell1",
        "markevery": 500,
        "plotevery": 20,
        "seed": 50,
        "name": "covtype.binary",
        "lamb": 1.,
        "power": 1.5,
        "gamma_init": 1e-7,
        "reference_optimizer": optim_univ.AutoConditionedFastGradientMethod,
        "reference_params": optim_univ.Parameters(epsilon=1e-14,
                                             maxit=25000,
                                             tol=1e-17),
        "maxit": 15000,
        "maxcalls": 15000,
        "epsilon": 1e-12,
        "alpha": 0,
        "ftol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros",
        "verbose": 100
    },
######
# logistic with power
#####
    ####
    # w8a
    #####
    {
        "loss": "logistic",
        "regularizer": "ellp",
        "markevery": 100,
        "plotevery": 20,
        "seed": 50,
        "name": "w8a",
        "lamb": 0.003,
        "power": 1.5,
        "reference_optimizer": optim_univ.AdaptiveProximalGradientMethod,
        "reference_params": optim_univ.Parameters(epsilon=1e-14,
                                             maxit=10000,
                                             tol=1e-17),
        "maxit": 4000,
        "maxcalls": 8000,
        "epsilon": 1e-12,
        "alpha": 0,
        "ftol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros",
        "verbose": 200
    },
    {
        "loss": "logistic",
        "regularizer": "ellp",
        "markevery": 100,
        "plotevery": 20,
        "seed": 50,
        "name": "w8a",
        "lamb": 0.005,
        "power": 1.5,
        "reference_optimizer": optim_univ.AdaptiveProximalGradientMethod,
        "reference_params": optim_univ.Parameters(epsilon=1e-14,
                                             maxit=10000,
                                             tol=1e-17),
        "maxit": 4000,
        "maxcalls": 8000,
        "epsilon": 1e-12,
        "alpha": 0,
        "ftol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros",
        "verbose": 200
    },
    {
        "loss": "logistic",
        "regularizer": "ellp",
        "markevery": 100,
        "plotevery": 20,
        "seed": 50,
        "name": "w8a",
        "lamb": 0.001,
        "power": 1.5,
        "reference_optimizer": optim_univ.AutoConditionedFastGradientMethod,
        "reference_params": optim_univ.Parameters(epsilon=1e-14,
                                             maxit=10000,
                                             tol=1e-17),
        "maxit": 4000,
        "maxcalls": 8000,
        "epsilon": 1e-12,
        "alpha": 0,
        "ftol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros",
        "verbose": 200
    },
    ####
    # Mushrooms
    ####
    {
        "loss": "logistic",
        "regularizer": "ellp",
        "markevery": 100,
        "plotevery": 20,
        "seed": 50,
        "name": "mushrooms",
        "lamb": 0.01,
        "power": 1.5,
        "reference_optimizer": optim_univ.AdaptiveProximalGradientMethod,
        "reference_params": optim_univ.Parameters(epsilon=1e-14,
                                             maxit=25000,
                                             tol=1e-17),
        "maxit": 10000,
        "maxcalls": 15000,
        "epsilon": 1e-12,
        "alpha": 0,
        "ftol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros",
        "verbose": 200
    },
    {
        "loss": "logistic",
        "regularizer": "ellp",
        "markevery": 100,
        "plotevery": 20,
        "seed": 50,
        "name": "mushrooms",
        "lamb": 0.005,
        "power": 1.5,
        "reference_optimizer": optim_univ.AutoConditionedFastGradientMethod,
        "reference_params": optim_univ.Parameters(epsilon=1e-14,
                                             maxit=25000,
                                             tol=1e-17),
        "maxit": 10000,
        "maxcalls": 15000,
        "epsilon": 1e-12,
        "alpha": 0,
        "ftol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros",
        "verbose": 200
    },
    {
        "loss": "logistic",
        "regularizer": "ellp",
        "markevery": 100,
        "plotevery": 20,
        "seed": 50,
        "name": "mushrooms",
        "lamb": 0.001,
        "power": 1.5,
        "reference_optimizer": optim_univ.AutoConditionedFastGradientMethod,
        "reference_params": optim_univ.Parameters(epsilon=1e-14,
                                             maxit=25000,
                                             tol=1e-17),
        "maxit": 10000,
        "maxcalls": 15000,
        "epsilon": 1e-12,
        "alpha": 0,
        "ftol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros",
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
                                                tol=1e-16,
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
        },
    ]

    print(config["name"] + ", loss=" + config["loss"]
          + ", regularizer=" + config["regularizer"]
          + ", lambda=" + str(config["lamb"])
                 + ", p=" + str(config["power"]))

    experiment = ClassificationLibsvm(name, config, optimizer_configs, num_runs)
    experiment.run()
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    fig, ax = plt.subplots(figsize=(4.5, 4))
    fig.suptitle(config["name"] + " ($\\lambda=" + str(config["lamb"])
                 + "$, $p=" + str(config["power"]) + "$)", fontsize=12)
    ax.grid(True)
    experiment.plot_suboptimality(markevery=config["markevery"], plotevery=config["plotevery"])

    filename = experiment.get_filename()
    suffix = ".pdf"
    plt.savefig(filename + suffix, bbox_inches='tight')
    #plt.show()
    plt.close(fig)
