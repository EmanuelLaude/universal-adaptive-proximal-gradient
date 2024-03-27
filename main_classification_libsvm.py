import numpy as np
import convex_composite.function as fun
import convex_composite.optimizer as optim

import matplotlib.pyplot as plt
import matplotlib

import experiments

from libsvmdata import fetch_libsvm
from sklearn.datasets import load_svmlight_file

class PnormClassificationLibsvm(experiments.Experiment):
    def __init__(self, name, config, optimizer_configs, num_runs):
        super().__init__(name, -1, config, optimizer_configs, num_runs)
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

        self.diffable = fun.AffineCompositeLoss(self.linear_transform, self.loss, weight=1 / m)

        if self.config["regularizer"] == "ellp":
            regularizer = fun.NormPower(self.config["power"], self.config["power"], config["lamb"])
            self.diffable = fun.AdditiveComposite((self.diffable, regularizer))
            self.proxable = fun.Zero()
        elif self.config["regularizer"] == "ell1":
            self.proxable = fun.OneNorm(config["lamb"])
        elif self.config["regularizer"] == "ell2":
            self.proxable = fun.NormPower(2, 2, config["lamb"])


    def get_filename(self):
        return (self.name
            + "_loss_" + self.config["loss"]
            + "_regularizer_" + self.config["regularizer"]
            + "_name_" + self.config["name"]
            + "_num_runs_" + str(self.num_runs)
            + "_seed_" + str(self.config["seed"])
            + "_lamb_" + str(self.config["lamb"])
            + "_power_" + str(self.config["power"]))


    def compute_optimum(self, x_init):
        problem = optim.CompositeOptimizationProblem(x_init, self.diffable, self.proxable)

        self.opt = np.Inf

        [m, _] = self.A.shape

        def callback(k, i, gamma, x, res):
            prediction = np.dot(self.X, x)
            train_error = np.sum(np.sign(prediction) != self.labels)

            objective_value = problem.eval_objective(x)
            if objective_value < self.opt:
                self.opt = objective_value

            if k % 300 == 0:
                print(k, i, gamma, objective_value, res, train_error, train_error / m * 100, np.linalg.norm(problem.diffable.eval_gradient(x)))

            return False

        optimizer = self.config["reference_optimizer"](self.config["reference_params"], problem, callback)
        if "gamma_init" not in self.config:
            optimizer.init_gamma()
        else:
            self.config["reference_params"].gamma_init = self.config["gamma_init"]

        optimizer.run()



# dataset, lamb, gamma_init, gt_optimizer, gt_maxit = "covtype.binary", 0.001, 1e-6, optim.UniversalFastPGMLan, 2000
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
    # mushrooms
    ####
    {
        "loss": "power_hinge",
        "regularizer": "ell1",
        "markevery": 100,
        "plotevery": 20,
        "seed": 50,
        "name": "mushrooms",
        "lamb": 0.01,
        "power": 1.5,
        "reference_optimizer": optim.AdaPGM,
        "reference_params": optim.Parameters(epsilon=1e-14,
                                             maxit=25000,
                                             tol=1e-17),
        "maxit": 10000,
        "maxcalls": 15000,
        "epsilon": 1e-12,
        "alpha": 0,
        "tol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros"
    },
    {
        "loss": "power_hinge",
        "regularizer": "ell1",
        "markevery": 100,
        "plotevery": 20,
        "seed": 50,
        "name": "mushrooms",
        "lamb": 0.005,
        "power": 1.5,
        "reference_optimizer": optim.UniversalFastPGMLan,
        "reference_params": optim.Parameters(epsilon=1e-14,
                                             maxit=25000,
                                             tol=1e-17),
        "maxit": 10000,
        "maxcalls": 15000,
        "epsilon": 1e-12,
        "alpha": 0,
        "tol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros"
    },
    {
        "loss": "power_hinge",
        "regularizer": "ell1",
        "markevery": 100,
        "plotevery": 20,
        "seed": 50,
        "name": "mushrooms",
        "lamb": 0.001,
        "power": 1.5,
        "reference_optimizer": optim.UniversalFastPGMLan,
        "reference_params": optim.Parameters(epsilon=1e-14,
                                             maxit=25000,
                                             tol=1e-17),
        "maxit": 10000,
        "maxcalls": 15000,
        "epsilon": 1e-12,
        "alpha": 0,
        "tol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros"
    },

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
        "reference_optimizer": optim.AdaPGM,
        "reference_params": optim.Parameters(epsilon=1e-14,
                                             maxit=25000,
                                             tol=1e-17),
        "maxit": 10000,
        "maxcalls": 4000,
        "epsilon": 1e-12,
        "alpha": 0,
        "tol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros"
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
        "reference_optimizer": optim.UniversalFastPGMLan,
        "reference_params": optim.Parameters(epsilon=1e-14,
                                             maxit=25000,
                                             tol=1e-17),
        "maxit": 10000,
        "maxcalls": 4000,
        "epsilon": 1e-12,
        "alpha": 0,
        "tol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros"
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
        "reference_optimizer": optim.UniversalFastPGMLan,
        "reference_params": optim.Parameters(epsilon=1e-14,
                                             maxit=25000,
                                             tol=1e-17),
        "maxit": 10000,
        "maxcalls": 7000,
        "epsilon": 1e-12,
        "alpha": 0,
        "tol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros"
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
        "reference_optimizer": optim.AdaPGM,
        "reference_params": optim.Parameters(epsilon=1e-14,
                                             maxit=25000,
                                             tol=1e-17),
        "maxit": 10000,
        "maxcalls": 4000,
        "epsilon": 1e-12,
        "alpha": 0,
        "tol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros"
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
        "reference_optimizer": optim.UniversalFastPGMLan,
        "reference_params": optim.Parameters(epsilon=1e-14,
                                             maxit=25000,
                                             tol=1e-17),
        "maxit": 10000,
        "maxcalls": 4000,
        "epsilon": 1e-12,
        "alpha": 0,
        "tol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros"
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
        "reference_optimizer": optim.UniversalFastPGMLan,
        "reference_params": optim.Parameters(epsilon=1e-14,
                                             maxit=25000,
                                             tol=1e-17),
        "maxit": 10000,
        "maxcalls": 7000,
        "epsilon": 1e-12,
        "alpha": 0,
        "tol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros"
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
        "reference_optimizer": optim.AdaPGM,
        "reference_params": optim.Parameters(epsilon=1e-14,
                                             maxit=25000,
                                             tol=1e-17),
        "gamma_init": 1e-7,
        "maxit": 15000,
        "maxcalls": 15000,
        "epsilon": 1e-12,
        "alpha": 0,
        "tol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros"
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
        "reference_optimizer": optim.AdaPGM,
        "reference_params": optim.Parameters(epsilon=1e-14,
                                             maxit=25000,
                                             tol=1e-17),
        "maxit": 15000,
        "maxcalls": 15000,
        "epsilon": 1e-12,
        "alpha": 0,
        "tol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros"
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
        "reference_optimizer": optim.UniversalFastPGMLan,
        "reference_params": optim.Parameters(epsilon=1e-14,
                                             maxit=25000,
                                             tol=1e-17),
        "maxit": 15000,
        "maxcalls": 15000,
        "epsilon": 1e-12,
        "alpha": 0,
        "tol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros"
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
        "reference_optimizer": optim.AdaPGM,
        "reference_params": optim.Parameters(epsilon=1e-14,
                                             maxit=10000,
                                             tol=1e-17),
        "maxit": 4000,
        "maxcalls": 8000,
        "epsilon": 1e-12,
        "alpha": 0,
        "tol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros"
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
        "reference_optimizer": optim.AdaPGM,
        "reference_params": optim.Parameters(epsilon=1e-14,
                                             maxit=10000,
                                             tol=1e-17),
        "maxit": 4000,
        "maxcalls": 8000,
        "epsilon": 1e-12,
        "alpha": 0,
        "tol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros"
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
        "reference_optimizer": optim.UniversalFastPGMLan,
        "reference_params": optim.Parameters(epsilon=1e-14,
                                             maxit=10000,
                                             tol=1e-17),
        "maxit": 4000,
        "maxcalls": 8000,
        "epsilon": 1e-12,
        "alpha": 0,
        "tol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros"
    },
    # {
    #     "loss": "power_hinge",
    #     "regularizer": "ell1",
    #     "markevery": 200,
    #     "plotevery": 20,
    #     "seed": 50,
    #     "name": "australian_scale",
    #     "lamb": 0.01,
    #     "power": 1.5,
    #     "reference_optimizer": optim.AdaPGM,
    #     "reference_params": optim.Parameters(epsilon=1e-14,
    #                                          maxit=10000,
    #                                          tol=1e-17),
    #     "maxit": 4000,
    #     "maxcalls": 5000,
    #     "epsilon": 1e-12,
    #     "alpha": 0,
    #     "tol": 1e-13,
    #     "add_bias": True,
    #     "init_proc": "np.zeros"
    # },
    # {
    #     "loss": "power_hinge",
    #     "regularizer": "ell1",
    #     "markevery": 200,
    #     "plotevery": 20,
    #     "seed": 50,
    #     "name": "australian_scale",
    #     "lamb": 0.005,
    #     "power": 1.5,
    #     "reference_optimizer": optim.AdaPGM,
    #     "reference_params": optim.Parameters(epsilon=1e-14,
    #                                          maxit=10000,
    #                                          tol=1e-17),
    #     "maxit": 4000,
    #     "maxcalls": 8000,
    #     "epsilon": 1e-12,
    #     "alpha": 0,
    #     "tol": 1e-13,
    #     "add_bias": True,
    #     "init_proc": "np.zeros"
    # },
    # {
    #     "loss": "power_hinge",
    #     "regularizer": "ell1",
    #     "markevery": 500,
    #     "plotevery": 20,
    #     "seed": 50,
    #     "name": "australian_scale",
    #     "lamb": 0.001,
    #     "power": 1.5,
    #     "reference_optimizer": optim.UniversalFastPGMLan,
    #     "reference_params": optim.Parameters(epsilon=1e-14,
    #                                          maxit=100000,
    #                                          tol=1e-17),
    #     "maxit": 100000,
    #     "maxcalls": 12500,
    #     "epsilon": 1e-12,
    #     "alpha": 0,
    #     "tol": 1e-14,
    #     "add_bias": True,
    #     "init_proc": "np.zeros"
    # },
    # {
    #     "loss": "power_hinge",
    #     "regularizer": "ell1",
    #     "markevery": 2000,
    #     "plotevery": 20,
    #     "seed": 50,
    #     "name": "australian_scale",
    #     "lamb": 0.0005,
    #     "power": 1.5,
    #     "reference_optimizer": optim.UniversalFastPGMLan,
    #     "reference_params": optim.Parameters(epsilon=1e-14,
    #                                          maxit=100000,
    #                                          tol=1e-17),
    #     "maxit": 100000,
    #     "maxcalls": 70000,
    #     "epsilon": 1e-12,
    #     "alpha": 0,
    #     "tol": 1e-14,
    #     "add_bias": True,
    #     "init_proc": "np.zeros"
    # },
    # {
    #     "loss": "power_hinge",
    #     "regularizer": "ell1",
    #     "markevery": 200,
    #     "plotevery": 20,
    #     "seed": 50,
    #     "name": "breast-cancer_scale",
    #     "lamb": 0.01,
    #     "power": 1.5,
    #     "reference_optimizer": optim.AdaPGM,
    #     "reference_params": optim.Parameters(epsilon=1e-14,
    #                                          maxit=100000,
    #                                          tol=1e-17),
    #     "maxit": 17000,
    #     "maxcalls": 5000,
    #     "epsilon": 1e-12,
    #     "alpha": 0,
    #     "tol": 1e-14,
    #     "add_bias": True,
    #     "init_proc": "np.zeros"
    # },
    # {
    #     "loss": "power_hinge",
    #     "regularizer": "ell1",
    #     "markevery": 300,
    #     "plotevery": 20,
    #     "seed": 50,
    #     "name": "breast-cancer_scale",
    #     "lamb": 0.005,
    #     "power": 1.5,
    #     "reference_optimizer": optim.AdaPGM,
    #     "reference_params": optim.Parameters(epsilon=1e-14,
    #                                          maxit=100000,
    #                                          tol=1e-17),
    #     "maxit": 8000,
    #     "maxcalls": 5000,
    #     "epsilon": 1e-12,
    #     "alpha": 0,
    #     "tol": 1e-14,
    #     "add_bias": True,
    #     "init_proc": "np.zeros"
    # },
    # {
    #     "loss": "power_hinge",
    #     "regularizer": "ell1",
    #     "markevery": 300,
    #     "plotevery": 20,
    #     "seed": 50,
    #     "name": "breast-cancer_scale",
    #     "lamb": 0.001,
    #     "power": 1.5,
    #     "reference_optimizer": optim.AdaPGM,
    #     "reference_params": optim.Parameters(epsilon=1e-14,
    #                                          maxit=100000,
    #                                          tol=1e-17),
    #     "maxit": 8000,
    #     "maxcalls": 10000,
    #     "epsilon": 1e-12,
    #     "alpha": 0,
    #     "tol": 1e-14,
    #     "add_bias": True,
    #     "init_proc": "np.zeros"
    # },



######
# logistic with power
#####
    # {
    #     "loss": "logistic",
    #     "regularizer": "ellp",
    #     "markevery": 100,
    #     "plotevery": 20,
    #     "seed": 50,
    #     "name": "australian_scale",
    #     "lamb": 0.005,
    #     "power": 1.5,
    #     "reference_optimizer": optim.AdaPGM,
    #     "reference_params": optim.Parameters(epsilon=1e-14,
    #                                          maxit=10000,
    #                                          tol=1e-17),
    #     "maxit": 4000,
    #     "maxcalls": 8000,
    #     "epsilon": 1e-12,
    #     "alpha": 0,
    #     "tol": 1e-13,
    #     "add_bias": True,
    #     "init_proc": "np.zeros"
    # },
    # {
    #     "loss": "logistic",
    #     "regularizer": "ellp",
    #     "markevery": 500,
    #     "plotevery": 20,
    #     "seed": 50,
    #     "name": "australian_scale",
    #     "lamb": 0.001,
    #     "power": 1.5,
    #     "reference_optimizer": optim.UniversalFastPGMLan,
    #     "reference_params": optim.Parameters(epsilon=1e-14,
    #                                          maxit=100000,
    #                                          tol=1e-17),
    #     "maxit": 100000,
    #     "maxcalls": 12500,
    #     "epsilon": 1e-12,
    #     "alpha": 0,
    #     "tol": 1e-14,
    #     "add_bias": True,
    #     "init_proc": "np.zeros"
    # },
    # {
    #     "loss": "logistic",
    #     "regularizer": "ellp",
    #     "markevery": 100,
    #     "plotevery": 20,
    #     "seed": 50,
    #     "name": "australian_scale",
    #     "lamb": 0.01,
    #     "power": 1.5,
    #     "reference_optimizer": optim.UniversalFastPGMLan,
    #     "reference_params": optim.Parameters(epsilon=1e-14,
    #                                          maxit=100000,
    #                                          tol=1e-17),
    #     "maxit": 100000,
    #     "maxcalls": 10000,
    #     "epsilon": 1e-12,
    #     "alpha": 0,
    #     "tol": 1e-14,
    #     "add_bias": True,
    #     "init_proc": "np.zeros"
    # },
    # {
    #     "loss": "logistic",
    #     "regularizer": "ellp",
    #     "markevery": 100,
    #     "plotevery": 100,
    #     "seed": 50,
    #     "name": "breast-cancer_scale",
    #     "lamb": 0.01,
    #     "power": 1.5,
    #     "reference_optimizer": optim.AdaPGM,
    #     "reference_params": optim.Parameters(epsilon=1e-14,
    #                                          maxit=100000,
    #                                          tol=1e-17),
    #     "maxit": 17000,
    #     "maxcalls": 8000,
    #     "epsilon": 1e-12,
    #     "alpha": 0,
    #     "tol": 1e-14,
    #     "add_bias": True,
    #     "init_proc": "np.zeros"
    # },
    # {
    #     "loss": "logistic",
    #     "regularizer": "ellp",
    #     "markevery": 100,
    #     "plotevery": 20,
    #     "seed": 50,
    #     "name": "breast-cancer_scale",
    #     "lamb": 0.005,
    #     "power": 1.5,
    #     "reference_optimizer": optim.AdaPGM,
    #     "reference_params": optim.Parameters(epsilon=1e-14,
    #                                          maxit=100000,
    #                                          tol=1e-17),
    #     "maxit": 8000,
    #     "maxcalls": 2000,
    #     "epsilon": 1e-12,
    #     "alpha": 0,
    #     "tol": 1e-14,
    #     "add_bias": True,
    #     "init_proc": "np.zeros"
    # },
    # {
    #     "loss": "logistic",
    #     "regularizer": "ellp",
    #     "markevery": 100,
    #     "plotevery": 20,
    #     "seed": 50,
    #     "name": "breast-cancer_scale",
    #     "lamb": 0.001,
    #     "power": 1.5,
    #     "reference_optimizer": optim.AdaPGM,
    #     "reference_params": optim.Parameters(epsilon=1e-14,
    #                                          maxit=100000,
    #                                          tol=1e-17),
    #     "maxit": 8000,
    #     "maxcalls": 10000,
    #     "epsilon": 1e-12,
    #     "alpha": 0,
    #     "tol": 1e-14,
    #     "add_bias": True,
    #     "init_proc": "np.zeros"
    # },
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
        "reference_optimizer": optim.AdaPGM,
        "reference_params": optim.Parameters(epsilon=1e-14,
                                             maxit=10000,
                                             tol=1e-17),
        "maxit": 4000,
        "maxcalls": 8000,
        "epsilon": 1e-12,
        "alpha": 0,
        "tol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros"
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
        "reference_optimizer": optim.AdaPGM,
        "reference_params": optim.Parameters(epsilon=1e-14,
                                             maxit=10000,
                                             tol=1e-17),
        "maxit": 4000,
        "maxcalls": 8000,
        "epsilon": 1e-12,
        "alpha": 0,
        "tol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros"
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
        "reference_optimizer": optim.UniversalFastPGMLan,
        "reference_params": optim.Parameters(epsilon=1e-14,
                                             maxit=10000,
                                             tol=1e-17),
        "maxit": 4000,
        "maxcalls": 8000,
        "epsilon": 1e-12,
        "alpha": 0,
        "tol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros"
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
        "reference_optimizer": optim.AdaPGM,
        "reference_params": optim.Parameters(epsilon=1e-14,
                                             maxit=25000,
                                             tol=1e-17),
        "maxit": 10000,
        "maxcalls": 15000,
        "epsilon": 1e-12,
        "alpha": 0,
        "tol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros"
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
        "reference_optimizer": optim.UniversalFastPGMLan,
        "reference_params": optim.Parameters(epsilon=1e-14,
                                             maxit=25000,
                                             tol=1e-17),
        "maxit": 10000,
        "maxcalls": 15000,
        "epsilon": 1e-12,
        "alpha": 0,
        "tol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros"
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
        "reference_optimizer": optim.UniversalFastPGMLan,
        "reference_params": optim.Parameters(epsilon=1e-14,
                                             maxit=25000,
                                             tol=1e-17),
        "maxit": 10000,
        "maxcalls": 15000,
        "epsilon": 1e-12,
        "alpha": 0,
        "tol": 1e-13,
        "add_bias": True,
        "init_proc": "np.zeros"
    }
]

for config in configs:
    optimizer_configs = [
        {
            "marker": "^",
            "color": "orange",
            "name": "LanUnivFastPGM",
            "label": "AC-FGM",
            "class": optim.UniversalFastPGMLan,
            "parameters": optim.Parameters(epsilon=config["epsilon"],
                                           maxit=config["maxit"],
                                           tol=1e-16,
                                           alpha=config["alpha"])
        },
        {
            "marker": "*",
            "color": "blue",
            "name": "AdaPGM1.2",
            "label": "AdaPG$^{1.2, 0.6}$",
            "class": optim.AdaPGM,
            "parameters": optim.Parameters(pi=1.2,
                                           maxit=config["maxit"],
                                           tol=1e-16)
        },
        {
            "marker": "o",
            "color": "purple",
            "name": "AdaPGM1.5",
            "label": "AdaPG$^{1.5, 0.75}$",
            "class": optim.AdaPGM,
            "parameters": optim.Parameters(pi=1.5,
                                           maxit=config["maxit"],
                                           tol=1e-16)
        },
        {
            "marker": "x",
            "color": "darkgreen",
            "name": "AdaPGM2.0",
            "label": "AdaPG$^{2, 1}$",
            "class": optim.AdaPGM,
            "parameters": optim.Parameters(pi=2.0,
                                           maxit=config["maxit"],
                                           tol=1e-16)
        },
        {
            "marker": "D",
            "color": "black",
            "name": "NesterovUnivFastPGM",
            "label": "F-NUPG",
            "class": optim.UniversalFastPGMNesterov,
            "parameters": optim.Parameters(epsilon=config["epsilon"],
                                           maxit=config["maxit"],
                                           tol=1e-16)
        },
        {
            "marker": "X",
            "color": "red",
            "name": "NesterovUnivPGM",
            "label": "NUPG",
            "class": optim.UniversalPGMNesterov,
            "parameters": optim.Parameters(epsilon=config["epsilon"],
                                           maxit=config["maxit"],
                                           tol=1e-16)
        }
    ]

    print(config["name"] + ", loss=" + config["loss"]
          + ", regularizer=" + config["regularizer"]
          + ", lambda=" + str(config["lamb"])
                 + ", p=" + str(config["power"]))

    experiment = PnormClassificationLibsvm(name, config, optimizer_configs, num_runs)
    experiment.run()
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle(config["name"] + " ($\\lambda=$" + str(config["lamb"])
                 + ", $p=$" + str(config["power"]) + ")", fontsize=16)
    ax.grid(True)
    experiment.plot(markevery=config["markevery"], plotevery=config["plotevery"], calls_to_lin_trans=True)

    filename = experiment.get_filename()
    suffix = ".pdf"
    plt.savefig(filename + suffix, bbox_inches='tight')
    #plt.show()
