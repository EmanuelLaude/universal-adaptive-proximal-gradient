import numpy as np
import convex_composite.function as fun
import convex_composite.optimizer as optim
import matplotlib.pyplot as plt
import matplotlib
import experiments

class PnormLasso(experiments.Experiment):
    def __init__(self, name, config, optimizer_configs, num_runs):
        super().__init__(name, config["n"], config, optimizer_configs, num_runs)

        m = config["m"]
        k = config["k"]

        assert k < self.n
        assert self.n > m

        assert config["norm"] == config["power"] or config["norm"] == 2

        B = np.random.uniform(-1., 1., [m, self.n])

        v = np.random.uniform(0, 1., m)
        self.yopt = v / np.linalg.norm(v, 2)

        p = np.dot(B.T, self.yopt)

        perm = np.argsort(np.abs(p))[::-1]


        alpha = np.zeros(self.n)
        xi = np.random.uniform(0, 1, self.n)

        self.xopt = np.zeros(self.n)

        for i in range(self.n):
            if i < k:
                alpha[perm[i]] = config["lamb"] / np.abs(p[perm[i]])
            elif np.abs(p[perm[i]]) < 0.1 * config["lamb"]:
                alpha[perm[i]] = config["lamb"]
            else:
                alpha[perm[i]] = config["lamb"] * xi[perm[i]] / np.abs(p[perm[i]])


        self.A = np.matmul(B, np.diag(alpha))


        xi = np.random.uniform(0, config["rho"] / np.sqrt(k), self.n)
        self.xopt = np.zeros(self.n)

        q = np.dot(self.A.T, self.yopt)
        for i in range(self.n):
            if i < k:
                self.xopt[perm[i]] = xi[perm[i]] * np.sign(q[perm[i]])

        conj_power = config["power"] / (config["power"] - 1)
        conj_norm = config["norm"] / (config["norm"] - 1)

        conj_loss = fun.NormPower(conj_power, conj_norm)
        self.b = conj_loss.eval_gradient(self.yopt) + np.dot(self.A, self.xopt)

        loss = fun.NormPower(config["power"], config["norm"])
        self.opt = loss.eval(conj_loss.eval_gradient(self.yopt)) + config["lamb"] * np.sum(np.abs(self.xopt))

        # x_init = np.zeros(n)#np.random.rand(n)

        self.linear_transform = fun.LinearTransform(self.A)
        self.diffable = fun.AffineCompositeLoss(self.linear_transform, loss, translations=self.b)
        self.proxable = fun.OneNorm(config["lamb"])


    def get_filename(self):
        return (self.name + "_config_"
            + self.config["name"]
            + "_num_runs_" + str(self.num_runs)
            + "_seed_" + str(self.config["seed"])
            + "_lamb_" + str(self.config["lamb"])
            + "_power_" + str(self.config["power"]))

    def compute_optimum(self, x_init):
        pass


# gamma_init undefined triggers call to optimizer.init_gamma()
name = "pnorm_lasso"
num_runs = 1
configs = [
    {
        "name": "lasso100x300x30",
        "m": 100,
        "n": 300,
        "k": 30,
        "markevery": 900,
        "plotevery": 20,
        "seed": 50,
        "lamb": 1.,
        "rho": 1.,
        "power": 1.5,
        "norm": 1.5,
        "maxit": 20000,
        "maxcalls": 20000,
        "epsilon": 1e-12,
        "alpha": 0,
        "tol": 1e-15,
        "init_proc": "np.zeros"
    },
    {
        "name": "lasso200x500x60",
        "m": 200,
        "n": 500,
        "k": 60,
        "markevery": 900,
        "plotevery": 20,
        "seed": 50,
        "lamb": 1.,
        "rho": 1.,
        "power": 1.5,
        "norm": 1.5,
        "maxit": 20000,
        "maxcalls": 40000,
        "epsilon": 1e-12,
        "alpha": 0,
        "tol": 1e-15,
        "init_proc": "np.zeros"
    },
    {
        "name": "lasso500x1000x100",
        "m": 500,
        "n": 1000,
        "k": 100,
        "markevery": 900,
        "plotevery": 20,
        "seed": 50,
        "lamb": 1.,
        "rho": 1.,
        "power": 1.5,
        "norm": 1.5,
        "maxit": 50000,
        "maxcalls": 50000,
        "epsilon": 1e-12,
        "alpha": 0,
        "tol": 1e-15,
        "init_proc": "np.zeros"
    },
    {
        "name": "lasso500x1000x200",
        "m": 500,
        "n": 1000,
        "k": 200,
        "markevery": 900,
        "plotevery": 20,
        "seed": 50,
        "lamb": 1.,
        "rho": 1.,
        "power": 1.5,
        "norm": 1.5,
        "maxit": 50000,
        "maxcalls": 50000,
        "epsilon": 1e-12,
        "alpha": 0,
        "tol": 1e-15,
        "init_proc": "np.zeros"
    },
    {
        "name": "lasso100x300x30",
        "m": 100,
        "n": 300,
        "k": 30,
        "markevery": 300,
        "plotevery": 20,
        "seed": 50,
        "lamb": 1.,
        "rho": 1.,
        "power": 1.7,
        "norm": 1.7,
        "maxit": 6000,
        "maxcalls": 6000,
        "epsilon": 1e-12,
        "alpha": 0,
        "tol": 1e-15,
        "init_proc": "np.zeros"
    },
    {
        "name": "lasso200x500x60",
        "m": 200,
        "n": 500,
        "k": 60,
        "markevery": 300,
        "plotevery": 20,
        "seed": 50,
        "lamb": 1.,
        "rho": 1.,
        "power": 1.7,
        "norm": 1.7,
        "maxit": 8000,
        "maxcalls": 8000,
        "epsilon": 1e-12,
        "alpha": 0,
        "tol": 1e-15,
        "init_proc": "np.zeros"
    },
    {
        "name": "lasso500x1000x100",
        "m": 500,
        "n": 1000,
        "k": 100,
        "markevery": 300,
        "plotevery": 20,
        "seed": 50,
        "lamb": 1.,
        "rho": 1.,
        "power": 1.7,
        "norm": 1.7,
        "maxit": 8000,
        "maxcalls": 8000,
        "epsilon": 1e-12,
        "alpha": 0,
        "tol": 1e-15,
        "init_proc": "np.zeros"
    },
    {
        "name": "lasso500x1000x200",
        "m": 500,
        "n": 1000,
        "k": 200,
        "markevery": 300,
        "plotevery": 20,
        "seed": 50,
        "lamb": 1.,
        "rho": 1.,
        "power": 1.7,
        "norm": 1.7,
        "maxit": 8000,
        "maxcalls": 8000,
        "epsilon": 1e-12,
        "alpha": 0,
        "tol": 1e-15,
        "init_proc": "np.zeros"
    },
    {
        "name": "lasso100x300x30",
        "m": 100,
        "n": 300,
        "k": 30,
        "markevery": 300,
        "plotevery": 20,
        "seed": 50,
        "lamb": 1.,
        "rho": 1.,
        "power": 1.9,
        "norm": 1.9,
        "maxit": 6000,
        "maxcalls": 6000,
        "epsilon": 1e-12,
        "alpha": 0,
        "tol": 1e-15,
        "init_proc": "np.zeros"
    },
    {
        "name": "lasso200x500x60",
        "m": 200,
        "n": 500,
        "k": 60,
        "markevery": 300,
        "plotevery": 20,
        "seed": 50,
        "lamb": 1.,
        "rho": 1.,
        "power": 1.9,
        "norm": 1.9,
        "maxit": 8000,
        "maxcalls": 8000,
        "epsilon": 1e-12,
        "alpha": 0,
        "tol": 1e-15,
        "init_proc": "np.zeros"
    },
    {
        "name": "lasso500x1000x100",
        "m": 500,
        "n": 1000,
        "k": 100,
        "markevery": 300,
        "plotevery": 20,
        "seed": 50,
        "lamb": 1.,
        "rho": 1.,
        "power": 1.9,
        "norm": 1.9,
        "maxit": 8000,
        "maxcalls": 8000,
        "epsilon": 1e-12,
        "alpha": 0,
        "tol": 1e-15,
        "init_proc": "np.zeros"
    },
    {
        "name": "lasso500x1000x200",
        "m": 500,
        "n": 1000,
        "k": 200,
        "markevery": 300,
        "plotevery": 20,
        "seed": 50,
        "lamb": 1.,
        "rho": 1.,
        "power": 1.9,
        "norm": 1.9,
        "maxit": 8000,
        "maxcalls": 8000,
        "epsilon": 1e-12,
        "alpha": 0,
        "tol": 1e-15,
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

    experiment = PnormLasso(name, config, optimizer_configs, num_runs)
    experiment.run()
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle("$m=" + str(config["m"]) + "$, " +
                 "$n=" + str(config["n"]) + "$, " +
                 "$k=" + str(config["k"]) + "$, " +
                 "$\\lambda=" + str(config["lamb"]) + "$, " +
                 "$p=" + str(config["power"]) + "$", fontsize=16)
    ax.grid(True)
    experiment.plot(markevery=config["markevery"], plotevery=config["plotevery"], calls_to_lin_trans=True)

    filename = experiment.get_filename()
    suffix = ".pdf"
    plt.savefig(filename + suffix, bbox_inches='tight')
    #plt.show()

