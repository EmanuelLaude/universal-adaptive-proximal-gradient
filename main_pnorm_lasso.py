import numpy as np
import convex_optim.function as fun
import convex_optim.optimizer_universal as optim_univ
import convex_optim.optimizer_base as optim_base
import matplotlib.pyplot as plt
import matplotlib
import benchmarks

class PnormLasso(benchmarks.Benchmark):
    def setup(self):
        n = self.config["n"]
        m = self.config["m"]
        k = self.config["k"]

        assert k < n
        assert n > m

        assert self.config["norm"] == self.config["power"] or self.config["norm"] == 2

        B = np.random.uniform(-1., 1., [m, n])

        v = np.random.uniform(0, 1., m)
        self.yopt = v / np.linalg.norm(v, 2)

        p = np.dot(B.T, self.yopt)

        perm = np.argsort(np.abs(p))[::-1]


        alpha = np.zeros(n)
        xi = np.random.uniform(0, 1, n)

        self.xopt = np.zeros(n)

        for i in range(n):
            if i < k:
                alpha[perm[i]] = config["lamb"] / np.abs(p[perm[i]])
            elif np.abs(p[perm[i]]) < 0.1 * config["lamb"]:
                alpha[perm[i]] = config["lamb"]
            else:
                alpha[perm[i]] = config["lamb"] * xi[perm[i]] / np.abs(p[perm[i]])


        self.A = np.matmul(B, np.diag(alpha))


        xi = np.random.uniform(0, config["rho"] / np.sqrt(k), n)
        self.xopt = np.zeros(n)

        q = np.dot(self.A.T, self.yopt)
        for i in range(n):
            if i < k:
                self.xopt[perm[i]] = xi[perm[i]] * np.sign(q[perm[i]])

        conj_power = config["power"] / (config["power"] - 1)
        conj_norm = config["norm"] / (config["norm"] - 1)

        conj_loss = fun.NormPower(conj_power, conj_norm)
        self.b = conj_loss.eval_gradient(self.yopt) + np.dot(self.A, self.xopt)

        loss = fun.NormPower(config["power"], config["norm"])
        self.fmin = loss.eval(conj_loss.eval_gradient(self.yopt)) + config["lamb"] * np.sum(np.abs(self.xopt))

        # x_init = np.zeros(n)#np.random.rand(n)

        self.linear_transform = fun.LinearTransform(self.A)
        self.diffable = fun.AffineCompositeLoss(loss, self.linear_transform, b=self.b)
        self.proxable = fun.FunctionTransform(fun.OneNorm(), rho=config["lamb"])

        return n

    def setup_problem(self, x_init):
        return optim_base.CompositeOptimizationProblem(x_init, self.diffable, self.proxable)


    def get_fmin(self):
        return (self.fmin, self.xopt)


    def get_filename(self):
        return ("results/" + self.name + "_config_"
            + self.config["name"]
            + "_num_runs_" + str(self.num_runs)
            + "_seed_" + str(self.config["seed"])
            + "_lamb_" + str(self.config["lamb"])
            + "_power_" + str(self.config["power"]))



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
        "ftol": 1e-15,
        "init_proc": "np.zeros",
        "verbose": 200
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
        "maxit": 40000,
        "maxcalls": 40000,
        "epsilon": 1e-12,
        "alpha": 0,
        "ftol": 1e-15,
        "init_proc": "np.zeros",
        "verbose": 500
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
        "ftol": 1e-15,
        "init_proc": "np.zeros",
        "verbose": 200
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
        "ftol": 1e-15,
        "init_proc": "np.zeros",
        "verbose": 200
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
        "ftol": 1e-15,
        "init_proc": "np.zeros",
        "verbose": 200
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
        "ftol": 1e-15,
        "init_proc": "np.zeros",
        "verbose": 200
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
        "ftol": 1e-15,
        "init_proc": "np.zeros",
        "verbose": 200
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
        "ftol": 1e-15,
        "init_proc": "np.zeros",
        "verbose": 200
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
        "ftol": 1e-15,
        "init_proc": "np.zeros",
        "verbose": 200
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
        "ftol": 1e-15,
        "init_proc": "np.zeros",
        "verbose": 200
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
        "ftol": 1e-15,
        "init_proc": "np.zeros",
        "verbose": 200
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
        "ftol": 1e-15,
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
        }
    ]

    experiment = PnormLasso(name, config, optimizer_configs, num_runs)
    experiment.run()
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    #fig, ax = plt.subplots(figsize=(6, 5))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    fig.suptitle("$m=" + str(config["m"]) + "$, " +
                 "$n=" + str(config["n"]) + "$, " +
                 "$k=" + str(config["k"]) + "$, " +
                 "$\\lambda=" + str(config["lamb"]) + "$, " +
                 "$p=" + str(config["power"]) + "$", fontsize=12)
    ax.grid(True)
    experiment.plot_suboptimality(markevery=config["markevery"], plotevery=config["plotevery"])

    filename = experiment.get_filename()
    suffix = ".pdf"
    plt.savefig(filename + suffix, bbox_inches='tight')
    #plt.show()
    plt.close(fig)

