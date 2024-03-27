from abc import ABC, abstractmethod
import numpy as np


class OptimizationProblem(ABC):
    def __init__(self, x_init):
        self.x_init = x_init

    @abstractmethod
    def eval_objective(self, x):
        pass


class CompositeOptimizationProblem(OptimizationProblem):
    def __init__(self, x_init, diffable, proxable):
        super().__init__(x_init)
        self.diffable = diffable
        self.proxable = proxable

    def eval_objective(self, x):
        return self.diffable.eval(x) + self.proxable.eval(x)


class Parameters:
    def __init__(self, pi = 1., epsilon = 1e-12, gamma_init = 1., maxit = 500, tol = 1e-5, initialization_procedure = 1
                 , Gamma_init = 1., alpha = 0.):
        self.maxit = maxit
        self.tol = tol
        self.gamma_init = gamma_init
        self.epsilon = epsilon
        self.pi = pi
        self.initialization_procedure = initialization_procedure
        self.Gamma_init = Gamma_init
        self.alpha = alpha

class Optimizer(ABC):
    def __init__(self, params, problem, callback = None):
        self.params = params
        self.problem = problem
        self.callback = callback

    def init_gamma(self):
        x_init = self.problem.x_init
        grad_x_init = self.problem.diffable.eval_gradient(x_init)
        gamma = self.params.gamma_init
        if self.params.gamma_init is None:
            x_eps = self.problem.proxable.eval_prox(x_init - 0.1 * grad_x_init, 0.1)  # proxgrad
            grad_x_eps = self.problem.diffable.eval_gradient(x_eps)
            L = np.linalg.norm(grad_x_init - grad_x_eps) / np.linalg.norm(x_init - x_eps)
            if L == 0:
                gamma = .1
            else:
                gamma = 1 / L

        x_prev, grad_x_prev, gamma_prev = x_init, grad_x_init, self.params.gamma_init
        x = self.problem.proxable.eval_prox(x_init - gamma * grad_x_init, gamma)
        grad_x = self.problem.diffable.eval_gradient(x)
        L = np.linalg.norm(grad_x - grad_x_prev) / np.linalg.norm(x - x_prev)
        if L == 0:
            gamma = gamma / 2
        else:
            gamma = 1 / L
        # print(gamma_prev, gamma)
        if gamma_prev / gamma > 10 or L == 0:
            x = self.problem.proxable.eval_prox(x_prev - gamma * grad_x_prev, gamma)
            grad_x = self.problem.diffable.eval_gradient(x)
            L = np.linalg.norm(grad_x - grad_x_prev) / np.linalg.norm(x - x_prev)
            if L == 0:
                gamma = gamma / 2
            else:
                gamma = 1 / L

        self.params.gamma_init = gamma
        #print(self.params.gamma_init)

    @abstractmethod
    def run(self):
        pass

class UniversalPGMNesterov(Optimizer):
    evals_per_iteration = 1
    evals_per_linesearch = 1


    def __init__(self, params, problem, callback = None):
        super().__init__(params, problem, callback)

        self.x = np.zeros(problem.x_init.shape)
        self.x[:] = problem.x_init[:]


    def run(self):
        gamma = self.params.gamma_init

        cum_num_backtracks = 0
        res = 0
        for k in range(self.params.maxit):
            if self.callback(k, cum_num_backtracks, gamma, self.x, res):
                break

            grad = self.problem.diffable.eval_gradient(self.x)
            value = self.problem.diffable.eval(self.x)

            while True:
                cum_num_backtracks = cum_num_backtracks + 1

                x = self.problem.proxable.eval_prox(self.x - gamma * grad, gamma)

                upper_bound = value + np.dot(grad, x - self.x) + 0.5 / gamma * np.dot(x - self.x, x - self.x) + self.params.epsilon / 2
                if self.problem.diffable.eval(x) <= upper_bound:
                    break

                gamma = gamma * 0.5

            gamma = gamma * 2
            res = np.linalg.norm(x - self.x, 2)

            self.x[:] = x[:]

            if res <= self.params.tol:
                break


class UniversalFastPGMNesterov(Optimizer):
    evals_per_iteration = 3
    evals_per_linesearch = 1

    def __init__(self, params, problem, callback = None):
        super().__init__(params, problem, callback)

        self.x = np.zeros(problem.x_init.shape)
        self.x[:] = problem.x_init[:]

        self.y = np.zeros(self.problem.x_init.shape)
        self.y[:] = self.x[:]

        self.A = 0


    def run(self):
        gamma = self.params.gamma_init

        v = np.zeros(self.problem.x_init.shape)
        v[:] = self.x[:]

        phi = np.zeros(self.problem.x_init.shape)
        theta = 0

        res = np.Inf
        cum_num_backtracks = 0
        for k in range(self.params.maxit):
            if self.callback(k, cum_num_backtracks, gamma, self.x, res):
                break

            while True:
                cum_num_backtracks = cum_num_backtracks + 1

                a = (gamma + np.sqrt(gamma ** 2 + 4 * gamma * self.A)) / 2
                A = self.A + a
                tau = a / A

                x = tau * v + (1 - tau) * self.y

                grad = self.problem.diffable.eval_gradient(x)
                value = self.problem.diffable.eval(x)
                x_hat = self.problem.proxable.eval_prox(v - a * grad, a)

                y = tau * x_hat + (1 - tau) * self.y

                upper_bound = (value + np.dot(grad, y - x)
                               + (0.5 / gamma) * np.dot(y - x, y - x)
                               + 0.5 * self.params.epsilon * tau)

                if self.problem.diffable.eval(y) <= upper_bound:
                    break

                gamma = gamma * 0.5

            gamma = gamma * 2

            self.y[:] = y[:]
            self.A = A

            phi = phi + a * grad
            theta = theta + a

            v = self.problem.proxable.eval_prox(self.problem.x_init - phi, theta)

            res = np.linalg.norm(x - self.x, 2)

            self.x[:] = x[:]



class AdaPGM(Optimizer):
    evals_per_iteration = 2
    evals_per_linesearch = 0

    def __init__(self, params, problem, callback = None):
        super().__init__(params, problem, callback)

        self.x = np.zeros(problem.x_init.shape)
        self.x[:] = problem.x_init[:]

        self.grad = problem.diffable.eval_gradient(self.x)

        self.gamma = self.params.gamma_init



    def run(self):
        if self.params.initialization_procedure == 0:
            x = self.problem.proxable.eval_prox(self.x - self.gamma * self.grad, self.gamma)
            gamma = self.gamma
        else:
            x_new = self.problem.proxable.eval_prox(self.x - self.params.gamma_init * self.grad, self.params.gamma_init)
            grad_x_new = self.problem.diffable.eval_gradient(x_new)
            L = np.linalg.norm(self.grad - grad_x_new) / np.linalg.norm(self.x - x_new)

            if self.params.pi - 2 * L < 0:
                self.gamma = self.params.gamma_init
            else:
                self.gamma = self.params.gamma_init * (self.params.pi * 2 * L) / (self.params.pi - 2 * L)
            gamma = self.params.gamma_init
            x = np.copy(x_new)
        res = np.Inf
        for k in range(self.params.maxit):
            if self.callback(k, 0, gamma, self.x, res):
                break

            grad = self.problem.diffable.eval_gradient(x)
            res = np.linalg.norm(x - self.x)

            if res <= self.params.tol:
                break

            ell = np.dot(grad - self.grad, x - self.x) / res ** 2
            L = np.linalg.norm(grad - self.grad) / res

            rho = gamma / self.gamma
            alpha = np.sqrt(1 / self.params.pi + rho)
            delta = gamma ** 2 * L ** 2 - (2 - self.params.pi) * gamma * ell + 1 - self.params.pi

            if delta <= 0.:
                beta = np.Inf
            else:
                beta = 1 / np.sqrt(2 * delta)

            self.gamma = gamma

            gamma = gamma * np.minimum(alpha, beta)

            self.x[:] = x[:]
            self.grad[:] = grad[:]

            x = self.problem.proxable.eval_prox(self.x - gamma * self.grad, gamma)



class UniversalFastPGMLan(Optimizer):
    evals_per_iteration = 2
    evals_per_linesearch = 0

    def __init__(self, params, problem, callback=None):
        super().__init__(params, problem, callback)

        # Initialize x
        self.x = np.zeros(problem.x_init.shape)
        self.x[:] = problem.x_init[:]
        self.grad = self.problem.diffable.eval_gradient(self.x)

    def compute_lipschitz_estimate(self, x, x_old, grad_x, grad_x_old, tau):
        inner_approx = self.problem.diffable.eval(x_old) - \
                       self.problem.diffable.eval(x) - np.inner(grad_x, x_old - x)
        L = pow(np.linalg.norm(grad_x - grad_x_old), 2) / (2 * inner_approx + self.params.epsilon / tau)
        return L

    def run(self):
        x = np.copy(self.x)
        y = np.copy(x)
        #tau = 0
        beta = 1 - np.sqrt(3) / 2
        #tau_old = 0
        self.grad = self.problem.diffable.eval_gradient(self.x)
        grad = np.copy(self.grad)
        L = 0
        res = 0
        gamma = self.params.gamma_init

        # Initial iterates are computed outside the main loop to avoid multiple if statements
        #  k = 1  #
        self.callback(0, 0, gamma, x, res)
        x_new = np.copy(x)
        grad_new = np.copy(grad)
        for i in range(1, 20):  # Terminate the ill-defined line-search after 10 iterates!
            x_new = self.problem.proxable.eval_prox(x - gamma * grad, gamma)
            grad_new = self.problem.diffable.eval_gradient(x_new)
            L = (np.sqrt(pow(np.linalg.norm(x_new - x), 2)
                         * pow(np.linalg.norm(grad_new - self.grad), 2) + pow(self.params.epsilon / 4,
                                                                              2)) - self.params.epsilon / 4) \
                / pow(np.linalg.norm(x_new - x), 2)
            if beta / (4 * (1 - beta) * L) <= gamma <= 1 / (3 * L):
                break
            gamma = gamma * 0.5
        # print(i)
        self.grad[:] = grad[:]
        grad[:] = grad_new[:]
        self.x[:] = x[:]
        x[:] = x_new[:]

        #  k = 2  #
        res = np.linalg.norm(x - self.x, 2)
        self.callback(1, 0, gamma, x, res)
        gamma = beta / (2 * L)
        self.x[:] = x[:]
        z = self.problem.proxable.eval_prox(y - gamma * grad, gamma)
        y = (1 - beta) * y + beta * z
        x = (z + 2 * x) / 3
        self.grad = np.copy(grad)
        grad = self.problem.diffable.eval_gradient(x)
        L = self.compute_lipschitz_estimate(x, self.x, grad, self.grad, 2)

        tau_old = 0
        tau = 2
        #  Recursion starts after k = 3  #
        for k in range(3, self.params.maxit + 1):
            if self.callback(k - 1, 0, gamma, x, res):
                break

            # Store the old values of tau
            tau_prev = tau_old
            tau_old = tau

            # Compute tau and beta
            tau = tau_old + self.params.alpha / 2 + 2 * gamma * L * (1 - self.params.alpha) / (
                                  beta * tau_old)

            # Compute gamma:
            gamma = min(np.abs(beta * tau_old / (4 * L)), ((tau_prev + 1) / tau_old) * gamma)
            # Main part of the algorithm
            self.x[:] = x[:]
            z = self.problem.proxable.eval_prox(y - gamma * grad, gamma)
            y = (1 - beta) * y + beta * z
            x = (z + tau * x) / (1 + tau)
            self.grad[:] = grad[:]
            grad = self.problem.diffable.eval_gradient(x)
            res = np.linalg.norm(x - self.x, 2)

            # Compute estimates
            L = self.compute_lipschitz_estimate(x, self.x, grad, self.grad, tau)

            if res <= self.params.tol:
                break