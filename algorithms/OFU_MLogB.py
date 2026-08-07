import numpy as np
from numpy.linalg import pinv
from algorithms.utils import sigmoid
from scipy.optimize import minimize, NonlinearConstraint


class OFU_MLogB(object):
    def __init__(self, d, K, delta, S, L, rho, R, tau, W, T, clip=1):
        """
        OFU-MLogB with normalized feature map f(z) = z / ||z||_2.

        The confidence radius is specialized to Theorem 1 in
        'Generalized Linear Bandits: Almost Optimal Regret with One-Pass Update'
        with:
            L_mu = 1/4
            g(tau) = 1
        """
        # basic attributes
        self.d = d
        self.delta = delta
        self.S = S
        self.K = K
        self.L = L
        self.rho = rho
        self.R = R
        self.dim = self.K * self.d
        self.clip = clip
        self.theta_star = np.asarray(W, dtype=float).reshape(-1)

        # theorem specialization
        self.L_mu = 0.25
        self.g_tau = 1.0

        # theorem parameters
        self.eta = 1.0 + self.R * self.S
        self.r_lambda = max(
            self.d * self.eta * (self.R ** 2),
            self.eta * self.R * self.S * self.L_mu / self.g_tau,
        )

        self.tau = tau
        self.T = T
        self.T0 = int(np.floor(np.sqrt(T)))
        # self.T0 = int(np.floor((4 * T) ** (2.0 / 3.0)))
        # self.T0 = int(np.floor((4 * T) ** (2.0 / 3.0)))

        # retained from your original code
        self.kappa = np.exp(S) * (1 + self.K * np.exp(S)) ** 2
        self.alpha = self.S + np.log(self.K + 1) / 2

        # state
        self.t = 1
        self.W_t = np.zeros((self.K * self.d, 1), dtype=float)

        # H_1 = lambda I
        self.H = self.r_lambda * np.identity(self.K * self.d)
        self.inv_H = (1.0 / self.r_lambda) * np.identity(self.K * self.d)

        self.Vtld = self.r_lambda * np.identity(self.d)
        self.inv_Vtld = (1.0 / self.r_lambda) * np.identity(self.d)

        self.sum = 0
        self.arms = []
        self.rewards = []
        self.vt = 0

        self.H_tm1 = self.r_lambda * np.identity(self.K * self.d)
        self.W_tm1 = np.zeros((self.K * self.d, 1), dtype=float)
        self.negative_term = 0.0

    # ------------------------------------------------------------------
    # normalized feature map: f(z) = z / ||z||_2
    # ------------------------------------------------------------------
    def feature_map(self, z):
        z = np.asarray(z, dtype=float).reshape(-1)
        norm = np.linalg.norm(z)
        if norm == 0.0:
            return z
        return z / norm

    def feature_col(self, z):
        return self.feature_map(z).reshape(-1, 1)

    @staticmethod
    def _to_float(x):
        return float(np.asarray(x).reshape(-1)[0])

    # ------------------------------------------------------------------
    # theorem-level confidence radius
    # ------------------------------------------------------------------
    def get_beta_t(self):
        beta_sq = (
            self.r_lambda * (self.S ** 2)
            + self.eta * np.log(1.0 / self.delta)
            + self.d * (self.eta ** 2 + self.eta)
            * np.log(1.0 + (self.L_mu * self.t) / (self.r_lambda * self.g_tau))
        )
        beta_sq = max(beta_sq, 0.0)
        return float(np.sqrt(beta_sq))

    # ------------------------------------------------------------------
    # arm selection
    # ------------------------------------------------------------------
    def select_arm(self, arms, feature_vector, wms):
        assert isinstance(arms, list), "List of arms as input required"

        # ---- Phase 1: pure exploration (t <= T0) ----
        # sample a decision uniformly at random from the decision set D
        if self.t <= self.T0:
            return int(np.random.randint(len(arms)))

        # ---- Phase 2: optimistic UCB over the estimated feasible set ----
        x_num = len(arms)
        ucb_s = np.full(x_num, -np.inf, dtype=float)
        ucb_s_cons = np.full(x_num, -np.inf, dtype=float)
        xi_t = np.zeros(x_num, dtype=float)

        beta_t = self.get_beta_t()

        for i, (x, feature) in enumerate(zip(arms, feature_vector)):
            # normalize the factual played feature
            x = self.feature_col(x.features)

            if self.K == 1:
                x_local_sq = self._to_float(x.T @ self.inv_H @ x)
                x_local_sq = max(x_local_sq, 0.0)
                x_local = float(np.sqrt(x_local_sq))
                reward_ucb = self._to_float(sigmoid(x.T @ self.W_t)) + beta_t * x_local
            else:
                reward_ucb = 0.0

            ctf_de = 0.0
            ctf_ie = 0.0
            ctf_se = 0.0

            fac_P_1 = 0.0
            ctf_P_1 = 0.0
            ctf_P_2 = 0.0
            ctf_P_3 = 0.0

            feature = list(feature)

            for item in [0, 2, 4, 6]:
                for mw_item in range(len(wms)):
                    if (feature[item][1] == wms[mw_item][0]) and (feature[item][2] == wms[mw_item][1]):
                        # normalize fairness features through the same feature map
                        fac_feature = self.feature_col(feature[item])
                        ctf_feature = self.feature_col(feature[item + 1])

                        p_aw = wms[mw_item][2]
                        p_a0w = wms[mw_item][3]
                        p_m_aw = wms[mw_item][4]
                        p_m_a0w = wms[mw_item][5]

                        sig_fac = self._to_float(sigmoid(fac_feature.T @ self.W_t))
                        sig_ctf = self._to_float(sigmoid(ctf_feature.T @ self.W_t))

                        ctf_de += (sig_ctf - sig_fac) * p_aw * p_m_aw
                        ctf_ie += sig_ctf * (p_m_aw - p_m_a0w) * p_aw
                        ctf_se += sig_ctf * p_m_a0w * (p_aw - p_a0w)

                        fac_width_sq = self._to_float(fac_feature.T @ self.inv_H @ fac_feature)
                        ctf_width_sq = self._to_float(ctf_feature.T @ self.inv_H @ ctf_feature)

                        fac_width = float(np.sqrt(max(fac_width_sq, 0.0)))
                        ctf_width = float(np.sqrt(max(ctf_width_sq, 0.0)))

                        fac_P_1 += fac_width * p_aw * p_m_aw
                        ctf_P_1 += ctf_width * p_aw * p_m_aw
                        ctf_P_2 += ctf_width * p_aw * p_m_a0w
                        ctf_P_3 += ctf_width * p_a0w * p_m_a0w

            xi_de = 0.25 * beta_t * (fac_P_1 + ctf_P_1)
            xi_ie = 0.25 * beta_t * (ctf_P_1 + ctf_P_2)
            xi_se = 0.25 * beta_t * (ctf_P_2 + ctf_P_3)

            xi_t[i] = max(xi_de, xi_ie, xi_se)
            ucb_s[i] = reward_ucb

            # optimistic feasibility test
            if (abs(ctf_de) - xi_de < self.tau) and \
               (abs(ctf_ie) - xi_ie < self.tau) and \
               (abs(ctf_se) - xi_se < self.tau):
                ucb_s_cons[i] = reward_ucb

        if np.all(np.isneginf(ucb_s_cons)):
            mixer = np.random.random(ucb_s.size)
            chosen_arm = list(np.lexsort((mixer, ucb_s)))[::-1][0]
        else:
            mixer = np.random.random(ucb_s_cons.size)
            chosen_arm = list(np.lexsort((mixer, ucb_s_cons)))[::-1][0]

        # print("beta_t:", beta_t)
        # print("ucb_s_cons:", ucb_s_cons)
        # print("xi_t:", xi_t)
        # print("W_t:", self.W_t.ravel())
        # print("chosen_arm:", chosen_arm)

        return chosen_arm

    # ------------------------------------------------------------------
    # state update
    # ------------------------------------------------------------------
    def update_state(self, x, y, y_index):
        x = self.feature_map(x).reshape(-1)
        assert isinstance(x, np.ndarray), "np.array required"

        self.arms.append(x)
        self.rewards.append(y_index)

        W_old = self.W_t.copy()
        self.estimator(x, y_index[1:])

        sigma = self.Sigma(self.W_t, x)
        gg_Wt = np.kron(np.diag(sigma) - np.outer(sigma, sigma.T), np.outer(x, x.T))

        self.H_tm1 = self.H.copy()
        self.H += gg_Wt
        self.inv_H = pinv(self.H)

        aat = np.outer(x, x.T)
        self.Vtld = self.Vtld + aat / self.kappa
        self.inv_Vtld = np.linalg.inv(self.Vtld)
        self.vt = np.kron(np.eye(self.K), aat)

        self.t += 1
        self.W_tm1 = W_old

    def re_init(self):
        self.t = 1
        self.W_t = np.zeros((self.K * self.d, 1), dtype=float)
        self.H = self.r_lambda * np.identity(self.K * self.d)
        self.inv_H = (1.0 / self.r_lambda) * np.identity(self.K * self.d)
        self.Vtld = self.r_lambda * np.identity(self.d)
        self.inv_Vtld = (1.0 / self.r_lambda) * np.identity(self.d)
        self.sum = 0
        self.arms = []
        self.rewards = []
        self.vt = 0
        self.H_tm1 = self.r_lambda * np.identity(self.K * self.d)
        self.W_tm1 = np.zeros((self.K * self.d, 1), dtype=float)
        self.negative_term = 0.0

    # ------------------------------------------------------------------
    # parameter estimator
    # ------------------------------------------------------------------
    def estimator(self, arm, reward):
        arm = self.feature_map(arm).reshape(-1)

        W_estimate = self.W_t
        sigma = self.Sigma(self.W_t, arm)
        g_Wt = np.kron((sigma - reward), arm)[:, None]
        gg_Wt = np.kron(np.diag(sigma) - np.outer(sigma, sigma.T), np.outer(arm, arm.T))

        M_t = (1.0 / (2.0 * self.eta)) * self.H + 0.5 * gg_Wt
        inv_Mt = pinv(M_t)
        unprojected_update = W_estimate - np.dot(inv_Mt, g_Wt)

        if np.linalg.norm(unprojected_update) > self.S:
            if self.K == 1:
                W_estimate = self.S * unprojected_update / np.linalg.norm(unprojected_update)
            else:
                W_estimate = self.projection(unprojected_update, M_t)[:, None]
            self.W_t = W_estimate
        else:
            self.W_t = unprojected_update

    # ------------------------------------------------------------------
    # multinomial logistic helpers
    # ------------------------------------------------------------------
    def Sigma(self, W, arm):
        arm = self.feature_map(arm).reshape(-1)
        W = W.reshape(self.K, self.d)
        z = np.matmul(W, arm)
        sigma = np.exp(z)
        sigma = sigma / (sigma.sum(axis=0) + 1.0)
        return sigma

    def nabla_Sigma(self, W, arm):
        arm = self.feature_map(arm).reshape(-1)
        W = W.reshape(self.K, self.d)
        z = np.matmul(W, arm)
        sigma = np.exp(z)
        sigma = sigma / (sigma.sum(axis=0) + 1.0)
        nabla_sigma = np.diag(sigma.reshape(-1)) - np.outer(sigma, sigma)
        return nabla_sigma

    # ------------------------------------------------------------------
    # projection
    # ------------------------------------------------------------------
    def proj_fun(self, W, un_projected, M):
        diff = W - un_projected.reshape(self.dim,)
        fun = np.dot(diff, np.dot(M, diff))
        return fun

    def projection(self, unprojected, M):
        fun = lambda t: self.proj_fun(t, unprojected, M)
        constraints = []
        for k in range(self.K):
            norm = lambda t, k=k: np.linalg.norm(t[self.d * k:self.d * k + self.d])
            constraint = NonlinearConstraint(norm, 0, self.S)
            constraints.append(constraint)
        opt = minimize(fun, x0=np.zeros(self.K * self.d), method='SLSQP', constraints=constraints)
        return opt.x

    def __str__(self):
        return "OFU-MLogB"


