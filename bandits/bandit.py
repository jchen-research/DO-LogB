from bandits.arms import ArmGaussian, ArmMultiLogistic
import numpy as np
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from algorithms.utils import sigmoid
from bandits.synthetic_data import AWMGenerator


class Bandit:
    def __init__(self, d, W, R, tau, fair_N, model, num_actions=10, gamma2_target=0.01):
        self.W = W
        self.arms = []
        self.d = d
        self.R = R
        self.model = model
        self.tau = tau
        self.fair_N = fair_N
        self.theta_star = W.reshape(-1)

        self.num_actions = int(num_actions)

        # self.action_values = np.array(
        #     [-5.0, -4.0, -3.0, -2.0, -1.0,
        #       1.0,  2.0,  3.0,  4.0,  5.0],
        #     dtype=float
        # )
        # self.ctf_index_map = np.array([9, 8, 7, 4, 3, 6, 5, 2, 1, 0], dtype=int)
        # self.ctf_index_map = np.array([9, 8, 7, 4, 3, 6, 5, 2, 1, 0], dtype=int)

        self.action_values = np.array(
            [-1.0, -0.8, -0.6, -0.4, -0.2,
              0.2,  0.4,  0.6,  0.8,  1.0],
            dtype=float
        )
        self.ctf_index_map = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)

        self.gamma2_target = float(gamma2_target)

        self.gen = AWMGenerator(
            beta_w0=0.0,
            beta_wa=1.20,
            beta_m0=0.0,
            beta_ma=1.00,
            beta_mw=1.00,
        )

        self.gamma2_exact = self.compute_gamma2_exact()

    @staticmethod
    def feature_map(z):
        z = np.asarray(z, dtype=float)
        norm = np.linalg.norm(z)
        if norm == 0.0:
            return z
        return z / norm

    def sigma_d(self, d_value):
        pmf = self.gen.balanced_context_pmf()
        Sigma = np.zeros((self.d, self.d), dtype=float)

        for (a, w, m), p in pmf.items():
            f = self.feature_map(np.array([a, w, m, d_value], dtype=float)).reshape(-1, 1)
            Sigma += p * (f @ f.T)

        return Sigma

    def compute_gamma2_exact(self):
        min_eig = np.inf
        for d_val in self.action_values:
            Sigma = self.sigma_d(d_val)
            lam_min = float(np.min(np.linalg.eigvalsh(Sigma)))
            min_eig = min(min_eig, lam_min)
        return float(min_eig)

    def _build_feature_bundle(self, a_t, d_t, d_t_ctf):
        w0_m0 = self.feature_map(np.array([a_t,     0, 0, d_t], dtype=float))
        w0_m0_ctf = self.feature_map(np.array([1-a_t, 0, 0, d_t_ctf], dtype=float))

        w0_m1 = self.feature_map(np.array([a_t,     0, 1, d_t], dtype=float))
        w0_m1_ctf = self.feature_map(np.array([1-a_t, 0, 1, d_t_ctf], dtype=float))

        w1_m0 = self.feature_map(np.array([a_t,     1, 0, d_t], dtype=float))
        w1_m0_ctf = self.feature_map(np.array([1-a_t, 1, 0, d_t_ctf], dtype=float))

        w1_m1 = self.feature_map(np.array([a_t,     1, 1, d_t], dtype=float))
        w1_m1_ctf = self.feature_map(np.array([1-a_t, 1, 1, d_t_ctf], dtype=float))

        return (
            w0_m0, w0_m0_ctf,
            w0_m1, w0_m1_ctf,
            w1_m0, w1_m0_ctf,
            w1_m1, w1_m1_ctf,
        )

    def _true_effects(self, feature_bundle, wms):
        ctf_de_true = 0.0
        ctf_ie_true = 0.0
        ctf_se_true = 0.0

        feature = list(feature_bundle)

        for item in [0, 2, 4, 6]:
            for row in wms:
                w_val, m_val, p_aw, p_a_alt_w, p_m_aw, p_m_a_alt_w = row

                if (feature[item][1] == w_val) and (feature[item][2] == m_val):
                    fac_feature = self.feature_map(feature[item])
                    ctf_feature = self.feature_map(feature[item + 1])

                    sig_fac = sigmoid(np.dot(fac_feature.T, self.theta_star))
                    sig_ctf = sigmoid(np.dot(ctf_feature.T, self.theta_star))

                    ctf_de_true += (sig_ctf - sig_fac) * p_aw * p_m_aw
                    ctf_ie_true += sig_ctf * (p_m_aw - p_m_a_alt_w) * p_aw
                    ctf_se_true += sig_ctf * p_m_a_alt_w * (p_aw - p_a_alt_w)

        return float(ctf_de_true), float(ctf_ie_true), float(ctf_se_true)

    def init_arms_set(self, max_tries=5000, enforce_fair_pool=True):
        for _ in range(max_tries):
            self.arms = []
            self.feature_vector = []
            feasible_index = []

            a_t, w_t, m_t, wms = self.gen.sample_context_balanced()

            # print("a_t, w_t, m_t:", a_t, w_t, m_t)
            # print("wms: ", wms)

            for j in range(self.num_actions):
                d_t = float(self.action_values[j])
                d_t_ctf = float(self.action_values[self.ctf_index_map[j]])

                raw_arm = np.array([a_t, w_t, m_t, d_t], dtype=float)
                new_arm = self.feature_map(raw_arm)

                if self.model == 'linear':
                    self.arms.append(ArmGaussian(new_arm))
                elif self.model == 'Mlog':
                    self.arms.append(ArmMultiLogistic(new_arm))
                else:
                    raise ValueError(f"Unknown model: {self.model}")

                bundle = self._build_feature_bundle(a_t, d_t, d_t_ctf)
                self.feature_vector.append(bundle)

                ctf_de_true, ctf_ie_true, ctf_se_true = self._true_effects(bundle, wms)
                max_true = np.max([abs(ctf_de_true), abs(ctf_ie_true), abs(ctf_se_true)])

                # print(
                #     f"arm {j:2d}: d={d_t:.2f}, d_ctf={d_t_ctf:.2f}, "
                #     f"max_true_effect={max_true:.6f}"
                # )

                if (
                    abs(ctf_de_true) < self.tau
                    and abs(ctf_ie_true) < self.tau
                    and abs(ctf_se_true) < self.tau
                ):
                    feasible_index.append(j)

            if (not enforce_fair_pool) or (len(feasible_index) >= self.fair_N):
                return self.arms, self.feature_vector, wms, feasible_index

        raise RuntimeError(
            "Could not generate a round with enough fair arms. "
            "Try increasing max_tries, loosening tau, or set enforce_fair_pool=False."
        )

    def play(self, choice):
        assert isinstance(choice, (int, np.integer)), 'Choice type should be int!'
        reward, reward_index = self.arms[choice].pull(self.W, self.R)
        action_played = self.arms[choice].features
        return reward, action_played, reward_index

    def get_expected_rewards(self):
        return np.array([arm.get_expected_reward(self.W) for arm in self.arms], dtype=float)

    def get_best_arm(self):
        current_rewards = self.get_expected_rewards()
        assert len(current_rewards) > 0, "Error: No action generated, current_rewards is empty"
        best_arm = np.argmax(current_rewards)
        assert isinstance(best_arm, np.integer), "Error: bestArm type should be int"
        best_reward = current_rewards[best_arm]
        return best_arm, best_reward

    def get_best_fair_arm(self, feasible_idx=None):
        current_rewards = self.get_expected_rewards()

        if feasible_idx is None:
            assert len(current_rewards) > 0, "Error: current_rewards is empty"
            best_local = int(np.argmax(current_rewards))
            return best_local, float(current_rewards[best_local])

        feasible_idx = np.asarray(feasible_idx, dtype=int)
        assert feasible_idx.size > 0, "Error: feasible set is empty"

        feasible_rewards = current_rewards[feasible_idx]
        best_in_feasible = int(np.argmax(feasible_rewards))

        best_arm = int(feasible_idx[best_in_feasible])
        best_reward = float(current_rewards[best_arm])

        return best_arm, best_reward
