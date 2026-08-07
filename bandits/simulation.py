import numpy as np
from tqdm import tqdm
import time
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from algorithms.utils import sigmoid


class Simulation(object):
    """
    Simulator of stochastic bandit interactions.
    """

    def __init__(self, bandit, W, policies, arm_num, d, T, tau):
        """
        Initialize the simulator.
        """
        self.bandit = bandit
        self.W = W
        self.policies = policies
        self.arm_num = arm_num
        self.d = d
        self.T = T
        self.theta_star = W.reshape(-1)
        self.tau = tau

        self.available_arms, self.feature_vector, self.wms, self.feasible_index = self.bandit.init_arms_set()

    def run_environment(self, T, n_mc, q, n_scatter):
        """
        Run Monte Carlo experiments.
        """
        t_saved = [i for i in range(T)]
        n_sub = np.size(t_saved)

        cum_regret = dict()
        avg_regret, q_regret, up_q_regret = dict(), dict(), dict()
        timedic = dict()

        cum_violation = dict()
        cum_rewards = dict()
        cum_penalized_reg = dict()

        for policy in self.policies:
            name = policy.__str__()
            cum_regret[name] = np.zeros((n_mc, n_sub))
            timedic[name] = np.zeros(n_sub)

            cum_violation[name] = np.zeros((n_mc, n_sub))
            cum_rewards[name] = np.zeros((n_mc, n_sub))
            cum_penalized_reg[name] = np.zeros((n_mc, n_sub))

        for nExp in tqdm(range(n_mc)):
            print('--------')
            print('Experiment number: ' + str(nExp))
            print('--------')

            for pol_idx, policy in enumerate(self.policies):
                time_init_pol = time.time()

                # Reinitialize policy and refresh the first round environment
                policy.re_init()
                self.available_arms, self.feature_vector, self.wms, self.feasible_index = self.bandit.init_arms_set()

                name = policy.__str__()
                optimal_rewards = np.zeros(T)
                rewards = np.zeros(T)

                inst_vio = np.zeros(T)
                penalized_regret = np.zeros(T)

                for t in range(T):
                    # time_init_pol = time.time()
                    # print("simulation_t: ", t)
                    
                    # print("###############################################")
                    # print("factual dataset:")
                    # for arm_idx, arm in enumerate(self.available_arms):
                    #     print(arm_idx, arm.features)

                    # Select arm
                    a_t = policy.select_arm(self.available_arms, self.feature_vector, self.wms)

                    # ------------------------------------------------------------
                    # True constraint violation for the selected arm
                    # ------------------------------------------------------------
                    ctf_de_vio = 0.0
                    ctf_ie_vio = 0.0
                    ctf_se_vio = 0.0

                    selected_feature = list(self.feature_vector[a_t])

                    for item in [0, 2, 4, 6]:
                        for mw_item in range(len(self.wms)):
                            if (selected_feature[item][1] == self.wms[mw_item][0]) and \
                               (selected_feature[item][2] == self.wms[mw_item][1]):

                                # IMPORTANT:
                                # Use the SAME feature map as the bandit so the
                                # violation computation is consistent with the model.
                                selected_fac_feature = self.bandit.feature_map(selected_feature[item])
                                selected_ctf_feature = self.bandit.feature_map(selected_feature[item + 1])

                                ctf_de_vio += (
                                    sigmoid(np.dot(selected_ctf_feature.T, self.theta_star))
                                    - sigmoid(np.dot(selected_fac_feature.T, self.theta_star))
                                ) * self.wms[mw_item][2] * self.wms[mw_item][4]

                                ctf_ie_vio += (
                                    sigmoid(np.dot(selected_ctf_feature.T, self.theta_star))
                                    * (self.wms[mw_item][4] - self.wms[mw_item][5])
                                    * self.wms[mw_item][2]
                                )

                                ctf_se_vio += (
                                    sigmoid(np.dot(selected_ctf_feature.T, self.theta_star))
                                    * self.wms[mw_item][5]
                                    * (self.wms[mw_item][2] - self.wms[mw_item][3])
                                )

                    penalize_flag = False
                    if (np.abs(ctf_de_vio) > self.tau) or \
                       (np.abs(ctf_ie_vio) > self.tau) or \
                       (np.abs(ctf_se_vio) > self.tau):
                        inst_vio[t] = max(
                            np.abs(ctf_de_vio),
                            np.abs(ctf_ie_vio),
                            np.abs(ctf_se_vio)
                        ) - self.tau
                        penalize_flag = True
                    else:
                        inst_vio[t] = 0.0

                    # ------------------------------------------------------------
                    # Benchmarks
                    # ------------------------------------------------------------
                    best_arm_index, instant_best_reward = self.bandit.get_best_arm()
                    best_arm_index_fea, instant_best_reward_fea = self.bandit.get_best_fair_arm(self.feasible_index)

                    # Play selected arm
                    round_reward, a_t_features, reward_index = self.bandit.play(a_t)
                    policy.update_state(a_t_features, round_reward, reward_index)

                    # print("self.feasible_index: ", self.feasible_index)
                    # print("best_arm_index: ", best_arm_index)
                    # print("best_arm_index_fea: ", best_arm_index_fea)

                    # ------------------------------------------------------------
                    # Record reward statistics
                    # ------------------------------------------------------------
                    expected_reward_round = self.bandit.get_expected_rewards()[a_t]
                    optimal_rewards[t] = instant_best_reward_fea
                    rewards[t] = expected_reward_round

                    if penalize_flag:
                        penalized_regret[t] = instant_best_reward_fea
                    else:
                        penalized_regret[t] = instant_best_reward_fea - expected_reward_round

                    time_end_pol = time.time() - time_init_pol
                    timedic[name][t] = time_end_pol

                    print("time:", t)
                    print("time_end_pol: ", time_end_pol)

                    if self.d == 2:
                        if t % n_scatter == 0:
                            h, m, s = seconds_to_hms(time_end_pol)
                            print(t, policy.__str__(), f"{h} hours, {m} minutes, and {s} seconds")

                    # Refresh the next round environment
                    self.available_arms, self.feature_vector, self.wms, self.feasible_index = self.bandit.init_arms_set()

                diff = optimal_rewards - rewards
                inst_regret = diff
                cum_regret[name][nExp, :] = np.cumsum(inst_regret)[t_saved]
                cum_rewards[name][nExp, :] = np.cumsum(rewards)[t_saved]
                cum_violation[name][nExp, :] = np.cumsum(inst_vio)[t_saved]
                cum_penalized_reg[name][nExp, :] = np.cumsum(penalized_regret)[t_saved]

        print("-- Building data out of the experiments ---")
        for policy in self.policies:
            name = policy.__str__()
            cum_reg = cum_regret[name]
            avg_regret[name] = np.mean(cum_reg, 0)
            q_regret[name] = np.percentile(cum_reg, q, 0)
            up_q_regret[name] = np.percentile(cum_reg, 100 - q, 0)

        print("--- Data built ---")
        return cum_regret, cum_violation, cum_rewards, cum_penalized_reg


def seconds_to_hms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return int(hours), int(minutes), int(seconds)


