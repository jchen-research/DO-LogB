from algorithms.OFU_MLogB import OFU_MLogB
from bandits.bandit import Bandit
from bandits.simulation import Simulation
from algorithms.utils import Myplot
import numpy as np
import os
import argparse


def Exp_Binary(policy, d, W, R, S, x_num, T, run_num, q, n_scat, tau, fair_N):
    # create the environments
    mab = Bandit(d, W, R, tau, fair_N, 'Mlog', num_actions=x_num)
    sim = Simulation(mab, W, policy, x_num, d, T, tau)

    # start the simulation
    cum_regret, cum_violation, cum_rewards, cum_penalized_reg = sim.run_environment(T, run_num, q, n_scat)

    output_dir = "..."
    os.makedirs(output_dir, exist_ok=True)

    np.savetxt(
        os.path.join(output_dir, f'cum_regret_{T}.txt'),
        cum_regret['OFU-MLogB'].mean(axis=0)
    )
    np.savetxt(
        os.path.join(output_dir, f'cum_regret_var_{T}.txt'),
        cum_regret['OFU-MLogB'].var(axis=0)
    )
    np.savetxt(
        os.path.join(output_dir, f'cum_violation_{T}.txt'),
        cum_violation['OFU-MLogB'].mean(axis=0)
    )
    np.savetxt(
        os.path.join(output_dir, f'cum_violation_var_{T}.txt'),
        cum_violation['OFU-MLogB'].var(axis=0)
    )
    np.savetxt(
        os.path.join(output_dir, f'cum_reward_{T}.txt'),
        cum_rewards['OFU-MLogB'].mean(axis=0)
    )
    np.savetxt(
        os.path.join(output_dir, f'cum_reward_var_{T}.txt'),
        cum_rewards['OFU-MLogB'].var(axis=0)
    )
    np.savetxt(
        os.path.join(output_dir, f'cum_penalized_regret_{T}.txt'),
        cum_penalized_reg['OFU-MLogB'].mean(axis=0)
    )
    np.savetxt(
        os.path.join(output_dir, f'cum_penalized_regret_var_{T}.txt'),
        cum_penalized_reg['OFU-MLogB'].var(axis=0)
    )  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--horizon",
        type=int,
        default=10001,
        help="Total number of steps T."
    )
    args = parser.parse_args()
    T = args.horizon
    S = 1
    W = np.array([[np.sqrt(S/2), 0, 0, np.sqrt(S/2)]])
    run_num = 1
    tau = 0.157
    fair_N = 1
    x_num = 10
    d = 4
    q = 5
    K = 1
    n_scat = 1000
    rho = np.arange(1, K + 1)
    R = 1.0
    L = 1.0
    delta = 1 / np.sqrt(T)

    policies = [OFU_MLogB(d, K, delta, S, L, rho, R, tau, W, T)]
    Exp_Binary(policies, d, W, R, S, x_num, T, run_num, q, n_scat, tau, fair_N)
    
