"""
Arm class
"""

import numpy as np
from algorithms.utils import sigmoid

class Arm(object):
    def pull(self, theta, sigma_noise):
        print('pulling from the parent class')
        pass

    def get_expected_reward(self, theta):
        print('Receiving reward from the parent class')

class ArmGaussian(Arm):
    """
    Arm vector with gaussian noise
    """

    def __init__(self, vector):
        """
        Constructor
        """
        assert isinstance(vector, np.ndarray), 'np.array required'
        self.features = vector / np.linalg.norm(vector)  # action for the arm, numpy-array
        self.d = vector.shape[0]
        

    def get_expected_reward(self, theta):
        """
        Return dot(A_t,theta)
        """
        assert isinstance(theta, np.ndarray), 'np.array required for the theta vector'
        return np.dot(self.features, theta)

    def pull(self, theta, sigma_noise):
        """
        We are in the stochastic setting.
        The reward is sampled according to Normal(dot(A_t,theta),sigma_noise**2)
        """
        return np.random.normal(self.get_expected_reward(theta), sigma_noise)


class ArmMultiLogistic(Arm):
    """
    Arm vector with MLog
    """

    def __init__(self, vector):
        """
        Constructor
        """
        assert isinstance(vector, np.ndarray), 'np.array required'
        self.features = vector / np.linalg.norm(vector)  # action for the arm, numpy-array
        self.dim = vector.shape[0]

    def get_expected_reward(self, W):
        """
        Return dot(A_t,theta)
        """
        assert isinstance(W, np.ndarray), 'np.array required for the theta vector'
        sigma = self.get_prob(W)
        rho = np.arange(0, W.shape[0] + 1)
        rho = rho/max(rho)
        return np.dot(rho, sigma)

    def pull(self, W, R):
        """
        Mlog
        """
        sigma = self.get_prob(W)
        rho = np.arange(0, W.shape[0] + 1)
        rho = rho / max(rho)
        y = np.random.multinomial(1, sigma)
        return np.dot(rho, y), y

    def get_prob(self, W):
        z = np.matmul(W, self.features)
        s = 0
        for k in z:
            s += np.exp(k)
        sigma = np.array([1 / (1 + s)])
        for k in z:
            sigma = np.append(sigma, np.exp(k) / (1 + s))
        return sigma
