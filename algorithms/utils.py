import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy.linalg import LinAlgError, pinv
from scipy.linalg import sqrtm
from scipy.optimize import minimize_scalar
from matplotlib.ticker import MultipleLocator
'''color and style for lb'''
binary_styles = ['^', 'd',  'o' ,'X']
binary_colors = [sns.xkcd_rgb["dark sky blue"],  sns.xkcd_rgb["red"], sns.xkcd_rgb["leaf green"], sns.xkcd_rgb["dusty orange"]]
'''color and style for glb'''
multi_styles = ['H', 'X',  'o','^','d','p','s']
multi_colors = [sns.xkcd_rgb["nice blue"],sns.xkcd_rgb["dusty orange"],sns.xkcd_rgb["red"],sns.xkcd_rgb["leaf green"],sns.xkcd_rgb["black"], sns.xkcd_rgb["dark yellow"],sns.xkcd_rgb["purple"]] 

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11

class Myplot(object):
    """
    Simulator of stochastic games.
    param:
        - data: input data.
        - T: iterations
        - run_num: repeart time for a single experiments
        - q: quntile for plotting shadow bar
        - envir: binary / multi
    """
    
    def __init__(self, data, T, run_num, q, envir):
        """"
        global init function, initializing all the internal parameters that are fixed
        """        
        if envir == 'binary':
            self.styles = binary_styles
            self.colors = binary_colors
        else:
            self.styles = multi_styles
            self.colors = multi_colors        
        self.T = T
        self.data = data
        self.envir = envir
        self.q = q
        self.run_num = run_num
        
    def plot_regret(self,t_num =None):
        plt.figure()
        i = 0
        len_tsaved = self.T
        t_saved = [i for i in range(len_tsaved)]
    
        for key, _, avgRegret, qRegret, QRegret, *_ in self.data:
            label = key
            plt.plot(t_saved[0:t_num], avgRegret[0:t_num], marker=self.styles[i],
                     markevery=0.1, ms=10.0, label=label, color=self.colors[i])
    
            plt.fill_between(t_saved[0:t_num], qRegret[0:t_num], QRegret[0:t_num], alpha=0.15,
                             linewidth=1.5, color=self.colors[i])
            i += 1
        plt.legend(loc=2, fontsize=13).draw_frame(True)
        plt.xlabel('Iteration',fontsize=23)
        plt.ylabel('Regret',fontsize=23)
        plt.xticks(fontsize=13)  # Set the fontsize for x-ticks
        plt.yticks(fontsize=13)  # Set the fontsize for x-ticks
        plt.show()
        
def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

def weighted_norm(x, A):
    return np.sqrt(np.dot(x.T, np.dot(A, x)))

def fit_online_logistic_estimate(arm, reward, current_estimate, vtilde_matrix, vtilde_inv_matrix, constraint_set_radius,
                                 diameter=1, precision=0.1, eta = 1 / (1/4 + 1/(1 + 1/2)) * 0.75):
    """
    ECOLog estimation procedure.
    """
    # some pre-computation
    sqrt_vtilde_matrix = sqrtm(vtilde_matrix)
    sqrt_vtilde_inv_matrix = sqrtm(vtilde_inv_matrix)
    z_theta_t = np.dot(sqrt_vtilde_matrix, current_estimate)
    z_estimate = z_theta_t
    inv_z_arm = np.dot(sqrt_vtilde_inv_matrix, arm)
    step_size = eta # slightly smaller step for stability
    iters = int((1/0.75) * np.ceil((9 / 4 + diameter / 8) * np.log(diameter / precision)))
    # print(iters)


    # few steps of projected gradient descent
    for _ in range(iters):
        pred_probas = sigmoid(np.sum(z_estimate * inv_z_arm))
        grad = z_estimate - z_theta_t + (pred_probas - reward) * inv_z_arm
        unprojected_update = z_estimate - step_size * grad
        z_estimate = project_ellipsoid(x_to_proj=unprojected_update,
                                       ell_center=np.zeros_like(arm),
                                       ecc_matrix=vtilde_matrix,
                                       radius=constraint_set_radius)
    theta_estimate = np.dot(sqrt_vtilde_inv_matrix, z_estimate)
    return theta_estimate


def fit_online_logistic_estimate_bar(arm, current_estimate, vtilde_matrix, vtilde_inv_matrix, constraint_set_radius,
                                     diameter=1, precision=0.1):
    """
    ECOLog estimation procedure to compute theta_bar.
    """
    # some pre-computation
    sqrt_vtilde_matrix = sqrtm(vtilde_matrix)
    sqrt_vtilde_inv_matrix = sqrtm(vtilde_inv_matrix)
    z_theta_t = np.dot(sqrt_vtilde_matrix, current_estimate)
    z_estimate = z_theta_t
    inv_z_arm = np.dot(sqrt_vtilde_inv_matrix, arm)
    step_size = 1 / (1 / 4 + 1 / (1 + diameter / 2)) * 0.75  # slightly smaller step for stability
    iters = int((1/0.75) * np.ceil((9 / 4 + diameter / 8) * np.log(diameter / precision)))


    #few steps of projected gradient descent
    for _ in range(iters):
        pred_probas = sigmoid(np.sum(z_estimate * inv_z_arm))
        grad = z_estimate - z_theta_t + (2*pred_probas - 1) * inv_z_arm
        unprojected_update = z_estimate - step_size * grad
        z_estimate = project_ellipsoid(x_to_proj=unprojected_update,
                                       ell_center=np.zeros_like(arm),
                                       ecc_matrix=vtilde_matrix,
                                       radius=constraint_set_radius)
    theta_estimate = np.dot(sqrt_vtilde_inv_matrix, z_estimate)
    return theta_estimate

def project_ellipsoid(x_to_proj, ell_center, ecc_matrix, radius, safety_check=False):
    """
    Orthogonal projection on ellipsoidal set
    :param x_to_proj: np.array(dim), point to project
    :param ell_center: np.array(dim), center of ellipsoid
    :param ecc_matrix: np.array(dimxdim), eccentricity matrix
    :param radius: float, ellipsoid radius
    :param safety_check: bool, check ecc_matrix psd
    """
    # start by checking if the point to project is already inside the ellipsoid
    ell_dist_to_center = np.dot((x_to_proj - ell_center).T, np.linalg.solve(ecc_matrix, x_to_proj - ell_center))
    is_inside = (ell_dist_to_center - radius ** 2) < 1e-3
    if is_inside:
        return x_to_proj


    # check eccentricity is symmetric PSD
    if safety_check:
        sym_check = np.allclose(ecc_matrix, ecc_matrix.T)
        psd_check = np.all(np.linalg.eigvals(ecc_matrix) > 0)
        if not sym_check or not psd_check:
            raise ValueError("Eccentricity matrix is not symetric or PSD")

    # some pre-computation
    dim = len(x_to_proj)
    sqrt_psd_matrix = sqrtm(ecc_matrix)
    y = np.dot(sqrt_psd_matrix, x_to_proj - ell_center)

    # opt function for projection
    def fun_proj(lbda):
        try:
            solve = np.linalg.solve(ecc_matrix + lbda * np.eye(dim), y)
            res = lbda * radius ** 2 + np.dot(y.T, solve)
        except LinAlgError:
            res = np.inf
        return res

    # find proj
    lbda_opt = minimize_scalar(fun_proj, method='bounded', bounds=(0, 1000), options={'maxiter': 500})
    eta_opt = np.linalg.solve(ecc_matrix + lbda_opt.x * np.eye(dim), y)
    x_projected = np.dot(sqrt_psd_matrix, eta_opt) + ell_center

    return x_projected

def moving_average(sequence, window_size):
    smoothed = []
    half_window = window_size // 2
    for i in range(len(sequence)):
        # Define start and end indices for the sliding window
        start_idx = max(0, i - half_window)
        end_idx = min(len(sequence), i + half_window + 1)
        smoothed.append(sum(sequence[start_idx:end_idx]) / (end_idx - start_idx))
    return np.array(smoothed)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # subtracting np.max(x) for numerical stability
    return exp_x / exp_x.sum(axis=0)

def logistic_loss_cal(feature_dict, label_dict, estimator_dict,q,window_size):
    feature_shape = feature_dict.shape
    est_shape = estimator_dict.shape
    pred_shape = int(est_shape[2]/feature_shape[2])
    loss_dict = np.zeros((feature_shape[0],feature_shape[1],1))
    for nExp in range(feature_shape[0]):
        for t in np.arange(feature_shape[1]):
            if t == 0:
                estimator_prev = np.zeros(np.shape(estimator_dict[nExp,0,:]))
            else:
                estimator_prev = estimator_dict[nExp,t,:]
            z_t =  np.insert(np.dot(estimator_prev.reshape(pred_shape,feature_shape[2]),\
                                       feature_dict[nExp,t,:]),0,0)
            loss_t = np.dot(np.log(1 / softmax(z_t)),label_dict[nExp,t,:])    
            loss_dict[nExp,t,:] = loss_t
        loss_dict[nExp,:] = moving_average(loss_dict[nExp,:], window_size) 
    avgLoss = np.mean(loss_dict, 0)
    qLoss = np.percentile(loss_dict, q, 0)
    QLoss = np.percentile(loss_dict, 100 - q, 0)
    return avgLoss.reshape(-1), qLoss.reshape(-1), QLoss.reshape(-1)    

def estimated_reward_cal(arm_set,estimator_dict,rho):
    feature_shape = arm_set[0][0].features.shape[0]
    est_shape = estimator_dict.shape
    pred_shape = int(est_shape[2]/feature_shape)
    r_hat_dict = np.zeros((est_shape[0],est_shape[1],len(arm_set[0])))
    for nExp in range(est_shape[0]):
        for t in np.arange(est_shape[1]):
            for i in np.arange(len(arm_set[0])):
                x_feature = arm_set[0][i].features
                if t == 0:
                    estimator_prev = np.zeros(np.shape(estimator_dict[nExp,0,:]))
                else:
                    estimator_prev = estimator_dict[nExp,t,:]
                z_t =  np.insert(np.dot(estimator_prev.reshape(pred_shape,feature_shape),\
                                           x_feature),0,0)
                r_hat_t = np.dot(rho, softmax(z_t))  
                r_hat_dict[nExp,t,i] = r_hat_t
    return r_hat_dict 
