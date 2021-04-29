
from jax.config import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from jax import jit
from jax.ops import index, index_add, index_update
import sys; sys.path.append('../')
import jax_opt.optimizers
import jax_opt.manifolds
from jax import random

import numpy as np
# np.random.seed(0)
import os

import mps_func
# def init_mps(L, chi, d):
# def get_mps_amp(mps_list, config):
# def get_mps_amp_batch(mps_list, config_batch):


def mps_2_mat(mps_list):
    mps_mat_list = []
    for i in range(L):
        dim1, dim2, dim3 = mps_list[i].shape
        mps_mat_list.append(mps_list[i].reshape([dim1*dim2, dim3]))

    return mps_mat_list

def mat_2_mps(mps_mat_list):
    mps_list = []
    for mat in mps_mat_list:
        dim_1, dim_2 = mat.shape
        mps_list.append(mat.reshape([dim_1//2, 2, dim_2]))

    return mps_list


@jit
def cost_function_kl(mps_mat_list, batch_config, batch_amp):
    '''
    Input:
        mps_mat_list: list of mps tensor
        batch_config : the sampled input X = configuration
        batch_amp : the sampled target Y = prob. amplitude
    Output:
        cost
    '''
    mps_list = mat_2_mps(mps_mat_list)
    mps_amp_array = mps_func.get_mps_amp_batch(mps_list, batch_config)
    mps_log_amp = jnp.log(mps_amp_array)
    target_log_amp = jnp.log(batch_amp)
    kl_cost = 2 * jnp.mean(jnp.real(target_log_amp) - jnp.real(mps_log_amp))
    return kl_cost


@jit
def cost_function_l2(mps_mat_list, batch_config, batch_amp):
    '''
    Input:
        mps_mat_list: list of mps tensor
        batch_config : the sampled input X = configuration
        batch_amp : the sampled target Y = prob. amplitude
    Output:
        cost
    '''
    mps_list = mat_2_mps(mps_mat_list)
    mps_amp_array = mps_func.get_mps_amp_batch(mps_list, batch_config)
    mps_log_amp = jnp.log(mps_amp_array)
    target_log_amp = jnp.log(batch_amp)
    mps_phase = jnp.imag(mps_log_amp)
    target_phase = jnp.imag(target_log_amp)
    cost_phase = jnp.mean(jnp.square((jnp.cos(mps_phase) - jnp.cos(target_phase))) +\
                          jnp.square((jnp.sin(mps_phase) - jnp.sin(target_phase))) )
    return cost_phase


@jit
def cost_function_joint(mps_mat_list, batch_config, batch_amp):
    '''
    The cost function to optimize
    C = R_KL + R_L2

    Input:
        mps_mat_list: list of mps tensor
        batch_config : the sampled input X = configuration
        batch_amp : the sampled target Y = prob. amplitude
    Output:
        cost
    '''
    mps_list = mat_2_mps(mps_mat_list)

    mps_amp_array = mps_func.get_mps_amp_batch(mps_list, batch_config)

    mps_log_amp = jnp.log(mps_amp_array)
    target_log_amp = jnp.log(batch_amp)

    kl_cost = 2 * jnp.mean(jnp.real(target_log_amp) - jnp.real(mps_log_amp))

    mps_phase = jnp.imag(mps_log_amp)
    target_phase = jnp.imag(target_log_amp)
    cost_phase = jnp.mean(jnp.square((jnp.cos(mps_phase) - jnp.cos(target_phase))) +\
                          jnp.square((jnp.sin(mps_phase) - jnp.sin(target_phase))) )

    total_cost = kl_cost + cost_phase
    # print("kl=", kl_cost, "phase=", cost_phase, "total=", total_cost)
    return total_cost

@jit
def cost_function_overlap(mps_mat_list, batch_config, batch_amp):
    '''
    The cost function to optimize
    C = - Re [ overlap ]

    Input:
        mps_mat_list: list of mps tensor
        batch_config : the sampled input X = configuration
        batch_amp : the sampled target Y = prob. amplitude
    Output:
        cost
    '''

    mps_list = mat_2_mps(mps_mat_list)
    mps_amp_array = mps_func.get_mps_amp_batch(mps_list, batch_config)

    # return -jnp.mean(jnp.real(mps_amp_array / batch_amp))
    return -(jnp.abs(jnp.mean(mps_amp_array / batch_amp))**2 )


# def cost_function_fidelity(circuit, target_state, product_state):
#     '''
#     The cost function to optimize
#     C = 1. - F
#     F = | <target_state | circuit | product_state > |^2
# 
#     Input:
#         circuit: list of list of unitary
#         target_state: a quantum many-body state
#         product_state: a product state which circuit acts on.
#     Output:
#         cost
#     '''
# 
#     iter_state = circuit_func.circuit_2_state(circuit, product_state)
#     fidelity = np.abs(circuit_func.overlap_exact(target_state, iter_state))**2
#     # fidelity = np.real(circuit_func.overlap_exact(target_state, iter_state))
#     return 1. - fidelity


def cost_function_fidelity(mps_mat_list, exact_mps):
    mps_list = mat_2_mps(mps_mat_list)
    mps_list = mps_func.lpr_2_plr(mps_list)
    overlap = mps_func.overlap(mps_list, exact_mps)
    return 1. - jnp.square(jnp.abs(overlap))


def training_sgd(mps_mat_list, X, Y, opt_type,
                 num_iter=10000, lr=0.05, batch_size=512,
                 exact_mps=None, data_dict=None, T=None, ckpt_path=None
                ):
    '''
    Inputs:
        mps_mat_list: list of mps tensor
        batch_config
        batch_amp
        opt_type
        num_iter: the number of iterations
        lr: the leraning rate
    '''
    # manifold = jax_opt.manifolds.StiefelManifold()
    manifold = jax_opt.manifolds.StiefelManifold(metric='euclidean', retraction='svd')
    if opt_type == 'rsgd':
        opt_init, opt_update, get_params = jax_opt.optimizers.rsgd(lr, manifold)
    elif opt_type == 'rmom':
        opt_init, opt_update, get_params = jax_opt.optimizers.rmomentum(lr, manifold, 0.95)
    elif opt_type == 'radam':
        opt_init, opt_update, get_params = jax_opt.optimizers.radam(lr, manifold)
    else:
        raise NotImplementedError


    cost_function = cost_function_overlap
    cost_function = cost_function_joint


    @jit
    def update(idx, opt_state, data):
        params = get_params(opt_state)
        gradient_direction = jax.tree_util.tree_map(jnp.conj,
                                                    jax.grad(cost_function)(params, *data)
                                                   )
        # gradient_direction = jax.grad(cost_function)(params, *data)
        return opt_update(idx,
                          gradient_direction,
                          opt_state)


    #################
    # set up params #
    #################

    params = jax.tree_util.tree_map(jnp.complex128,
                                    mps_mat_list)
    opt_state = opt_init(params)

    # data = [batch_config, batch_amp]
    cost_list = []
    cost_kl_list = []
    cost_l2_list = []
    # current_cost = cost_function(get_params(opt_state), *data)
    # print("begin: ", current_cost)
    # cost_list.append(current_cost)

    key = random.PRNGKey(0)

    print(X.shape, Y.shape)
    ED_prob = (np.abs(Y)**2).flatten()
    for step in range(1, num_iter+1):
        batch_mask = np.random.choice(len(Y), batch_size, p=ED_prob)
        # batch_mask = jax.random.choice(key, len(Y), [batch_size], p=ED_prob);
        # key, subkey = random.split(key)

        X_mini_batch = X[batch_mask]
        X_mini_batch = X_mini_batch[:, :, 0].astype(int)
        Y_mini_batch = Y[batch_mask].flatten()
        data = [X_mini_batch, Y_mini_batch]


        opt_state = update(step, opt_state, data)
        current_cost = cost_function(get_params(opt_state), *data)
        current_cost_kl = cost_function_kl(get_params(opt_state), *data)
        current_cost_l2 = cost_function_l2(get_params(opt_state), *data)
        # print("step : ", step, "cost : ", current_cost)
        cost_list.append(current_cost)
        cost_kl_list.append(current_cost_kl)
        cost_l2_list.append(current_cost_l2)


        if step % 500 == 0:
            data_dict['lr'].append(lr)
            data_dict['cost_avg'].append(np.average(cost_list))
            data_dict['cost_var'].append(np.var(cost_list))
            cost_list = []
            data_dict['kl_avg'].append(np.average(cost_kl_list))
            data_dict['kl_var'].append(np.var(cost_kl_list))
            cost_kl_list = []
            data_dict['l2_avg'].append(np.average(cost_l2_list))
            data_dict['l2_var'].append(np.var(cost_l2_list))
            cost_l2_list = []

            mps_mat_list = get_params(opt_state)
            mps_list = mat_2_mps(mps_mat_list)
            mps_list = mps_func.lpr_2_plr(mps_list)
            overlap = mps_func.overlap(mps_list, exact_mps)
            mps_list = mps_func.plr_2_lpr(mps_list)
            data_dict['fidelity'].append(np.abs(overlap)**2)

            print("step %d, Cost=%f, kl=%f, l2=%f, F=%f" % (step, data_dict['cost_avg'][-1], data_dict['kl_avg'][-1], data_dict['l2_avg'][-1], np.abs(overlap)**2 ))
            np.save(ckpt_path + '/chi%d_T%.2f.npy' % (chi, T), mps_mat_list)
            np.save(ckpt_path + '/data_dict.npy', data_dict, allow_pickle=True)

        if step % 10000 == 0:
            if (np.abs(overlap)**2 > 1 - 1e-4) or (data_dict['cost_avg'][-1] > np.average(data_dict['cost_avg'][-10:-5])):
            # if (np.abs(overlap)**2 > 1 - 1e-4) or (np.average(data_dict['cost_avg'][-10:]) > np.average(data_dict['cost_avg'][-20:-10])):
                break
            else:
                pass


    ## Either call this or call mps function
    mps_list = mps_func.lpr_2_plr(mps_list)
    Sx_list = [np.array([[0., 1.], [1., 0.]]) for i in range(L)]
    data_dict['Sx_list'] = np.real(mps_func.expectation_values_1_site(mps_list, Sx_list))
    data_dict['SvN_list'] = mps_func.get_entanglement(mps_list)
    data_dict['S2_list'] = mps_func.get_renyi_n_entanglement(mps_list, 2)
    mps_list = mps_func.lpr_2_plr(mps_list)

    # sx_expectation = many_body.sx_expectation(N_sys//2+1, y.flatten(), N_sys)
    # print("<Sx> = ", sx_expectation)
    # data_dict["Sx"] = sx_expectation
    # S2, SvN = many_body.entanglement_entropy(N_sys//2 + 1, y.flatten(), N_sys)
    # data_dict["renyi_2"] = S2
    # data_dict["SvN"] = SvN

    return get_params(opt_state), data_dict



def training_mps(mps_mat_list, exact_mps, opt_type,
                 num_iter=10000, lr=0.05, batch_size=512):
    '''
    Inputs:
        mps_mat_list: list of mps tensor
        batch_config
        batch_amp
        opt_type
        num_iter: the number of iterations
        lr: the leraning rate
    '''
    # manifold = jax_opt.manifolds.StiefelManifold()
    manifold = jax_opt.manifolds.StiefelManifold(metric='euclidean', retraction='svd')
    if opt_type == 'rsgd':
        opt_init, opt_update, get_params = jax_opt.optimizers.rsgd(lr, manifold)
    elif opt_type == 'rmom':
        opt_init, opt_update, get_params = jax_opt.optimizers.rmomentum(lr, manifold, 0.95)
    elif opt_type == 'radam':
        opt_init, opt_update, get_params = jax_opt.optimizers.radam(lr, manifold)
    else:
        raise NotImplementedError

    cost_function = cost_function_fidelity

    @jit
    def update(idx, opt_state, data):
        params = get_params(opt_state)
        gradient_direction = jax.tree_util.tree_map(jnp.conj,
                                                    jax.grad(cost_function)(params, data)
                                                   )
        # gradient_direction = jax.grad(cost_function)(params, *data)
        return opt_update(idx,
                          gradient_direction,
                          opt_state)


    #################
    # set up params #
    #################

    params = jax.tree_util.tree_map(jnp.complex128,
                                    mps_mat_list)
    opt_state = opt_init(params)
    cost_list = []

    for step in range(1, num_iter+1):
        opt_state = update(step, opt_state, exact_mps)
        current_cost = cost_function(get_params(opt_state), exact_mps)
        print("step : ", step, "cost : ", current_cost)
        cost_list.append(current_cost)

    return get_params(opt_state), cost_list













if __name__ == '__main__':
    L = 20
    batch_size = 512
    chi = 6
    supervised_model = '1D_ZZ_1.00X_0.25XX_global_TE_L20'
    T = 1.00
    num_iter = 5000


    #################
    # Prepare DATA  #
    ######################
    N_sys = L
    X_computation_basis = np.genfromtxt('ExactDiag/basis_L%d.csv' % L, delimiter=',')
    X = np.zeros([2**L, L, 2], dtype=int)
    X[:,:,0] = X_computation_basis.reshape([2**L, L]).astype(int)
    X[:,:,1] = 1-X_computation_basis.reshape([2**L, L]).astype(int)
    Y = np.load('ExactDiag/wavefunction/%s/ED_wf_T%.2f.npy' % (supervised_model, T))
    Y = np.array(Y, dtype=np.complex128)[:, None]

    path = 'Result/'
    ckpt_path = path + \
            'wavefunction/Supervised/' + '%s_T%.2f/' % (supervised_model, T) + \
            'MPS_%d' % chi

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)


    try:
        data_dict = np.load(ckpt_path + '/data_dict.npy', allow_pickle=True).item()
        print("found data_dict")
    except:
        print("no data_dict found; create new data_dict")
        data_dict = {'cost_avg': [], 'kl_avg': [], 'l2_avg': [], 'fidelity': [],
                     'cost_var': [], 'kl_var': [], 'l2_var': [],
                     'lr': []
                    }
    # ###################################################################


    ######################
    # Prepare MPS
    ######################
    try:
        mps_mat_list = np.load(ckpt_path + '/chi%d_T%.2f.npy' % (chi, T), allow_pickle=True)
        mps_mat_list = [mps for mps in mps_mat_list]
        print(" load success !!! \n\n\n")
    except Exception as e:
        print("error = ", e)
        mps_list = mps_func.init_mps(L=L, chi=chi, d=2)
        mps_mat_list = mps_2_mat(mps_list)

    exact_mps = mps_func.state_2_MPS(Y, 20, 1024)


    mps_list = mat_2_mps(mps_mat_list)
    mps_list = mps_func.lpr_2_plr(mps_list)
    overlap = mps_func.overlap(mps_list, exact_mps)
    exact_overlap = mps_func.overlap(exact_mps, exact_mps)
    var_overlap = mps_func.overlap(mps_list, mps_list)
    print("\n overlap = %.5f" % np.abs(overlap))
    print("exact mps norm = %.5f" % exact_overlap.real)
    print("var mps norm = %.5f\n" % var_overlap.real)
    mps_list = mps_func.plr_2_lpr(mps_list)


    # ###################################################################
    # ### get full batch information
    # y_list = []
    # end_idx=0
    # for i in range((2**N_sys) // 1024):
    #     start_idx, end_idx = i*1024, (i+1)*1024
    #     # yi = Net.get_amp(X[start_idx:end_idx])
    #     yi = mps_func.get_mps_amp_batch(mps_list, X[start_idx:end_idx,:,0].astype(int))
    #     y_list.append(yi)

    # if end_idx != 2**N_sys:
    #     yi = mps_func.get_mps_amp_batch(mps_list, X[end_idx:,:,0].astype(int))
    #     y_list.append(yi)

    # y = np.concatenate(y_list)
    # print(('y norm : ', np.linalg.norm(y)))
    # measured_fidelity = np.square(np.abs(Y.flatten().dot(y.flatten().conj())))
    # print(" explicit fidelity = ", measured_fidelity)
    # ###################################################################



    ###################################################################
    mps_mat_list, data_dict = training_sgd(mps_mat_list, X, Y,
                                           opt_type='radam', num_iter=300000,
                                           batch_size=batch_size, lr=1e-3,
                                           exact_mps=exact_mps,
                                           data_dict=data_dict, T=T, ckpt_path=ckpt_path
                                          )
    mps_mat_list, data_dict = training_sgd(mps_mat_list, X, Y,
                                           opt_type='radam', num_iter=300000,
                                           batch_size=batch_size, lr=1e-4,
                                           exact_mps=exact_mps,
                                           data_dict=data_dict, T=T, ckpt_path=ckpt_path
                                          )
    # mps_mat_list, cost_list = training_mps(mps_mat_list, exact_mps,
    #                                        opt_type='radam', num_iter=1000, lr=1e-2)
    np.save(ckpt_path + '/chi%d_T%.2f.npy' % (chi, T), mps_mat_list)
    np.save(ckpt_path + '/data_dict.npy', data_dict, allow_pickle=True)

    ###################################################################


