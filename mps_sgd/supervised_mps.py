import sys; sys.path.append('../../')
sys.path.append('../')
import tensor_network_functions.mps_func as mps_func

import numpy as np
from jax import random
import jax_opt.manifolds
import jax_opt.optimizers
from jax.ops import index, index_add, index_update
from jax import jit
import jax.numpy as jnp
import jax
from jax.config import config
config.update("jax_enable_x64", True)


def mps_2_mat(mps_list):
    '''
    [left, phys, right]
    '''
    L = len(mps_list)
    mps_mat_list = []
    for i in range(L):
        dim1, dim2, dim3 = mps_list[i].shape
        mps_mat_list.append(mps_list[i].reshape([dim1*dim2, dim3]))

    return mps_mat_list


def mat_2_mps(mps_mat_list):
    '''
    [left, phys, right]
    '''
    mps_list = []
    for mat in mps_mat_list:
        dim_1, dim_2 = mat.shape
        mps_list.append(mat.reshape([dim_1//2, 2, dim_2]))

    return mps_list


@jit
def cost_function_kl(mps_mat_list, batch_config, batch_amp):
    '''
    [left, phys, right]

    Input:
        mps_mat_list: list of mps tensor
        batch_config : the sampled input X = configuration
        batch_amp : the sampled target Y = prob. amplitude
    Output:
        cost
    '''
    mps_list = mat_2_mps(mps_mat_list)
    mps_amp_array = mps_func.get_mps_amp_batch_jax(mps_list, batch_config)
    mps_log_amp = jnp.log(mps_amp_array)
    target_log_amp = jnp.log(batch_amp)
    kl_cost = 2 * jnp.mean(jnp.real(target_log_amp) - jnp.real(mps_log_amp))
    return kl_cost


@jit
def cost_function_kl_unnormalized(mps_mat_list, batch_config, batch_amp):
    '''
    [left, phys, right]

    Compute the KL divergence between the target amplitude and the given mps
    wavefunction, which is not normalized.
    Input:
        mps_mat_list: list of mps tensor ( NOT necessarily in isometric form )
        batch_config : the sampled input X = configuration
        batch_amp : the sampled target Y = prob. amplitude
    Output:
        cost
    '''
    mps_list = mat_2_mps(mps_mat_list)
    mps_unnorm_amp_array = mps_func.get_mps_amp_batch_jax(mps_list, batch_config)
    mps_norm = jnp.sqrt(jnp.abs(mps_func.overlap_lpr(mps_list, mps_list)))
    mps_log_amp = jnp.log(mps_unnorm_amp_array / mps_norm)
    target_log_amp = jnp.log(batch_amp)
    kl_cost = 2 * jnp.mean(jnp.real(target_log_amp) - jnp.real(mps_log_amp))
    return kl_cost


@jit
def cost_function_l2(mps_mat_list, batch_config, batch_amp):
    '''
    [left, phys, right]

    Input:
        mps_mat_list: list of mps tensor
        batch_config : the sampled input X = configuration
        batch_amp : the sampled target Y = prob. amplitude
    Output:
        cost
    '''
    mps_list = mat_2_mps(mps_mat_list)
    mps_amp_array = mps_func.get_mps_amp_batch_jax(mps_list, batch_config)
    mps_log_amp = jnp.log(mps_amp_array)
    target_log_amp = jnp.log(batch_amp)
    mps_phase = jnp.imag(mps_log_amp)
    target_phase = jnp.imag(target_log_amp)
    cost_phase = jnp.mean(jnp.square((jnp.cos(mps_phase) - jnp.cos(target_phase))) +
                          jnp.square((jnp.sin(mps_phase) - jnp.sin(target_phase))))
    return cost_phase


@jit
def cost_function_joint(mps_mat_list, batch_config, batch_amp):
    '''
    [left, phys, right]

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

    mps_amp_array = mps_func.get_mps_amp_batch_jax(mps_list, batch_config)

    mps_log_amp = jnp.log(mps_amp_array)
    target_log_amp = jnp.log(batch_amp)

    kl_cost = 2 * jnp.mean(jnp.real(target_log_amp) - jnp.real(mps_log_amp))

    mps_phase = jnp.imag(mps_log_amp)
    target_phase = jnp.imag(target_log_amp)
    cost_phase = jnp.mean(jnp.square((jnp.cos(mps_phase) - jnp.cos(target_phase))) +
                          jnp.square((jnp.sin(mps_phase) - jnp.sin(target_phase))))

    total_cost = kl_cost + cost_phase
    # print("kl=", kl_cost, "phase=", cost_phase, "total=", total_cost)
    return total_cost


@jit
def cost_function_joint_unnormlized(mps_mat_list, batch_config, batch_amp):
    '''
    [left, phys, right]

    The cost function to optimize
    C = R_KL + R_L2

    Input:
        mps_mat_list: list of mps tensor ( NOT necessarily in isometric form )
        batch_config : the sampled input X = configuration
        batch_amp : the sampled target Y = prob. amplitude
    Output:
        cost
    '''
    mps_list = mat_2_mps(mps_mat_list)

    mps_unnorm_amp_array = mps_func.get_mps_amp_batch_jax(mps_list, batch_config)
    mps_norm = jnp.sqrt(jnp.abs(mps_func.overlap_lpr(mps_list, mps_list)))
    mps_log_amp = jnp.log(mps_unnorm_amp_array / mps_norm)

    target_log_amp = jnp.log(batch_amp)

    kl_cost = 2 * jnp.mean(jnp.real(target_log_amp) - jnp.real(mps_log_amp))

    mps_phase = jnp.imag(mps_log_amp)
    target_phase = jnp.imag(target_log_amp)
    cost_phase = jnp.mean(jnp.square((jnp.cos(mps_phase) - jnp.cos(target_phase))) +
                          jnp.square((jnp.sin(mps_phase) - jnp.sin(target_phase))))

    total_cost = kl_cost + cost_phase
    # print("kl=", kl_cost, "phase=", cost_phase, "total=", total_cost)
    return total_cost


@jit
def cost_function_overlap(mps_mat_list, batch_config, batch_amp):
    '''
    [left, phys, right]

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
    mps_amp_array = mps_func.get_mps_amp_batch_jax(mps_list, batch_config)

    # return -jnp.mean(jnp.real(mps_amp_array / batch_amp))
    return -(jnp.abs(jnp.mean(mps_amp_array / batch_amp))**2)


@jit
def cost_function_overlap_unnormalized(mps_mat_list, batch_config, batch_amp):
    '''
    [left, phys, right]

    The cost function to optimize
    C = - Re [ overlap ]

    Input:
        mps_mat_list: list of mps tensor ( NOT necessarily in isometric form )
        batch_config : the sampled input X = configuration
        batch_amp : the sampled target Y = prob. amplitude
    Output:
        cost
    '''

    mps_list = mat_2_mps(mps_mat_list)

    mps_unnorm_amp_array = mps_func.get_mps_amp_batch_jax(mps_list, batch_config)
    mps_norm = jnp.sqrt(jnp.abs(mps_func.overlap_lpr(mps_list, mps_list)))
    mps_amp_array = mps_unnorm_amp_array / mps_norm

    # return -jnp.mean(jnp.real(mps_amp_array / batch_amp))
    return -(jnp.abs(jnp.mean(mps_amp_array / batch_amp))**2)


@jit
def cost_function_fidelity(mps_mat_list, exact_mps):
    '''
    mps_mat_list: [left, phys, right]
    exact_mps: [phys, left, right]
    '''
    mps_list = mat_2_mps(mps_mat_list)
    mps_list = mps_func.lpr_2_plr(mps_list)
    overlap = mps_func.overlap(mps_list, exact_mps)
    return 1. - jnp.square(jnp.abs(overlap))


@jit
def cost_function_fidelity_unnormalized(mps_mat_list, exact_mps):
    '''
    mps_mat_list: [left, phys, right]
    exact_mps: [phys, left, right]
    '''
    mps_list = mat_2_mps(mps_mat_list)
    mps_list = mps_func.lpr_2_plr(mps_list)
    overlap = mps_func.overlap(mps_list, exact_mps)
    norm_square = mps_func.overlap_lpr(mps_list, mps_list)
    return 1. - overlap * jnp.conjugate(overlap) / norm_square


def training_r_sgd(mps_mat_list, X, Y, opt_type,
                  num_iter=10000, lr=0.05, batch_size=512,
                  exact_mps=None, data_dict=None, T=None, chi=None, ckpt_path=None
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
    manifold = jax_opt.manifolds.StiefelManifold(
        metric='euclidean', retraction='svd')
    if opt_type == 'r_sgd':
        opt_init, opt_update, get_params = jax_opt.optimizers.r_sgd(
            lr, manifold)
    elif opt_type == 'r_mom':
        opt_init, opt_update, get_params = jax_opt.optimizers.r_momentum(
            lr, manifold, 0.95)
    elif opt_type == 'r_adam':
        opt_init, opt_update, get_params = jax_opt.optimizers.r_adam(
            lr, manifold)
    else:
        raise NotImplementedError

    cost_function = cost_function_overlap
    cost_function = cost_function_joint

    @jit
    def update(idx, opt_state, data):
        params = get_params(opt_state)
        gradient_direction = jax.tree_util.tree_map(jnp.conj,
                                                    jax.grad(cost_function)(
                                                        params, *data)
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

    print(X.shape, Y.shape)
    ED_prob = (np.abs(Y)**2).flatten()
    # key = random.PRNGKey(0)
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

            print("step %d, Cost=%f, kl=%f, l2=%f, F=%f" % (
                step, data_dict['cost_avg'][-1], data_dict['kl_avg'][-1], data_dict['l2_avg'][-1], np.abs(overlap)**2))
            np.save(ckpt_path + '/chi%d_T%.2f.npy' %
                    (chi, T), mps_mat_list, allow_pickle=True)
            np.save(ckpt_path + '/data_dict.npy', data_dict, allow_pickle=True)

        if step % 10000 == 0:
            if (np.abs(overlap)**2 > 1 - 1e-4) or (data_dict['cost_avg'][-1] > np.average(data_dict['cost_avg'][-10:-5])):
                # if (np.abs(overlap)**2 > 1 - 1e-4) or (np.average(data_dict['cost_avg'][-10:]) > np.average(data_dict['cost_avg'][-20:-10])):
                break
            else:
                pass

    # Either call this or call mps function
    mps_list = mps_func.lpr_2_plr(mps_list)
    Sx_list = [np.array([[0., 1.], [1., 0.]]) for i in range(len(mps_mat_list))]
    data_dict['Sx_list'] = np.real(
        mps_func.expectation_values_1_site(mps_list, Sx_list))
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


def training_sgd(mps_mat_list, X, Y, opt_type,
                 num_iter=10000, lr=0.05, batch_size=512,
                 exact_mps=None, data_dict=None, T=None, chi=None, ckpt_path=None
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
    if opt_type == 'sgd':
        opt_init, opt_update, get_params = jax.experimental.optimizers.sgd(
            lr)
    elif opt_type == 'mom':
        opt_init, opt_update, get_params = jax.experimental.optimizers.momentum(
            lr, 0.95)
    elif opt_type == 'adam':
        opt_init, opt_update, get_params = jax.experimental.optimizers.adam(
            lr)
    else:
        raise NotImplementedError

    cost_function = cost_function_overlap_unnormalized
    cost_function = cost_function_joint_unnormlized

    @jit
    def update(idx, opt_state, data):
        params = get_params(opt_state)
        gradient_direction = jax.tree_util.tree_map(jnp.conj,
                                                    jax.grad(cost_function)(
                                                        params, *data)
                                                    )
        # gradient_direction = jax.grad(cost_function)(params, *data)
        return opt_update(idx, gradient_direction, opt_state)

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

    print(X.shape, Y.shape)
    ED_prob = (np.abs(Y)**2).flatten()
    # key = random.PRNGKey(0)
    for step in range(1, num_iter+1):
        batch_mask = np.random.choice(len(Y), batch_size, p=ED_prob)
        # batch_mask = jax.random.choice(key, len(Y), [batch_size], p=ED_prob);
        # key, subkey = random.split(key)

        X_mini_batch = X[batch_mask]
        X_mini_batch = X_mini_batch[:, :, 0].astype(int)
        Y_mini_batch = Y[batch_mask].flatten()
        data = [X_mini_batch, Y_mini_batch]

        opt_state = update(step, opt_state, data)

        # list_opt_state = list(opt_state)
        # params = list_opt_state[0]
        # params = jax.tree_util.tree_map(lambda x: x / jnp.linalg.norm(x),
        #                                 get_params(opt_state))
        # list_opt_state[0] = params
        # opt_state = tuple(list_opt_state)

        # params = jax.tree_util.tree_map(lambda x: x / jnp.amax(x),
        #                                 get_params(opt_state))
        # opt_state = opt_init(params)


        current_cost = cost_function(get_params(opt_state), *data)
        current_cost_kl = cost_function_kl_unnormalized(get_params(opt_state), *data)
        current_cost_l2 = cost_function_l2(get_params(opt_state), *data)

        # [TODO] to delete this
        # print("step : ", step, "cost : ", current_cost, current_cost_kl, current_cost_l2)
        # mps_list = mat_2_mps(get_params(opt_state))
        # print("overlap, ... ", mps_func.overlap_lpr(mps_list, mps_list))

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

            mps_mat_list = get_params(opt_state)  # [l, p, r]
            mps_list = mat_2_mps(mps_mat_list)  # [l, p, r]
            mps_list = mps_func.lpr_2_plr(mps_list)  # [p, l, r])

            # mps_list, trunc_err = mps_func.left_canonicalize(mps_list, no_trunc=True, normalized=True)
            # assert np.isclose(trunc_err, 0)
            # reset_mps_list = mps_func.plr_2_lpr(mps_list)  # [l, p, r]
            # reset_mps_mat_list = mps_2_mat(reset_mps_list)  # [l, p, r]
            # opt_state = opt_init(reset_mps_mat_list)  # [l, p, r]


            overlap = mps_func.overlap(mps_list, exact_mps)
            overlap = overlap / np.sqrt(np.abs(mps_func.overlap(mps_list, mps_list)))
            mps_list = mps_func.plr_2_lpr(mps_list)
            data_dict['fidelity'].append(np.abs(overlap)**2)

            print("step %d, Cost=%f, kl=%f, l2=%f, F=%f" % (
                step, data_dict['cost_avg'][-1], data_dict['kl_avg'][-1], data_dict['l2_avg'][-1], np.abs(overlap)**2))
            np.save(ckpt_path + '/chi%d_T%.2f.npy' %
                    (chi, T), mps_mat_list, allow_pickle=True)
            np.save(ckpt_path + '/data_dict.npy', data_dict, allow_pickle=True)

        if step % 10000 == 0:
            if (np.abs(overlap)**2 > 1 - 1e-4) or (data_dict['cost_avg'][-1] > np.average(data_dict['cost_avg'][-10:-5])):
                # if (np.abs(overlap)**2 > 1 - 1e-4) or (np.average(data_dict['cost_avg'][-10:]) > np.average(data_dict['cost_avg'][-20:-10])):
                break
            else:
                pass

    # Either call this or call mps function
    mps_list = mps_func.lpr_2_plr(mps_list)

    mps_list, trunc_err = mps_func.left_canonicalize(mps_list, no_trunc=True, normalized=True)
    assert np.isclose(trunc_err, 0)

    Sx_list = [np.array([[0., 1.], [1., 0.]]) for i in range(len(mps_mat_list))]
    data_dict['Sx_list'] = np.real(
        mps_func.expectation_values_1_site(mps_list, Sx_list))
    data_dict['SvN_list'] = mps_func.get_entanglement(mps_list)
    data_dict['S2_list'] = mps_func.get_renyi_n_entanglement(mps_list, 2)
    mps_list = mps_func.plr_2_lpr(mps_list)

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
    manifold = jax_opt.manifolds.StiefelManifold(
        metric='euclidean', retraction='svd')
    if opt_type == 'r_sgd':
        opt_init, opt_update, get_params = jax_opt.optimizers.r_sgd(
            lr, manifold)
    elif opt_type == 'r_mom':
        opt_init, opt_update, get_params = jax_opt.optimizers.r_momentum(
            lr, manifold, 0.95)
    elif opt_type == 'r_adam':
        opt_init, opt_update, get_params = jax_opt.optimizers.r_adam(
            lr, manifold)
    else:
        raise NotImplementedError

    cost_function = cost_function_fidelity

    @jit
    def update(idx, opt_state, data):
        params = get_params(opt_state)
        gradient_direction = jax.tree_util.tree_map(jnp.conj,
                                                    jax.grad(cost_function)(
                                                        params, data)
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
        opt_state = update(step, opt_state, data=exact_mps)
        current_cost = cost_function(get_params(opt_state), exact_mps)
        print("step : ", step, "cost : ", current_cost)
        cost_list.append(current_cost)

    return get_params(opt_state), cost_list
