'''
In this file, the unitary is labeled with the output dimension first, i.e.
U_ij |\psi_j>
instead o f U_{ji} | \psi_j>
'''
import jax
import jax.numpy as np
from jax.ops import index, index_add, index_update
from jax.config import config
config.update("jax_enable_x64", True)
import sys; sys.path.append('../')
import jax_opt.optimizers
import jax_opt.manifolds

import numpy as onp
import pickle
import os, sys
import circuit_func_jax as circuit_func
import misc
import parse_args
import mps_func

'''
    Algorithm:
        (1.) first call circuit_2_state to get the (list of) exact reprentation of
        circuit up to each layer.
        (2.) Load the target state | psi >
        (3.) maximizing the overlap | < psi | circuit > |^2
        ...
'''
def cost_function_fidelity(circuit, target_state, product_state):
    '''
    The cost function to optimize
    C = 1. - F
    F = | <target_state | circuit | product_state > |^2

    Input:
        circuit: list of list of unitary
        target_state: a quantum many-body state
        product_state: a product state which circuit acts on.
    Output:
        loss
    '''

    iter_state = circuit_func.circuit_2_state(circuit, product_state)
    fidelity = np.abs(circuit_func.overlap_exact(target_state, iter_state))**2
    return 1. - fidelity


def rsgd(circuit, target_state, product_state, opt_type, num_iter=10000, lr=0.05, brickwall=True):
    '''
    Inputs:
        cirucit
        target_state
        product_state
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

    def update(idx, opt_state, data):
        params = get_params(opt_state)
        gradient_direction = jax.tree_util.tree_map(np.conj,
                                                    jax.grad(cost_function_fidelity)(params, *data)
                                                   )

        if brickwall:
            ### Brickwall condition ###
            for dep_idx, layer in enumerate(gradient_direction):
                for site_idx, U in enumerate(layer):
                    if (dep_idx + site_idx) % 2 != 0:
                        gradient_direction[dep_idx][site_idx] *= 0.
        else:
            pass


        return opt_update(idx,
                          gradient_direction,
                          opt_state)


    #################
    # set up params #
    #################

    params = jax.tree_util.tree_map(np.complex128,
                                    circuit)
    # params = circuit
    opt_state = opt_init(params)

    data = [target_state, product_state]
    loss_list = []
    current_loss = cost_function_fidelity(get_params(opt_state), *data)
    print("begin: ", 1.-current_loss)
    loss_list.append(current_loss)

    for step in range(num_iter):
        opt_state = update(step, opt_state, data)
        A = get_params(opt_state)[0][0]

        current_loss = cost_function_fidelity(get_params(opt_state), *data)
        print(step, 1.-current_loss)
        loss_list.append(current_loss)

    return loss_list



if __name__ == "__main__":
    onp.random.seed(1)
    np.set_printoptions(linewidth=2000, precision=5, threshold=4000)

    args = parse_args.parse_args()

    L = args.L
    J = 1.
    depth = args.depth
    N_iter = args.N_iter
    T = args.T  # the target state is corresponding to time T.


    save_each = 100
    tol = 1e-12
    cov_crit = tol * 0.1
    max_N_iter = N_iter

    Sz_list = [np.array([[1, 0.], [0., -1.]]) for i in range(L)]


    ############### LOAD TARGET STATE ######################

    ## One can modify this part to load different target state.
    ## Provide filename in variable "filename"
    ## Load the state by

    ## `` target_mps = pickle.load(open(filename, 'rb')) ``

    ## Here we give an example of cat state, i.e. GHZ state
    ## as target state.

    # cat_state = onp.zeros([2**L])
    # cat_state[0] = cat_state[-1] = 1./np.sqrt(2)

    ## Here we give an example of Haar random state
    ## as target state.

    random_state = onp.random.normal(0, 1, [2**L]) + 1j * onp.random.normal(0, 1, [2**L])
    random_state /= np.linalg.norm(random_state)
    target_state = random_state



    ############### PARAMETERS INITIALIZATION #################
    ## We should test identity initialization and
    ## trotterization initialization
    dt = T / depth

    idx = 0
    my_circuit = []
    t_list = [0]
    error_list = []
    Sz_array = np.zeros([N_iter, L], dtype=np.complex128)
    ent_array = np.zeros([N_iter, L-1], dtype=np.float64)


    ################# CIRCUIT INITIALIZATION  ######################
    # product_state = [np.array([1., 0.]).reshape([2, 1, 1]) for i in range(L)]
    product_state = np.zeros([2**L], dtype=np.complex128)
    # product_state[0] = 1.
    product_state = index_add(product_state, index[0], 1.)


    for dep_idx in range(depth):
        # identity_layer = [np.eye(4, dtype=np.complex128).reshape([2, 2, 2, 2]) for i in range(L-1)]
        # my_circuit.append(identity_layer)

        random_layer = []
        for idx in range(L-1):
            if (idx + dep_idx) % 2 == 0:
                ## We add .T to compare with the result from polor decomposition
                ## Over there the convention is ji, instead of ij.
                random_layer.append(circuit_func.random_2site_U(2).T)
            else:
                ## (set to identity for brickwall circuit)
                random_layer.append(onp.eye(4, dtype=np.complex128).reshape([4,4]))

        my_circuit.append(random_layer)
        current_depth = dep_idx + 1

    iter_state = circuit_func.circuit_2_state(my_circuit, product_state)
    '''
    Sz_array[0, :] = mps_func.expectation_values_1_site(mps_of_layer[-1], Sz_list)
    ent_array[0, :] = mps_func.get_entanglement(mps_of_last_layer)
    '''
    fidelity_reached = np.abs(circuit_func.overlap_exact(target_state, iter_state))**2
    print("fidelity reached : ", fidelity_reached)
    error_list.append(1. - fidelity_reached)


    loss_list = rsgd(my_circuit, target_state, product_state, opt_type='radam', num_iter=N_iter, lr=0.5)
    print(loss_list)
    '''
    TODO: fix the convention in circuit_func_jax, the var_exact is still using the old convention.
    '''

    stop_crit = 1e-1
    assert np.isclose(circuit_func.overlap_exact(target_state, target_state), 1.)
    for idx in range(0, N_iter):
        #################################
        #### variational optimzation ####
        #################################
        # mps_of_last_layer, my_circuit = circuit_func.var_circuit(target_mps, mps_of_last_layer,
        #                                                   my_circuit, product_state)

        iter_state, my_circuit = circuit_func.var_circuit_exact(target_state, iter_state,
                                                         my_circuit, product_state, brickwall=True)
        #################
        #### Measure ####
        #################
        assert np.isclose(circuit_func.overlap_exact(iter_state, iter_state), 1.)
        '''
        Sz_array[idx, :] = mps_func.expectation_values_1_site(mps_of_last_layer, Sz_list)
        ent_array[idx, :] = mps_func.get_entanglement(mps_of_last_layer)
        '''
        fidelity_reached = np.abs(circuit_func.overlap_exact(target_state, iter_state))**2

        print("fidelity reached : ", fidelity_reached)
        error_list.append(1. - fidelity_reached)
        t_list.append(idx)

        ################
        ## Forcing to stop if already converge
        ################
        if (fidelity_reached > 1 - 1e-12 or np.abs((error_list[-1] - error_list[-2])/error_list[-1]) < 1e-4) and idx > save_each:
            break

    num_data = len(error_list)
    '''
    Sz_array = Sz_array[:num_data, :]
    ent_array = ent_array[:num_data, :]
    '''



