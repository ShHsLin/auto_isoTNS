import numpy as np
import pickle
import os, sys
sys.path.append('..')
import circuit_func, misc
import parse_args
import mps_func

'''
    Algorithm:
        (1.) first call circuit_2_mps to get the list of mps-reprentation of
        circuit up to each layer.
        (2.) Load the target state | psi >
        (3.) var optimize layer-n by maximizing < psi | U(n) | n-1>
        (4.) collapse layer-n optimized on |psi> getting new |psi>
        (5.) var optimize layer-n-1 by maximizing < psi | U(n-1) | n-2 >
        [TODO] check the index n above whether is consistent with the code.
        ...
'''

if __name__ == "__main__":
    np.random.seed(1)
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

    cat_state = np.zeros([2**L])
    cat_state[0] = cat_state[-1] = 1./np.sqrt(2)
    # target_state = cat_state

    target_state = np.random.normal(0, 1, [2**L]) + 1j * np.random.normal(0, 1, [2**L])
    target_state /= np.linalg.norm(target_state)
    target_mps = mps_func.state_2_MPS(target_state, L, chimax=10000)



    ############### PARAMETERS INITIALIZATION #################
    ## We should test identity initialization and
    ## trotterization initialization
    dt = T / depth

    idx = 0
    my_circuit = []
    t_list = [0]
    error_list = []
    Sz_array = np.zeros([N_iter, L], dtype=np.complex)
    ent_array = np.zeros([N_iter, L-1], dtype=np.double)


    ################# CIRCUIT INITIALIZATION  ######################
    product_state = [np.array([1., 0.]).reshape([2, 1, 1]) for i in range(L)]
    for dep_idx in range(depth):
        # identity_layer = [np.eye(4, dtype=np.complex).reshape([2, 2, 2, 2]) for i in range(L-1)]
        # my_circuit.append(identity_layer)

        random_layer = []
        for idx in range(L-1):
            if (idx + dep_idx) % 2 == 0:
                random_layer.append(circuit_func.random_2site_U(2))
            else:
                random_layer.append(np.eye(4).reshape([2,2,2,2]))

        my_circuit.append(random_layer)
        current_depth = dep_idx + 1

    mps_of_layer = circuit_func.circuit_2_mps(my_circuit, product_state)
    mps_of_last_layer = [A.copy() for A in mps_of_layer[current_depth]]

    Sz_array[0, :] = mps_func.expectation_values_1_site(mps_of_layer[-1], Sz_list)
    ent_array[0, :] = mps_func.get_entanglement(mps_of_last_layer)
    fidelity_reached = np.abs(mps_func.overlap(target_mps, mps_of_last_layer))**2
    error_list.append(1. - fidelity_reached)


    stop_crit = 1e-1
    assert np.isclose(mps_func.overlap(target_mps, target_mps), 1.)
    for idx in range(0, N_iter):
        #################################
        #### variational optimzation ####
        #################################
        # mps_of_last_layer, my_circuit = circuit_func.var_circuit(target_mps, mps_of_last_layer,
        #                                                   my_circuit, product_state)

        mps_of_last_layer, my_circuit = circuit_func.var_circuit2(target_mps, product_state,
                                                           my_circuit, brickwall=True)
        #################
        #### Measure ####
        #################
        assert np.isclose(mps_func.overlap(mps_of_last_layer, mps_of_last_layer), 1.)
        Sz_array[idx, :] = mps_func.expectation_values_1_site(mps_of_last_layer, Sz_list)
        ent_array[idx, :] = mps_func.get_entanglement(mps_of_last_layer)
        fidelity_reached = np.abs(mps_func.overlap(target_mps, mps_of_last_layer))**2

        print("fidelity reached : ", fidelity_reached)
        error_list.append(1. - fidelity_reached)
        t_list.append(idx)

        ################
        ## Forcing to stop if already converge
        ################
        if (fidelity_reached > 1 - 1e-12 or np.abs((error_list[-1] - error_list[-2])/error_list[-1]) < 1e-4) and idx > save_each:
            break

    num_data = len(error_list)
    Sz_array = Sz_array[:num_data, :]
    ent_array = ent_array[:num_data, :]



