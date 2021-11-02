import os, sys
sys.path.append('../../')
import tensor_network_functions.mps_func as mps_func
import supervised_mps
import parse_args
import numpy as np


if __name__ == '__main__':
    args = parse_args.parse_args()
    if bool(args.debug):
        np.random.seed(0)

    L = args.L
    batch_size = args.batch_size
    assert batch_size == 512
    chi = args.chi
    supervised_model = args.supervised_model  # '1D_ZZ_1.00X_0.25XX_global_TE_L20'
    T = args.T
    path = args.path
    if len(path) > 0 and path[-1] != '/':
        path = path + '/'

    #################
    # Prepare DATA  #
    ######################
    N_sys = L
    X_computation_basis = np.genfromtxt(
        'ExactDiag/basis_L%d.csv' % L, delimiter=',')
    X = np.zeros([2**L, L, 2], dtype=int)
    X[:, :, 0] = X_computation_basis.reshape([2**L, L]).astype(int)
    X[:, :, 1] = 1-X_computation_basis.reshape([2**L, L]).astype(int)
    Y = np.load('ExactDiag/wavefunction/%s/ED_wf_T%.2f.npy' %
                (supervised_model, T))
    Y = np.array(Y, dtype=np.complex128)[:, None]

    ckpt_path = path + \
        'wavefunction/Supervised/' + '%s_T%.2f/' % (supervised_model, T) + \
        'MPS_%d' % chi

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    try:
        data_dict = np.load(ckpt_path + '/data_dict.npy',
                            allow_pickle=True).item()
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
        mps_mat_list = np.load(ckpt_path + '/chi%d_T%.2f.npy' %
                               (chi, T), allow_pickle=True)
        mps_mat_list = [mps for mps in mps_mat_list]
        print(" load success !!! \n\n\n")
    except Exception as e:
        print("error = ", e)
        mps_list = mps_func.init_mps(L=L, chi=chi, d=2)
        mps_mat_list = supervised_mps.mps_2_mat(mps_list)

    exact_mps = mps_func.state_2_MPS(Y, 20, 1024)

    mps_list = supervised_mps.mat_2_mps(mps_mat_list)
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
    #     yi = mps_func.get_mps_amp_batch_jax(mps_list, X[start_idx:end_idx,:,0].astype(int))
    #     y_list.append(yi)

    # if end_idx != 2**N_sys:
    #     yi = mps_func.get_mps_amp_batch_jax(mps_list, X[end_idx:,:,0].astype(int))
    #     y_list.append(yi)

    # y = np.concatenate(y_list)
    # print(('y norm : ', np.linalg.norm(y)))
    # measured_fidelity = np.square(np.abs(Y.flatten().dot(y.flatten().conj())))
    # print(" explicit fidelity = ", measured_fidelity)
    # ###################################################################

    ###################################################################
    mps_mat_list, data_dict = supervised_mps.training_sgd(mps_mat_list, X, Y,
                                                          opt_type='adam', num_iter=300000,
                                                          batch_size=batch_size, lr=1e-2,
                                                          exact_mps=exact_mps,
                                                          data_dict=data_dict, T=T, chi=chi, ckpt_path=ckpt_path
                                                          )
    mps_mat_list, data_dict = supervised_mps.training_sgd(mps_mat_list, X, Y,
                                                          opt_type='adam', num_iter=300000,
                                                          batch_size=batch_size, lr=1e-3,
                                                          exact_mps=exact_mps,
                                                          data_dict=data_dict, T=T, chi=chi, ckpt_path=ckpt_path
                                                          )
    mps_mat_list, data_dict = supervised_mps.training_sgd(mps_mat_list, X, Y,
                                                          opt_type='adam', num_iter=300000,
                                                          batch_size=batch_size, lr=1e-4,
                                                          exact_mps=exact_mps,
                                                          data_dict=data_dict, T=T, chi=chi, ckpt_path=ckpt_path
                                                          )

    # mps_mat_list, cost_list = training_mps(mps_mat_list, exact_mps,
    #                                        opt_type='r_adam', num_iter=1000, lr=1e-2)
    np.save(ckpt_path + '/chi%d_T%.2f.npy' %
            (chi, T), mps_mat_list, allow_pickle=True)
    np.save(ckpt_path + '/data_dict.npy', data_dict, allow_pickle=True)

    ###################################################################
