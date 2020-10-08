import tensorflow as tf
import sys; sys.path.insert(0, '../tf_opt/')
import QGOpt as qgo  ## tf_opt from QGOpt library


@tf.function
def renyi_entropy(p_i, alpha):
    if tf.abs(alpha - 1.) < 1e-4:
        filtered_p_i = p_i[p_i > 1e-8]
        return -tf.reduce_sum(filtered_p_i * tf.math.log(filtered_p_i) )
    else:
        return tf.math.log(tf.reduce_sum(tf.math.pow(p_i, alpha))) / ( 1. - alpha)

@tf.function
def get_renyi_entropy(wf, alpha):
    '''
    compute the renyi-alpha entanglement entropy of |wf>
    Input:
        wf: of dimension (chi1, d1, d2, chi2)

    Ourput:
        entropy
        singular values
    '''
    dim_chi1, dim_d1, dim_d2, dim_chi2 = wf.shape
    theta = tf.reshape(wf, [dim_chi1 * dim_d1, dim_d2 * dim_chi2])
    sing_vals, _, _ = tf.linalg.svd(theta, full_matrices=False)
    prob = tf.square(sing_vals)
    return renyi_entropy(prob, alpha), sing_vals

@tf.function
def get_U_wf(U, wf):
    '''
    A wrapper function to reshape unitary U to rank-4 tensor
    tensor_U: of dimension (d1', d2', d1, d2)
    and apply U on wf.
    notice that we pick the convention of
    " U_ij |wf>_j "


    Input:
        U: (dim_d1*dim_d2, dim_d1*dim_d2)
        wf: (dim_chi1, dim_d1, dim_d2, dim_chi2)
    Output:
        U_wf
    '''
    dim_chi1, dim_d1, dim_d2, dim_chi2 = wf.shape
    tensor_U = tf.reshape(U, [dim_d1, dim_d2, dim_d1, dim_d2])
    U_wf = tf.tensordot(tensor_U, wf, ([2, 3], [1, 2]))  ## d1',d2', chi1, chi2
    U_wf = tf.transpose(U_wf, [2, 0, 1, 3]) ## chi1, d1', d2', chi2
    return U_wf


@tf.function
def get_U_wf_renyi_entropy(U, wf, alpha):
    '''
    A wrapper function to reshape unitary U to rank-4 tensor
    tensor_U: of dimension (d1', d2', d1, d2)
    and compute the renyi-alpha entanglement entropy of U|wf>
    notice that we pick the convention of
    " U_ij |wf>_j "

    Input:
        U: (dim_d1*dim_d2, dim_d1*dim_d2)
        wf: (dim_chi1, dim_d1, dim_d2, dim_chi2)
    Output:
        loss
        sing_vals
    '''
    U_wf = get_U_wf(U, wf)
    loss, sing_vals = get_renyi_entropy(U_wf, alpha)
    return loss, sing_vals

@tf.function
def disentangling_step(u, wf, alpha):
    print("connect to disentangling step")
    '''
    minimizing renyi-alpha entropy
    '''
    with tf.GradientTape() as tape:
        # transforming real variable back to the complex representation
        # (it is only necessary to have real variables, but in the body
        # of a graph on can use complex tensors)
        uc = qgo.manifolds.real_to_complex(u)
        # uc = tf.complex(u[..., 0], u[..., 1])

        U_wf = get_U_wf(uc, wf)

        renyi_half, _ = get_renyi_entropy(U_wf, tf.constant(0.5, dtype=tf.float64))
        renyi_1, _ = get_renyi_entropy(U_wf, tf.constant(1, dtype=tf.float64))
        renyi_2, sing_vals = get_renyi_entropy(U_wf, tf.constant(2., dtype=tf.float64))

        loss, _ = get_renyi_entropy(U_wf, alpha)
        # loss, _ = get_U_wf_renyi_entropy(uc, wf, alpha)
        # loss, _ = get_U_wf_renyi_entropy(qgo.manifolds.real_to_complex(u), wf, alpha)

    grads = tape.gradient(loss, u)  # gradient

    return grads, loss, renyi_half, renyi_1, renyi_2, sing_vals

@tf.function
def opt_trunc_step(u, wf, bond_dimension):
    print("connect to opt_trunc")
    '''
    minimizing the truncation error = sum(sing_vals[bond_dim:]**2)
    '''
    with tf.GradientTape() as tape:
        # transforming real variable back to the complex representation
        # (it is only necessary to have real variables, but in the body
        # of a graph on can use complex tensors)
        uc = qgo.manifolds.real_to_complex(u)
        # uc = tf.complex(u[..., 0], u[..., 1])

        U_wf = get_U_wf(uc, wf)

        renyi_half, _ = get_renyi_entropy(U_wf, tf.constant(0.5, dtype=tf.float64))
        renyi_1, _ = get_renyi_entropy(U_wf, tf.constant(1, dtype=tf.float64))
        renyi_2, sing_vals = get_renyi_entropy(U_wf, tf.constant(2, dtype=tf.float64))

        loss = tf.reduce_sum(tf.square(sing_vals[bond_dimension:]))

    grads = tape.gradient(loss, u)  # gradient

    return grads, loss, renyi_half, renyi_1, renyi_2, sing_vals



def find_u(wf, loss, loss_para, opt_type, iters=10000,
           lr=0.5,):
    '''
    Inputs:
        wf: wavefunction/tensor to be disentangled.
            we assume it is of the shape (chi1, d1, d2, chi2)
        loss:
            the loss function to be minimized.
            (1.) truncation error (2.) Renyi-alpha entropy
        loss_para:
            When loss function == truncation error,
            one should specify the bond dimension, i.e.
            the number of singular values to kept,
            When loss function == Renyi-alpha entropy,
            one should specify the alhpa
        opt_type:
        iters: number of iterations
        lr: the learning rate
    Outputs:
        wf: the original(?) wf
        u_opt: the optimal unitary to act on wf
        info: the information about the minimization process in disentangling.
    '''
    chi1, d1, d2, chi2 = wf.shape
    wf = tf.convert_to_tensor(wf, dtype=tf.complex128)
    # [TODO]: Add checking wf tf or numpy
    if loss == 'trunc':
        opt_step = opt_trunc_step
        loss_para = tf.constant(loss_para, dtype=tf.int32)
    elif loss == 'renyi':
        opt_step = disentangling_step
        loss_para = tf.constant(loss_para, dtype=tf.float64)
    else:
        raise NotImplementedError


    if opt_type == 'EV':
        EV_alg = True
    else:
        EV_alg = False


    manifold = qgo.manifolds.StiefelManifold()
    #=================================#
    uc = tf.eye(d1*d2, dtype=tf.complex128)
    u = tf.Variable(qgo.manifolds.complex_to_real(uc))
    # uc = manifold.random([d1*d2, d1*d2], dtype=tf.complex128)
    # u = tf.Variable(qgo.manifolds.complex_to_real(uc))
    #=================================#

    # Riemannian Adam,
    # we pass m that is an example of
    # complex Stiefel manifold to guide optimizer
    # how to perform optimization on complex
    # Stiefel manifold
    manifold = qgo.manifolds.StiefelManifold()
    if opt_type is None or opt_type == 'Adam':
        opt = qgo.optimizers.RAdam(manifold, lr, ams=False)
    elif opt_type == 'SGD' or opt_type =='EV':
        opt = qgo.optimizers.RSGD(manifold, lr, momentum=0.95)
    else:
        raise

    errs = [] # will be filled by err vs number of iterations
    renyi_2_list = []
    renyi_1_list = []
    renyi_half_list = []

    sing_vals_list = []


    for tf_idx in tf.range(1, iters+1):
        grads, loss, renyi_half, renyi_1, renyi_2, sing_vals = opt_step(u, wf, loss_para)
        if EV_alg:
            grads_c = qgo.manifolds.real_to_complex(grads)
            _, UU, VV = tf.linalg.svd(grads_c)
            update_u = UU @ tf.transpose(tf.math.conj(VV))
            u.assign(qgo.manifolds.complex_to_real(update_u))
        else:
            opt.apply_gradients(zip([grads], [u]))  # optimization step



        # errs.append(tf.math.sqrt(loss))
        renyi_2_list.append(renyi_2.numpy())
        renyi_1_list.append(renyi_1.numpy())
        renyi_half_list.append(renyi_half.numpy())
        errs.append(loss.numpy())
        # assert np.isclose(np.sum(sing_vals.numpy() ** 2), 1.)
        try:
            assert tf.math.abs(tf.reduce_sum(sing_vals ** 2) - 1.) < 1e-6
        except:
            import pdb;pdb.set_trace()

        sing_vals_list.append(sing_vals.numpy())


        if errs[-1] < 1e-16 or (tf_idx > 1000 and
                                np.abs(np.mean(errs[-5:]) - errs[-1])/np.abs(errs[-1]) < 1e-5):
            break




    eye_uc = tf.eye(d1*d2, dtype=tf.complex128)
    _, original_sing = get_U_wf_renyi_entropy(eye_uc, wf, 2.)
    data = {}
    data['original_sing'] = original_sing.numpy()
    data['errs'] = errs
    data['renyi_2_list'] = renyi_2_list
    data['renyi_1_list'] = renyi_1_list
    data['renyi_half_list'] = renyi_half_list
    data['sing_vals_list'] = sing_vals_list

    uc = qgo.manifolds.real_to_complex(u)
    return wf.numpy(), uc.numpy(), data


if __name__ == '__main__':
    import numpy as np
    np.random.seed(0)

    test_tensor = np.random.rand(4, 2, 2, 4) - 0.5 +\
            1j*(np.random.rand(4, 2, 2, 4) - 0.5)
    U, S, Vd = np.linalg.svd(test_tensor.reshape([8, 8]))
    S /= np.linalg.norm(S)
    test_tensor = U.dot(np.diag(S).dot(Vd)).reshape([4, 2, 2, 4])

    wf, u, data = find_u(test_tensor, 'trunc', 6, 'SGD', 100, lr=0.5)
    # wf, u, data = find_u(test_tensor, 'renyi', 2, 'Adam', 100, lr=0.5)
    print(data['errs'])
