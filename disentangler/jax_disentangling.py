import jax
from jax import jit, grad
import sys; sys.path.append('../')
import jax_opt.optimizers
import jax_opt.manifolds

def renyi_entropy(p_i, alpha):
    if jax.numpy.abs(alpha - 1.) < 1e-4:
        filtered_p_i = p_i[p_i > 1e-8]
        return -jax.numpy.sum(filtered_p_i * tf.math.log(filtered_p_i) )
    else:
        return jax.numpy.log(jax.numpy.sum(jax.numpy.power(p_i, alpha))) / ( 1. - alpha)


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
    theta = jax.numpy.reshape(wf, [dim_chi1 * dim_d1, dim_d2 * dim_chi2])
    _, sing_vals, _ = jax.numpy.linalg.svd(theta, full_matrices=False)
    prob = jax.numpy.square(sing_vals)
    return renyi_entropy(prob, alpha), sing_vals

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
    tensor_U = jax.numpy.reshape(U, [dim_d1, dim_d2, dim_d1, dim_d2])
    U_wf = jax.numpy.tensordot(tensor_U, wf, ([2, 3], [1, 2]))  ## d1',d2', chi1, chi2
    U_wf = jax.numpy.transpose(U_wf, [2, 0, 1, 3]) ## chi1, d1', d2', chi2
    return U_wf

def loss_renyi_entropy(U, wf, alpha):
    '''
    Loss Function.
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
    return loss

def loss_truncation(U, wf, bond_dimension):
    '''
    Loss Function.
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
    _, sing_vals = get_renyi_entropy(U_wf, 2.)
    loss = jax.numpy.sum(jax.numpy.square(sing_vals[bond_dimension:]))
    return loss

def disentangling_step(u, wf, alpha):
    print("connect to disentangling step")
    '''
    minimizing renyi-alpha entropy
    '''

    raise NotImplementedError

    # Should rewrite loss function as def loss(params, data) --> loss
    # so that one can easily do
    # value, grads = jax.value_and_grad(loss_fn)(opt.get_params(opt_state))


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
        # loss, _ = loss_renyi_entropy(uc, wf, alpha)
        # loss, _ = loss_renyi_entropy(qgo.manifolds.real_to_complex(u), wf, alpha)

    grads = tape.gradient(loss, u)  # gradient

    return grads, loss, renyi_half, renyi_1, renyi_2, sing_vals

def opt_trunc_step(u, wf, bond_dimension):
    print("connect to opt_trunc")
    '''
    minimizing the truncation error = sum(sing_vals[bond_dim:]**2)
    '''

    raise NotImplementedError
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


def find_u(wf, loss_type, loss_para, opt_type, iters=10000, lr=0.5):
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
    wf = jax.numpy.complex128(wf)

    if loss_type == 'trunc':
        loss_fn = loss_truncation
        loss_para = jax.numpy.int32(loss_para)
        # opt_step = opt_trunc_step
    elif loss_type == 'renyi':
        loss_fn = loss_renyi_entropy
        loss_para = jax.numpy.float64(loss_para)
        # opt_step = disentangling_step
    else:
        raise NotImplementedError


    # manifold = jax_opt.manifolds.StiefelManifold(metric='euclidean', retraction='svd')
    manifold = jax_opt.manifolds.StiefelManifold()
    if opt_type == 'rsgd':
        opt_init, opt_update, get_params = jax_opt.optimizers.rsgd(lr, manifold)
    elif opt_type == 'rmom':
        opt_init, opt_update, get_params = jax_opt.optimizers.rmomentum(lr, manifold, 0.95)
    elif opt_type == 'radam':
        opt_init, opt_update, get_params = jax_opt.optimizers.radam(lr, manifold)
    else:
        raise NotImplementedError


    # @jit
    def update(idx, opt_state, data):
        params = get_params(opt_state)
        return opt_update(idx,
                          jax.numpy.conj(grad(loss_fn)(params, *data)),
                          opt_state)

    # set up params
    U = jax.numpy.eye(d1*d2, dtype=jax.numpy.complex128)
    params = U
    opt_state = opt_init(params)

    data = [wf, loss_para]
    loss_list = []
    current_loss = loss_fn(get_params(opt_state), *data)
    loss_list.append(current_loss)

    for step in range(iters):
        opt_state = update(step, opt_state, data)

        current_loss = loss_fn(get_params(opt_state), *data)
        loss_list.append(current_loss)

    params = get_params(opt_state)
    import numpy
    print(numpy.array(loss_list))

    return params

if __name__ == '__main__':
    import numpy as np
    np.random.seed(0)

    test_tensor = np.random.rand(4, 2, 2, 4) - 0.5 +\
            1j*(np.random.rand(4, 2, 2, 4) - 0.5)

    U, S, Vd = np.linalg.svd(test_tensor.reshape([8, 8]))
    S /= np.linalg.norm(S)
    wf = U.dot(np.diag(S).dot(Vd)).reshape([4, 2, 2, 4])

    # find_u(wf, 'renyi', 2, 'radam', iters=100, lr=0.5)
    u = find_u(wf, 'trunc', 6, 'rmom', iters=100, lr=0.5)
