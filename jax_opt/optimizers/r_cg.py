import jax
jax.config.update('jax_enable_x64', True)

#from jax.experimental.optimizers import optimizer
from jax.example_libraries.optimizers import optimizer


@optimizer
def r_cg(manifold, f, data):
    """
    Construct optimizer triple for stochastic gradient descent.
    we do not take step size as argument because we perform line search.

    Args:
        manifold: on which the Riemannian optimization is performed.
        f: callable. Function to minimize

    Returns:
        An (init_fun, update_fun, get_params) triple.
    """
    def init(x0):
        return x0, jax.numpy.zeros_like(x0), jax.numpy.ones_like(x0)

    def update(i, grad, state):
        '''
        direction = steepest direction = - nabla f.
        notice the negative sign.

        So we would transport with direction * lr.
        '''
        x, prev_direction, prev_r_grad = state
        rgrad = manifold.egrad_to_rgrad(x, grad)
        # beta = manifold.inner(x, rgrad, (rgrad-prev_r_grad)) / manifold.inner(x, prev_r_grad, prev_r_grad)
        beta = manifold.inner(x, rgrad, rgrad) / manifold.inner(x, prev_r_grad, prev_r_grad)
        current_direction = - rgrad + beta * prev_direction

        line_search = BacktrackingLineSearch
        step_size = line_search(f, None, x, current_direction, manifold, df_x=rgrad, args=data)

        new_x, current_direction = \
                manifold.retraction_transport(x,
                                              current_direction,
                                              step_size * current_direction)
        new_x, rgrad = \
                manifold.retraction_transport(x,
                                              rgrad,
                                              step_size * current_direction)


        return new_x, current_direction, rgrad

    def get_params(state):
        x, _, _ = state
        return x

    return init, update, get_params


def BacktrackingLineSearch(f, df, x, p, manifold, df_x = None, f_x = None, args = (),
        alpha = 0.0001, beta = 0.9, eps = 1e-8, Verbose = False):
    """
    Backtracking linesearch
    f: function
    x: current point
    p: direction of search
    df_x: gradient at x
    f_x = f(x) (Optional)
    args: optional arguments to f (optional)
    alpha, beta: backtracking parameters
    eps: (Optional) quit if norm of step produced is less than this
    Verbose: (Optional) Print lots of info about progress

    Reference: Nocedal and Wright 2/e (2006), p. 37

    Usage notes:
    -----------
    Recommended for Newton methods; less appropriate for quasi-Newton or conjugate gradients
    """

    if f_x is None:
        f_x = f(x, *args)
    if df_x is None:
        df_x = df(x, *args)

    assert 0 < alpha < 1, 'Invalid value of alpha in backtracking linesearch'
    assert 0 < beta < 1, 'Invalid value of beta in backtracking linesearch'

    # derphi = jax.numpy.dot(df_x, p)
    derphi = manifold.inner(x, df_x, p)

    assert derphi.shape == (1, 1) or derphi.shape == ()
    assert derphi < 0, 'Attempted to linesearch uphill'

    stp = 1.0
    fc = 0
    len_p = jax.numpy.linalg.norm(p)


    #Loop until Armijo condition is satisfied
    while f(manifold.retraction(x, stp*p), *args) > f_x + alpha * stp * derphi:
        stp *= beta
        fc += 1
        if Verbose:
            print('linesearch iteration', fc, ':', stp,
                  f(manifold.retraction(x, stp*p), *args), f_x + alpha * stp * derphi)
        if stp * len_p < eps:
            print('Step is  too small, stop')
            break

    if Verbose:
        print('linesearch done')

    return stp

