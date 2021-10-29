import jax
jax.config.update('jax_enable_x64', True)

from jax.experimental.optimizers import optimizer, make_schedule


@optimizer
def r_adam(step_size, manifold, b1=0.9, b2=0.999, eps=1e-8):
    """
    Construct optimizer triple for Adam.
    Args:
        step_size:
            positive scalar, or a callable representing a step size schedule
            that maps the iteration index to positive scalar.
        manifold:
            the manifold on which the Riemannian optimization is performed.
        b1:
            optional, a positive scalar value for beta_1, the exponential decay rate
            for the first moment estimates (default 0.9).
        b2:
            optional, a positive scalar value for beta_2, the exponential decay rate
            for the second moment estimates (default 0.999).
        eps:
            optional, a positive scalar value for epsilon, a small constant for
            numerical stability (default 1e-8).

    Returns:
        An (init_fun, update_fun, get_params) triple.
    """
    step_size = make_schedule(step_size)
    def init(x0):
        m0 = jax.numpy.zeros_like(x0)
        v0 = jax.numpy.zeros_like(x0)
        return x0, m0, v0

    def update(i, grad, state):
        x, m, v = state
        rgrad = manifold.egrad_to_rgrad(x, grad)
        m = (1 - b1) * rgrad + b1 * m  # First  moment estimate.
        ## v = (1 - b2) * jax.numpy.square(rgrad) + b2 * v  # Second moment estimate.
        ##
        ## The square of rgrad is estimated by manifold inner
        ## https://arxiv.org/pdf/2002.01113.pdf
        ## ... manifold-wise adaptive learning rate that assign a same
        ## learning rate for all entries in a parameter matrix as in (Absil et al., 2009)
        v = (1 - b2) * manifold.inner(x, rgrad, rgrad) + b2 * v

        mhat = m / (1 - b1 ** (i + 1))  # Bias correction.
        vhat = v / (1 - b2 ** (i + 1))

        # search_dir = -step_size(i) * mhat / (jax.numpy.sqrt(vhat) + eps)
        search_dir = -step_size(i) * mhat / (jax.numpy.sqrt(vhat) + eps * jax.numpy.sqrt(1 - b2 ** (i + 1)))
        x, m = manifold.retraction_transport(x, m, search_dir)
        return x, m, v

    def get_params(state):
        x, _, _ = state
        return x

    return init, update, get_params
