import jax
jax.config.update('jax_enable_x64', True)

from jax.experimental.optimizers import optimizer


@optimizer
def r_cg(manifold, f):
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
        return x0, jax.numpy.zeros_like(x0)

    def update(i, grad, state):
        '''
        direction = steepest direction = - nabla f.
        notice the negative sign.

        So we would transport with direction * lr.
        '''
        x, prev_direction = state
        rgrad = manifold.egrad_to_rgrad(x, grad)
        beta = ...
        current_direction = - rgrad + beta * prev_direction

        step_size = line_search(f, current_direction, manifold)

        new_x, current_direction = \
                manifold.retraction_transport(x,
                                              current_direction,
                                              step_size * current_direction)
        return new_x, current_direction

    def get_params(state):
        x, _ = state
        return x

    return init, update, get_params




        x, velocity = state
        rgrad = manifold.egrad_to_rgrad(x, grad)
        # velocity = mass * velocity + rgrad  # both are in tangent space Tx
        velocity = mass * velocity + (1 - mass) * rgrad  # both are in tangent space Tx
        new_x, velocity =\
                manifold.retraction_transport(x,
                                              velocity,
                                              -step_size(i) * velocity)
        return new_x, velocity


