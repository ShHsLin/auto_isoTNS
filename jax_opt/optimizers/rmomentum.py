import jax
jax.config.update('jax_enable_x64', True)

from jax.experimental.optimizers import optimizer, make_schedule


@optimizer
def rmomentum(step_size, manifold, mass):
    """Construct optimizer triple for stochastic gradient descent.
    Args:
        step_size:
            positive scalar, or a callable representing a step size schedule
            that maps the iteration index to positive scalar.
        manifold:
            the manifold to perform riemannian optimization on.
        mass:
            positive scaler representing the momentum coefficient

    Returns:
        An (init_fun, update_fun, get_params) triple.
    """
    step_size = make_schedule(step_size)
    def init(x0):
        return x0, jax.numpy.zeros_like(x0)

    def update(i, grad, state):
        '''
        x, velocity = state
        velocity = mass * velocity + g
        x = x - step_size(i) * velocity
        return x, velocity
        '''
        rgrad = manifold.egrad_to_rgrad(x, grad)
        # velocity = mass * velocity + rgrad  # both are in tangent space Tx
        velocity = mass * velocity + (1 - mass) * rgrad  # both are in tangent space Tx
        new_x, velocity =\
                manifold.retraction_transport(x,
                                              velocity,
                                              -step_size(i) * velocity)
        return new_x, velocity

    def get_params(x):
        return x

    return init, update, get_params


