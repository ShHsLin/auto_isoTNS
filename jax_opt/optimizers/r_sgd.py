import jax
jax.config.update('jax_enable_x64', True)

#from jax.experimental.optimizers import optimizer, make_schedule
from jax.example_libraries.optimizers import optimizer, make_schedule

@optimizer
def r_sgd(step_size, manifold):
    """Construct optimizer triple for stochastic gradient descent.
    Args:
        step_size: positive scalar, or a callable representing a step size schedule
        that maps the iteration index to positive scalar.

    Returns:
        An (init_fun, update_fun, get_params) triple.
    """
    step_size = optax.constant_schedule(step_size)
    def init(x0):
        return x0

    def update(i, grad, x):
        rgrad = manifold.egrad_to_rgrad(x, grad)
        new_x = manifold.retraction(x, -step_size(i) * rgrad)
        return new_x

    def get_params(x):
        return x

    return init, update, get_params


