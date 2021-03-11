
Example Usage:
```
opt = optimizers.sgd(learning_rate)
opt_state = opt.init(params)

def step(step, opt_state):
  value, grads = jax.value_and_grad(loss_fn)(opt.get_params(opt_state))
  opt_state = opt.update(step, grads, opt_state)
  return value, opt_state

for step in range(num_steps):
  value, opt_state = step(step, opt_state)
```


where the optimizer is defined by,
```
@optimizer
def sgd(step_size):
  """Construct optimizer triple for stochastic gradient descent.

  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to positive scalar.

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  step_size = make_schedule(step_size)
  def init(x0):
    return x0
  def update(i, g, x):
    return x - step_size(i) * g
  def get_params(x):
    return x
  return Optimizer(init, update, get_params)

```


