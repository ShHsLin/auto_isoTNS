
### Requirement
numpy, jax


Exact contraction + Polar decomposition
```
python run_optimization_exact.py --N_iter 1000 --depth 4
```

Exact contraction + Riemannian gradient descent
```
python run_optimization_exact_jax.py --N_iter 1000 --depth 4
```



Detail of possible flags:
```
Quantum Circuit Simulation

optional arguments:
  -h, --help            show this help message and exit
  --L L                 system size. Default: 10
  --depth DEPTH         depth of the circuitDefault: 2
  --N_iter N_ITER       (maximum) number of iteration in the
                        optimizationDefault: 1
  --brickwall BRICKWALL
                        Whether or not using brickwalltype in 1 for true, 0
                        for falseDefault: False
```
