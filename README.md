# auto-isoTNS


```
project
│   README.md
│
└───tests
│
└───tf_opt
│   │
│   └───manifolds
│   │   │   base_manifold.py
│   │   │   stiefel.py
│   │   │   ...
│   │  
│   └───optimizers
│       │   SGD.py
│       │   Adam.py
│       │   CG.py
│       │   ...
│   
└───jax_opt
│   
└───pytorch_opt
│   
└───disentangler
│   │   tf_disentangling.py
│   │   jax_disentangling.py
│   │   pytorch_disentangling.py
│
└───mosesmove
    │   tri_splitter.py
    │   greedy.py
    │   als.py
    │   gd.py

 
```


#### Idea
- We separate out the implementations for optimizer classes, which are backend dependent.

- The optimizer classes implement the Riemannian optimization on manifold choosen. We focus particularly on Stiefel manifold which is manifold of isometry. For standard optimizer, one can directly call from tf,jax,... instead.

- The disentangler part is harder to implement in backend independent fashion and also is not necessary. I plan to have disentangler: tensor --> tensor. So basically, having disentangler implemented in all different backends.

- The tri-splitter and mosesmove part should be completed within the tensornetwork library and should be backend independent, i.e. all function be: Node --> Node. (gd.py probbly not)


#### Target project
- isoTNS with mosesmove
- isoTNS with GD
- applications with quantum circuits



#### Problems:
- operation missing in implementing disentangling and direct reference from backend require.
  For example, to get the diagonal of a matrix, the norm of a tensor, ...

#### TODO:
- implemented jax disentangler
- implemented jax optimizer, following the structure like tf optimizer. But of course this has to be in jax update rule.
- setup test for optimizer and disentangler
