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
│   │   truncation.py
│   │   renyi_2.py
│   │   renyi_half.py
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

- The disentangler and mosesmove part should be completed with tensornetwork library and should be backend independent. (Check whether node split can avoid backend-dependent SVD)


#### Target project
- isoTNS with mosesmove
- isoTNS with GD
- applications with quantum circuits





