'''
This file contains functions related to circuits.
This file works with jax.numpy to realize auto-differentiation.

The gate in this code has the convention (0,1,2,3) which corresponds to

   2      3
   |      |
   |      |
______________
|            |
|            |
______________
   |      |
   |      |
   0      1

  |0>    |0>


In terms of matrix vector notation, U(i,j, i', j') |state(i,j)>, which is a bit annoying.
should chagne in the clean version of the code.

'''
from scipy import integrate
from scipy.linalg import expm
import misc, os
import sys
import mps_func

## 
## We use jax.numpy if possible
## or autograd.numpy
##
## Regarding wether choosing to use autograd or jax
## see https://github.com/google/jax/issues/193
## Basically, if we do not use gpu or tpu and work with 
## small problem size and do not want to spend time fixing
## function with @jit, then autograd should be fine.

import jax.numpy as np
import numpy as onp


def polar(A):
    '''
    return:
        polar decomposition of A
    '''
    d,chi1,chi2 = A.shape
    Y,D,Z = misc.svd(A.reshape(d*chi1,chi2), full_matrices=False)
    return np.dot(Y,Z).reshape([d,chi1,chi2])

def random_2site_U(d, factor=1e-2):
    try:
        A = factor * (np.random.uniform(size=[d**2, d**2]) +
                      1j * np.random.uniform(size=[d**2, d**2]))

    except:
        A = factor * (onp.random.uniform(size=[d**2, d**2]) +
                      1j * onp.random.uniform(size=[d**2, d**2]))

    A = A-A.T.conj()
    U = (np.eye(d**2)-A).dot(np.linalg.inv(np.eye(d**2)+A))
    return U
    # return U.reshape([d] * 4)

def construct_product_state(A_list):
    iter_state = A_list[0]
    for i in range(1, L):
        iter_state = np.kron(iter_state, A_list[i])

    return iter_state


def circuit_2_state(circuit, product_state):
    '''
    Input:
        circuit is a list of list of U, i.e.
        [[U0(t0), U1(t0), U2(t0), ...], [U0(t1), U1(t1), ...], ...]
        circuit[0] corresponds to layer-0,
        circuit[1] corresponds to layer-1, and so on.

    Goal:
        We compute the exact representaion of U(cirucit)|product_state>

    return:
        the final state
    '''
    depth = len(circuit)
    L = len(circuit[0]) + 1
    iter_state = product_state

    for dep_idx in range(depth):
        U_list = circuit[dep_idx]
        iter_state = apply_U_all_exact(iter_state, U_list)

    return iter_state

def var_circuit_exact(target_state, iter_state, circuit, product_state,
                      brickwall=False, verbose=False,
                     ):
    '''
    Goal:
        Do a sweep from top of the circuit down to product state,
        and do a sweep from bottom of the circuit to top.

        When sweeping from top to bottom, we take the target state (top) and
        circuit state (bottom), we remove one unitary at a time by
        applying the conjugate unitary.
        We then form the environment. Once a new unitary is obtained, it is
        applied to the "bra", i.e. top state.

        When sweeping from bottom to top, we take the product state, and the
        top state resulting from the first part of the algorithm.
        We remove one unitary at a time from the top state by applying the
        conjugate unitary. We then form the environment. Once a new unitary
        is obtained, it is applied to the bottom state.

        In this approach, we do not store the intermediate state representation.
        We avoid it by the removing unitary approach.
        There is no disadvantage of non exact cancellation of unitary as in
        MPS simulatino, since all the computation here is numerical exact.

    Input:
        target_state
        iter_state:  The state U(circuit)|product state>
        circuit: list of list of unitary
        product_state: the initial product state that the circuit acts on.
        breakwall: whether using breakwall type of circuit
    Output:
        iter_state: the state U(circuit)|product state> of the updated circuit
        circuit: list of list of unitary
    '''
    current_depth = len(circuit)
    L = len(circuit[0]) + 1
    top_state = target_state
    bottom_state = iter_state

    if verbose:
        print("Sweeping from top to bottom, overlap (before) : ",
              overlap_exact(target_state, iter_state))

    for var_dep_idx in range(current_depth-1, -1, -1):
        for idx in range(L - 2, -1, -1):
            remove_gate = circuit[var_dep_idx][idx]
            remove_gate_conj = remove_gate.reshape([4, 4]).T.conj()
            remove_gate_conj = remove_gate_conj.reshape([2, 2, 2, 2])
            bottom_state = apply_gate_exact(bottom_state, remove_gate_conj, idx)
            # now bottom_state is state without remove_gate,
            # we can now variational finding the optimal gate to replace it.

            if brickwall and (var_dep_idx + idx) % 2 == 1:
                new_gate = np.eye(4).reshape([2, 2, 2, 2])
            else:
                new_gate = var_gate_exact(top_state, idx, bottom_state)
                # new_gate, Lp_cache, Rp_cache = var_gate_w_cache(top_mps, idx, bottom_mps, Lp_cache, Rp_cache)
                circuit[var_dep_idx][idx] = new_gate

            # conjugate the gate
            # <psi|U = (U^\dagger |psi>)^\dagger
            new_gate_conj = new_gate.reshape([4, 4]).T.conj()
            new_gate_conj = new_gate_conj.reshape([2, 2, 2, 2])
            # new_gate_conj = np.einsum('ijkl->klij', new_gate).conj()

            top_state = apply_gate_exact(top_state, new_gate_conj, idx)

    overlap_abs = np.abs(overlap_exact(bottom_state, product_state))
    # print("overlap_abs = ", overlap_abs)
    assert np.isclose(overlap_abs, 1, rtol=1e-8)
    bottom_state = product_state

    if verbose:
        print("Sweeping from bottom to top, overlap (before) : ",
              overlap_exact(top_state, bottom_state)
             )
    for var_dep_idx in range(0, current_depth):
        for idx in range(L-1):
            gate = circuit[var_dep_idx][idx]
            top_state = apply_gate_exact(top_state, gate, idx)
            ## This remove the gate from top_state
            ## Because <\phi | U_{ij} | \psi> = inner( U_{ij}^\dagger |\phi>, |\psi> ) 
            ## applying U would remove the U_{ij}^\dagger, and
            ## partial_Tr[ |\phi>, |\psi> ] = Env

            if brickwall and (var_dep_idx + idx) % 2 == 1:
                new_gate = np.eye(4).reshape([2, 2, 2, 2])
            else:
                new_gate = var_gate_exact(top_state, idx, bottom_state)

            circuit[var_dep_idx][idx] = new_gate

            bottom_state = apply_gate_exact(bottom_state, new_gate, idx)

    ## finish sweeping
    return bottom_state, circuit

def var_gate_exact(top_state, site, bottom_state):
    '''
    Goal:
        to find argmax_{gate} <top_state | gate | down_state>
        where gate is actting on (site, site+1)
    Input:
        top_state: (did not have conjugation yet!!!)
        site: gate is applying on (site, site+1)
        bottom_state
    Return:
        new_gate
    '''
    total_dim = top_state.size
    L = int(np.log2(total_dim))
    top_theta = np.reshape(top_state, [(2**site), 4, 2**(L-(site+2))])
    bottom_theta = np.reshape(bottom_state, [(2**site), 4, 2**(L-(site+2))])

    M = np.tensordot(top_theta.conj(), bottom_theta, axes=([0, 2], [0, 2]))  # [ ..., upper_p, ...], [..., lower_p, ...] --> upper_p, lower_p
    M = M.T  # the convention is lower_p, upper_p

    ### For detailed explanation of the formula, see function var_gate
    U, _, Vd = misc.svd(M, full_matrices=False)
    new_gate = np.dot(U, Vd).conj()
    new_gate = new_gate.reshape([2, 2, 2, 2])

    return new_gate

def apply_gate_exact(state, gate, idx):
    '''
    Goal:
        Apply gate on the state vector
        assuming local dimension d=2
    Input:
        state: a vector
        gate: the gate to apply
        idx: the gate is applying on (idx, idx+1)

    Return:
        state
    '''
    total_dim = state.size
    L = np.log2(total_dim).astype(int)
    theta = np.reshape(state, [(2**idx), 4, 2**(L-(idx+2))])
    theta = np.tensordot(gate, theta, [1, 1])  ## [ij] [..., j, ...] --> [i, ..., ...]
    state = (np.transpose(theta, [1, 0, 2])).flatten()
    return state

def overlap_exact(psi1, psi2):
    '''
    take in two state and return the overlap <psi1 | psi2>
    psi1 is not taken complex conjugate beforehand.
    '''
    return np.dot(np.conjugate(psi1), psi2)

def apply_U_all_exact(exact_state, U_list, cache=False):
    #[TODO] Write also applying gates backward (L-2, L-1), (L-3, L-2), ...?
    '''
    Goal:
        apply a list of two site gates in U_list according to the order to sites
        [(0, 1), (1, 2), (2, 3), ... ]
        the computation is exact
    Input:
        Exact state

        U_list: a list of two site unitary gates
        cache: indicate whether the intermediate state should be store.
    Output:
        if cache is True, we will return a list of exact states,
        which gives the list of mps of length L, which corresponds to
        applying 0, 1, 2, ... L-1 gates.
    '''
    L = len(U_list) + 1
    if cache:
        list_states = []
        list_states.append(exact_state.copy())

    for site_i in range(L-1):
        gate = U_list[site_i]
        # apply gate on (site_i, site_i+1)
        exact_state = apply_gate_exact(exact_state, gate, site_i)

        if cache:
            list_states.append(exact_state.copy())
        else:
            pass

    if cache:
        return list_states
    else:
        return exact_state


