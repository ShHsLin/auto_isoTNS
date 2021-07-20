'''
This file contains functions related to circuits.

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


- - -

Currently in the process of changing the notation to

   0      1
   |      |
   |      |
______________
|            |
|            |
______________
   |      |
   |      |
   2      3

  |0>    |0>


U(i,j, i', j') |state(i', j')>

function modified:
    apply_gate_exact
    apply_gate

[TODO] modify:
    the variational functions.
    var_circuit_exact : we can copy the change from circuit_func_jax.
    var_circuit


'''
from scipy import integrate
from scipy.linalg import expm
import misc, os
import sys
import mps_func

## We use jax.numpy if possible
## or autograd.numpy
##
## Regarding wether choosing to use autograd or jax
## see https://github.com/google/jax/issues/193

# import autograd.numpy as np
# from autograd import grad
# import jax.numpy as np
# from jax import random
# from jax import grad, jit, vmap
# from jax.config import config
# config.update("jax_enable_x64", True)

import numpy as np
np.seterr(all='raise')
# print("some function may be broken")

import numpy as onp

'''
current file is used purely numpy
for jax.numpy see circuit_func_jax.py
'''



def get_H(Hamiltonian, L, J, g, h=0, change_basis=False):
    if Hamiltonian == 'TFI':
        return get_H_TFI(L, J, g, h, change_basis=change_basis)
    elif Hamiltonian == 'XXZ':
        return get_H_XXZ(L, J, g)
    else:
        raise

def get_H_TFI(L, J, g, h=0, change_basis=False):
    '''
    Originally we have:
        H_TFI = - J XX - g Z - h X

    if change_basis:
        H_TFI = - J ZZ + g X + h Z

    Return:
        H: list of local hamiltonian
    '''
    sx = np.array([[0, 1], [1, 0]])
    sz = np.array([[1, 0], [0, -1]])
    eye2 = np.eye(2)
    d = 2

    if not change_basis:
        def hamiltonian(gl, gr, hl, hr, J):
            return (-np.kron(sx, sx) * J - gr * np.kron(eye2, sz) - gl * np.kron(sz, eye2)
                    -hr * np.kron(eye2, sx) - hl * np.kron(sx, eye2)
                   ).reshape([d] * 4)
    else:
        def hamiltonian(gl, gr, hl, hr, J):
            return (-np.kron(sz, sz) * J + gr * np.kron(eye2, sx) + gl * np.kron(sx, eye2)
                    +hr * np.kron(eye2, sz) + hl * np.kron(sz, eye2)
                   ).reshape([d] * 4)

    H = []
    for j in range(L - 1):
        if j == 0:
            gl = g
            hl = h
        else:
            gl = 0.5 * g
            hl = 0.5 * h
        if j == L - 2:
            gr = 1. * g
            hr = 1. * h
        else:
            gr = 0.5 * g
            hr = 0.5 * h

        H.append(hamiltonian(gl, gr, hl, hr, J))
    return H

def get_H_XXZ(L, J, g):
    '''
    H_XXZX = J XX + J YY + Jg ZZ
    '''
    sx = np.array([[0, 1], [1, 0]])
    sy = np.array([[0, -1.j], [1.j, 0]])
    sz = np.array([[1, 0], [0, -1]])
    eye2 = np.eye(2)
    d = 2

    def hamiltonian(g, J):
        return (np.kron(sx, sx) * J + np.kron(sy, sy) * J + np.kron(sz, sz) * J * g ).reshape([d] * 4)

    H = []
    for j in range(L - 1):
        H.append(hamiltonian(g, J).real)

    return H

def make_U(H, t):
    """ U = exp(-t H) """
    d = H[0].shape[0]
    return [expm(-t * h.reshape((d**2, -1))).reshape([d] * 4) for h in H]

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
    return U.reshape([d] * 4)
    # M = onp.random.rand(d ** 2, d ** 2)
    # Q, _ = np.linalg.qr(0.5 - M)
    # return Q.reshape([d] * 4)

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
    # A_list = [np.array([1., 0.]).reshape([2, 1, 1]) for i in range(L)]
    # A_list = [t.copy() for t in product_state]
    if type(product_state) == list:
        iter_state = product_state[0]
        for i in range(1, L):
            iter_state = np.kron(iter_state, product_state[i])
    else:
        iter_state = product_state

    for dep_idx in range(depth):
        U_list = circuit[dep_idx]
        iter_state = apply_U_all_exact(iter_state, U_list, cache=False)

    return iter_state

def circuit_2_mps(circuit, product_state, chi=None):
    '''
    Input:
        circuit is a list of list of U, i.e.
        [[U0(t0), U1(t0), U2(t0), ...], [U0(t1), U1(t1), ...], ...]
        circuit[0] corresponds to layer-0,
        circuit[1] corresponds to layer-1, and so on.

    Goal:
        We compute the mps representaion of each layer
        without truncation.
        The mps is in canonical form.
        We do not make any truncation, that
        should be fine.

        [ToDo] : add a truncation according to chi?

    return:
        list of mps representation of each layer; each in left canonical form
        mps_of_layer[0] gives the product state |psi0>
        mps_of_layer[1] gives the U(0) |psi0>
    '''
    depth = len(circuit)
    L = len(circuit[0]) + 1
    # A_list = [np.array([1., 0.]).reshape([2, 1, 1]) for i in range(L)]
    A_list = [t.copy() for t in product_state]

    mps_of_layer = []
    mps_of_layer.append([A.copy() for A in A_list])
    # A_list is modified inplace always.
    # so we always have to copy A tensors.
    for dep_idx in range(depth):
        U_list = circuit[dep_idx]
        ### A_list is modified inplace
        mps_func.right_canonicalize(A_list)
        A_list, trunc_error = apply_U_all(A_list, U_list, cache=False)
        mps_of_layer.append([A.copy() for A in A_list])

    return mps_of_layer

def circuit_2_mpo(circuit, mpo, chi=None):
    '''
    Input:
        circuit is a list of list of U, i.e.
        [[U0(t0), U1(t0), U2(t0), ...], [U0(t1), U1(t1), ...], ...]
        circuit[0] corresponds to layer-0,
        circuit[1] corresponds to layer-1, and so on.

        mpo: is the operator with [phys_up, left right, phys_down],
        which we denote as [p, l, r, q].

    Goal:
        We compute the mpo representaion of each layer
        without truncation.
        The mpo is in canonical form and treated by standard mps algorithm.
        We do not make any truncation, that should be fine.

        [ToDo] : add a truncation according to chi?

    return:
        list of mpo representation of each layer; each in left canonical form
        mpo_of_layer[0] gives the original mpo \hat{O}
        mpo_of_layer[1] gives the   U(0) \hat{O}
    '''
    depth = len(circuit)
    L = len(circuit[0]) + 1
    # A_list = [np.array([1., 0.]).reshape([2, 1, 1]) for i in range(L)]
    A_list = [t.copy() for t in mpo]

    mpo_of_layer = []
    # A_list is modified inplace always.
    # so we always have to copy A tensors.
    for dep_idx in range(depth):
        U_list = circuit[dep_idx]
        ### A_list is modified inplace
        A_list = mps_func.plrq_2_plr(A_list)
        mps_func.right_canonicalize(A_list, normalized=False)
        A_list = mps_func.plr_2_plrq(A_list)
        mpo_of_layer.append([A.copy() for A in A_list])

        A_list, trunc_error = apply_U_all_mpo(A_list, U_list, cache=False,
                                              normalized=False
                                             )

    mpo_of_layer.append([A.copy() for A in A_list])

    return mpo_of_layer

def var_circuit(target_mps, bottom_mps, circuit, product_state,
                brickwall=False
               ):
    '''
    Goal:
        Do a sweep from top of the circuit down to product state,
        and do a sweep from bottom of the circuit to top.

        When sweeping from top to bottom, we take the target and bottom mps,
        we remove one unitary at a time by applying the conjugate unitary.
        We then form the environment. Once a new unitary is obtained, it is
        applied to the "bra", i.e. top mps.

        When sweeping from bottom to top, we take the product state, and the
        top mps resulting from the first part of the algorithm.
        We remove one unitary at a time from the top mps by applying the
        conjugate unitary. We then form the environment. Once a new unitary
        is obtained, it is applied to the bottom mps.

        In this approach, we do not need to store the intermediate mps representation.
        We avoid it by the removing unitary approach.
        This however, has a disadvantage that the unitary is not cancelled completely.
        For example the bottom mps after sweeping from top to bottom usually is not a
        product state, but bond dimension 2.
        The top mps would have bond dimension that is larger than necessary.

        The other approach using caching mps and avoid unitary removal is implemented
        in var_circuit_2 function.

    Input:
        target_mps: can be not normalized, but should be in left canonical form.
        bottom_mps: the mps representing the contraction of full circuit
            with product state
        circuit: list of list of unitary
        product_state: the initial product state that the circuit acts on.
        breakwall: whether using breakwall type of circuit
    Output:
        mps_final: the mps representation of the updated circuit
        circuit: list of list of unitary
    '''
    current_depth = len(circuit)
    L = len(circuit[0]) + 1
    top_mps = [A.copy() for A in target_mps]

    print("Sweeping from top to bottom, overlap (before) : ",
          mps_func.overlap(target_mps, bottom_mps))
    for var_dep_idx in range(current_depth-1, -1, -1):
        Lp_cache = [np.ones([1, 1])] + [None] * (L-1)
        Rp_cache = [None] * (L-1) + [np.ones([1, 1])]
        for idx in range(L - 2, -1, -1):
            remove_gate = circuit[var_dep_idx][idx]
            remove_gate_conj = remove_gate.reshape([4, 4]).T.conj()
            remove_gate_conj = remove_gate_conj.reshape([2, 2, 2, 2])
            apply_gate(bottom_mps, remove_gate_conj, idx, move='left')
            # now bottom_mps is mps without remove_gate,
            # we can now variational finding the optimal gate to replace it.

            if brickwall and (var_dep_idx + idx) % 2 == 1:
                new_gate = np.eye(4).reshape([2, 2, 2, 2])
            else:
                new_gate, Lp_cache, Rp_cache = var_gate_w_cache(top_mps, idx, bottom_mps, Lp_cache, Rp_cache)
                circuit[var_dep_idx][idx] = new_gate

            # conjugate the gate
            # <psi|U = (U^\dagger |psi>)^\dagger
            new_gate_conj = new_gate.reshape([4, 4]).T.conj()
            new_gate_conj = new_gate_conj.reshape([2, 2, 2, 2])
            # new_gate_conj = np.einsum('ijkl->klij', new_gate).conj()

            apply_gate(top_mps, new_gate_conj, idx, move='left')


        mps_func.left_canonicalize(top_mps)
        mps_func.left_canonicalize(bottom_mps)

    max_chi_bot = np.amax([np.amax(t.shape) for t in bottom_mps])
    max_chi_top = np.amax([np.amax(t.shape) for t in top_mps])
    print("after sweep down, X(top_mps) = ", max_chi_top, " X(bot_mps) = ", max_chi_bot)


    overlap_abs = np.abs(mps_func.overlap(bottom_mps, product_state))
    # print("overlap_abs = ", overlap_abs)
    assert np.isclose(overlap_abs, 1, rtol=1e-8)
    bottom_mps = [t.copy() for t in product_state]

    print("Sweeping from bottom to top, overlap (before) : ",
          mps_func.overlap(top_mps, bottom_mps)
         )
    for var_dep_idx in range(0, current_depth):
        mps_func.right_canonicalize(top_mps)
        mps_func.right_canonicalize(bottom_mps)

        Lp_cache = [np.ones([1, 1])] + [None] * (L-1)
        Rp_cache = [None] * (L-1) + [np.ones([1, 1])]
        for idx in range(L-1):
            gate = circuit[var_dep_idx][idx]
            apply_gate(top_mps, gate, idx, move='right')
            ## This remove the gate from top_mps
            ## Because <\phi | U_{ij} | \psi> = inner( U_{ij}^\dagger |\phi>, |\psi> ) 
            ## applying U would remove the U_{ij}^\dagger, and
            ## partial_Tr[ |\phi>, |\psi> ] = Env

            if brickwall and (var_dep_idx + idx) % 2 == 1:
                new_gate = np.eye(4).reshape([2, 2, 2, 2])
            else:
                new_gate, Lp_cache, Rp_cache = var_gate_w_cache(top_mps, idx, bottom_mps, Lp_cache, Rp_cache)

            circuit[var_dep_idx][idx] = new_gate

            apply_gate(bottom_mps, new_gate, idx, move='right')

    ## finish sweeping
    ## bottom_mps is mps_final

    max_chi_bot = np.amax([np.amax(t.shape) for t in bottom_mps])
    max_chi_top = np.amax([np.amax(t.shape) for t in top_mps])
    print("after sweep up, X(top_mps) = ", max_chi_top, " X(bot_mps) = ", max_chi_bot)
    return bottom_mps, circuit

def var_circuit_mpo(target_mpo, circuit,
                    brickwall=False
                   ):
    '''
    Goal:
        Do a full sweep down and a full sweep up, to find the set of {U_i} that gives
        arg max_{U_i} Re Tr [U1*U2*U3... target_MPO ]
        Notice that target_MPO is the conjugate of the approximated N-qubit operator,
        i.e. U1*U2*U3... ~= complex_conjugate(target_MPO)

        1.) We take the target_MPO and apply the circuit on top.
        2.) Sweeping from top to bottom by removing one unitary at a time.
            This is done by applying conjugated unitary.

            We then form the environment and get the new gate.
            The new gate obtained is applied to the lower part of MPO.

        3.) When sweeping from bottom to top, we take the mpo, and apply the full circuit
            to the lower part of the mpo.
            Again one unitary is removed at a time from below to get the new gate.
            New gate is obtained from the environment.
            New gate is applied to the upper part of MPO.

        In this approach, we do not need to store the intermediate mps representation.
        We avoid it by the removing unitary approach.
        This however, has a disadvantage that the unitary is not cancelled completely.

        Might need to extend the var_circuit2 to mpo version.

    Input:
        target_mpo:
            the conjugated N-qubit operator to be approximated.
            conjugation should be taken care of outside this function.
            can be not normalized

        circuit: list of list of unitary
        breakwall: whether using breakwall type of circuit
    Output:
        mpo_final: the mpo representation of the updated circuit
        circuit: list of list of unitary


    Note:
        apply_gate_mpo(A_list, gate, idx, pos, move, no_trunc=False, chi=None, normalized=False)
        var_gate_mpo_w_cache(mpo, site, Lp_cache, Rp_cache)
    '''
    total_trunc_error = 0.
    current_depth = len(circuit)
    L = len(circuit[0]) + 1
    # top_mps = [A.copy() for A in target_mps]

    mpo_of_layer = circuit_2_mpo(circuit, target_mpo)
    iter_mpo = mpo_of_layer[-1]

    # iter_mpo2 is a copy of target_mpo
    iter_mpo2 = [t.copy() for t in mpo_of_layer[0]]
    iter_mpo2 = mps_func.plrq_2_plr(iter_mpo2)
    mps_func.left_canonicalize(iter_mpo2, normalized=False)
    iter_mpo2 = mps_func.plr_2_plrq(iter_mpo2)


    # [TODO] add an function to check the cost function?
    # print("Sweeping from top to bottom, overlap (before) : ",
    #       mps_func.overlap(target_mps, bottom_mps))

    for var_dep_idx in range(current_depth-1, -1, -1):
        Lp_cache = [np.ones([1])] + [None] * (L-1)
        Rp_cache = [None] * (L-1) + [np.ones([1])]
        for idx in range(L - 2, -1, -1):
            remove_gate = circuit[var_dep_idx][idx]
            remove_gate_conj = remove_gate.reshape([4, 4]).T.conj()
            remove_gate_conj = remove_gate_conj.reshape([2, 2, 2, 2])
            trunc_error = apply_gate_mpo(iter_mpo, remove_gate_conj, idx, pos='up', move='left')
            total_trunc_error += trunc_error
            # now iter_mpo is mpo without remove_gate,
            # we can now variational finding the optimal gate to replace it.

            if brickwall and (var_dep_idx + idx) % 2 == 1:
                new_gate = np.eye(4).reshape([2, 2, 2, 2])
            else:
                new_gate, Lp_cache, Rp_cache = var_gate_mpo_w_cache(iter_mpo, idx, Lp_cache, Rp_cache)
                circuit[var_dep_idx][idx] = new_gate

            trunc_error = apply_gate_mpo(iter_mpo, new_gate, idx, pos='down', move='left')
            total_trunc_error += trunc_error
            trunc_error = apply_gate_mpo(iter_mpo2, new_gate, idx, pos='down', move='left')


        iter_mpo = mps_func.plrq_2_plr(iter_mpo)
        mps_func.left_canonicalize(iter_mpo, normalized=False)
        iter_mpo = mps_func.plr_2_plrq(iter_mpo)

        iter_mpo2 = mps_func.plrq_2_plr(iter_mpo2)
        mps_func.left_canonicalize(iter_mpo2, normalized=False)
        iter_mpo2 = mps_func.plr_2_plrq(iter_mpo2)

    max_chi_mpo = np.amax([np.amax(t.shape) for t in iter_mpo])
    print("after sweep down, X(mpo) = ", max_chi_mpo)

    print("total truncation along the sweep : ", total_trunc_error)
    assert total_trunc_error < 1e-12
    total_trunc_error = 0.

    # [TODO] add an function to check the cost function?
    # overlap_abs = np.abs(mps_func.overlap(bottom_mps, product_state))
    # # print("overlap_abs = ", overlap_abs)
    # assert np.isclose(overlap_abs, 1, rtol=1e-8)
    # bottom_mps = [t.copy() for t in product_state]

    # print("Sweeping from bottom to top, overlap (before) : ",
    #       mps_func.overlap(top_mps, bottom_mps)
    #      )

    for var_dep_idx in range(0, current_depth):
        iter_mpo2 = mps_func.plrq_2_plr(iter_mpo2)
        mps_func.right_canonicalize(iter_mpo2, normalized=False)
        iter_mpo2 = mps_func.plr_2_plrq(iter_mpo2)

        Lp_cache = [np.ones([1])] + [None] * (L-1)
        Rp_cache = [None] * (L-1) + [np.ones([1])]
        for idx in range(L-1):
            remove_gate = circuit[var_dep_idx][idx]
            remove_gate_conj = remove_gate.reshape([4, 4]).T.conj()
            remove_gate_conj = remove_gate_conj.reshape([2, 2, 2, 2])
            trunc_error = apply_gate_mpo(iter_mpo2, remove_gate_conj, idx, pos='down', move='right')
            total_trunc_error += trunc_error

            if brickwall and (var_dep_idx + idx) % 2 == 1:
                new_gate = np.eye(4).reshape([2, 2, 2, 2])
            else:
                new_gate, Lp_cache, Rp_cache = var_gate_mpo_w_cache(iter_mpo2, idx, Lp_cache, Rp_cache)
                circuit[var_dep_idx][idx] = new_gate

            circuit[var_dep_idx][idx] = new_gate

            trunc_error = apply_gate_mpo(iter_mpo2, new_gate, idx, pos='up', move='right')
            total_trunc_error += trunc_error

    ## finish sweeping

    max_chi_mpo = np.amax([np.amax(t.shape) for t in iter_mpo2])
    print("after sweep down, X(mpo) = ", max_chi_mpo)
    print("total truncation along the sweep : ", total_trunc_error)
    assert total_trunc_error < 1e-12
    total_trunc_error = 0.

    return iter_mpo2, circuit

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

def var_circuit2(target_mps, product_state, circuit, brickwall=False):
    #[TODO] extend this to brickwall
    """
    Goal:
        Do a sweep updating gates from top of the circuit down to product state,
        and do a sweep updating gates from bottom of the circuit to top.

    Input:
        target_mps: can be not normalized, but should be in left canonical form.
        product_state
        circuit

    Output:
        new_circuit
    """
    L = len(target_mps)
    circuit_depth = len(circuit)

    top_mps = [t.copy() for t in target_mps]  # in left canonical form
    bottom_mps_cache = circuit_2_mps(circuit, product_state) # in left canonical form
    # All mps in bottom_mps_cache is in left canonical form

    #[TODO] Is this necessary?
    # When append to top_mps_cache, we always need to use copy
    # such that the cache is really an independent list.
    # (The other way may be to create a new list everytime in var_layer and return it.)
    top_mps_cache = []
    top_mps_cache.append([t.copy() for t in top_mps])

    for dep_idx in range(circuit_depth-1, -1, -1):
        top_mps, new_layer = var_layer(top_mps,
                                       circuit[dep_idx],
                                       bottom_mps_cache[dep_idx],
                                       direction='down',
                                       brickwall=brickwall,
                                       dep_idx=dep_idx,
                                      )
        assert(len(new_layer) == L-1)
        circuit[dep_idx] = new_layer
        top_mps_cache.append([t.copy() for t in top_mps])

    assert( len(top_mps_cache)  == circuit_depth + 1)
    # top_mps_cache [0] --> target_mps
    # top_mps_cache [1] --> target_mps + 1-layer
    # top_mps_cache [x] --> target_mps + x-layer
    bottom_mps = bottom_mps_cache[0]
    for dep_idx in range(0, circuit_depth):
        bottom_mps, new_layer = var_layer(top_mps_cache[-2-dep_idx],
                                          circuit[dep_idx],
                                          bottom_mps,
                                          direction='up',
                                          brickwall=brickwall,
                                          dep_idx=dep_idx,
                                         )
        circuit[dep_idx] = new_layer

    return bottom_mps, circuit

def var_layer(top_mps, layer_gate, bottom_mps, direction,
              brickwall=False, dep_idx=None):
    '''
    Goal:
        See graphical representation below
    Input:
        top_mps: in left canonical form
        layer_gate:
        bottom_mps: in left canonical form
    Output:
        top_mps, new_layer


    We try to maximize the quantity:

    <top_mps | layer_gate | bottom_mps>

    =

    |    |    |    |
    -------------------- layer 0

    -------------------- layer 1
    .
    .
    -------------------- layer n

    -------------------- Imaginary time evolution

    -------------------- layer n
    .
    .
    -------------------- layer x
    .
    .
    -------------------- layer 1

    -------------------- layer 0
    |    |    |    |

    =

    -------------------- mps-representation of all layer-(n) circuit
    |    |    |    |

    -------------------- Imaginary time evolution
    .
    .
    -------------------- layer x+1

    -------------------- layer x

    |____|____|____|____ mps-representation of all layer-(x-1) circuit

    =

    ______________ top_mps
    |   |   |
    ______________ layer x
    |   |   |
    ______________ bottom_mps

    '''
    L = len(layer_gate) + 1

    Lp_cache = [np.ones([1, 1])] + [None] * (L-1)
    Rp_cache = [None] * (L-1) + [np.ones([1, 1])]

    if direction == 'down':
        # Form upward cache

        # we copy the tensor in bottom_mps, so that bottom_mps is not modified.
        # [Notice] apply_U_all function modified the A_list inplace.
        # [Todo] maybe change the behavior above. not inplace?
        A_list, trunc_err = mps_func.right_canonicalize([t.copy() for t in bottom_mps])
        upward_cache_list, trunc_error = apply_U_all(A_list,
                                                     layer_gate,
                                                     cache=True)

        assert(len(upward_cache_list) == L)
        # There are L states, because with L-1 gates, including not applying gate
        # idx=0 not applying gate,
        # idx=1, state after applying gate-0 on site-0, site-1.
        # idx=2, state after applying gate-1 on site-1, site-2.
        new_layer = [None] * (L-1)

        for idx in range(L - 2, -1, -1):
            mps_cache = upward_cache_list[idx]
            if brickwall and (dep_idx + idx) % 2 == 1:
                new_gate = np.eye(4).reshape([2,2,2,2])
                new_layer[idx] = new_gate
            else:
                new_gate, Lp_cache, Rp_cache = var_gate_w_cache(top_mps, idx, mps_cache, Lp_cache, Rp_cache)
                # new_gate = var_gate(top_mps, idx, mps_cache)
                new_layer[idx] = new_gate

            # conjugate the gate
            # <psi|U = (U^\dagger |psi>)^\dagger
            new_gate_conj = new_gate.reshape([4, 4]).T.conj()
            new_gate_conj = new_gate_conj.reshape([2, 2, 2, 2])
            # new_gate_conj = np.einsum('ijkl->klij', new_gate).conj()

            apply_gate(top_mps, new_gate_conj, idx, move='left')

        # top_mps ends up in right canonical form
        top_mps, trunc_err = mps_func.left_canonicalize(top_mps)
        # top_mps brought back to left canonical form
        assert trunc_err < 1e-12

        return top_mps, new_layer
    elif direction == 'up':
        # Form downward cache
        downward_cache_list = []
        downward_cache_list.append([t.copy() for t in top_mps])
        for idx in range(L-2, -1, -1):
            conj_gate = layer_gate[idx].reshape([4,4]).T.conj()
            conj_gate = conj_gate.reshape([2, 2, 2, 2])
            apply_gate(top_mps, conj_gate, idx, move='left')
            downward_cache_list.append([t.copy() for t in top_mps])

        assert(len(downward_cache_list) == L)
        new_layer = [None] * (L-1)

        bottom_mps, trunc_err = mps_func.right_canonicalize([t.copy() for t in bottom_mps])
        for idx in range(L-1):
            top_mps = downward_cache_list[-2-idx]
            if brickwall and (dep_idx + idx) % 2 == 1:
                new_gate = np.eye(4).reshape([2,2,2,2])
                new_layer[idx] = new_gate
            else:
                new_gate, Lp_cache, Rp_cache = var_gate_w_cache(top_mps, idx, bottom_mps, Lp_cache, Rp_cache)
                # new_gate = var_gate(top_mps, idx, bottom_mps)
                new_layer[idx] = new_gate

            apply_gate(bottom_mps, new_gate, idx, move='right')

        return bottom_mps, new_layer

    else:
        raise NotImplementedError

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

def var_gate_w_cache(new_mps, site, mps_ket, Lp_cache, Rp_cache):
    '''
    Goal:
        to find argmax_{gate} <new_mps | gate | mps_ket>
        where gate is actting on (site, site+1)
    Input:
        new_mps: list of tensors of the new_mps, convention [p, l, r]
        site: gate is applying on (site, site+1)
        mps_ket: list of tensors of the mps_ket, convention [p, l, r]
        Lp_cache: matrix
        Rp_cache: matrix
    Return:
        new_gate
    '''
    L = len(new_mps)
    # Lp = np.ones([1, 1])
    # Lp_list = [Lp]
    # Lp_cache = [Lp, ... ]

    for i in range(site):
        Lp = Lp_cache[i]
        if Lp_cache[i+1] is None:
            Lp = np.tensordot(Lp, mps_ket[i], axes=(0, 1))
            Lp = np.tensordot(Lp, new_mps[i].conj(), axes=([0, 1], [1,0]))
            Lp_cache[i+1] = Lp
        else:
            pass

    # Rp = np.ones([1, 1])
    # Rp_list = [Rp]

    for i in range(L-1, site+1, -1):
        Rp = Rp_cache[i]
        if Rp_cache[i-1] is None:
            Rp = np.tensordot(mps_ket[i], Rp, axes=(2, 0))
            Rp = np.tensordot(Rp, new_mps[i].conj(), axes=([0, 2], [0, 2]))
            Rp_cache[i-1] = Rp

    L_env = Lp_cache[site]
    # L_env = Lp_list[site]
    R_env = Rp_cache[site+1]
    # R_env = Rp_list[L-2-site]

    theta_top = np.tensordot(new_mps[site].conj(), new_mps[site + 1].conj(), axes=(2,1)) # p l, q r
    theta_bot = np.tensordot(mps_ket[site], mps_ket[site + 1],axes=(2,1))

    M = np.tensordot(L_env, theta_bot, axes=([0], [1])) #l, p, q, r
    M = np.tensordot(M, R_env, axes=([3], [0])) #l, p, q, r
    M = np.tensordot(M, theta_top, axes=([0, 3], [1, 3])) #lower_p, lower_q, upper_p, upper_q
    M = M.reshape([4, 4])

    ### For detailed explanation of the formula, see function var_gate
    U, _, Vd = misc.svd(M, full_matrices=False)
    new_gate = np.dot(U, Vd).conj()
    new_gate = new_gate.reshape([2, 2, 2, 2])

    return new_gate, Lp_cache, Rp_cache

def var_gate_mpo_w_cache(mpo, site, Lp_cache, Rp_cache):
    '''
    Goal:
        to find argmax_{gate}  Tr[  gate \hat{O} ]
        where gate is actting on (site, site+1)
    Input:
        mpo: list of tensors of the mpo operator, convention [p, l, r, q]
        site: gate is applying on (site, site+1)
        Lp_cache: vector
        Rp_cache: vector
    Return:
        new_gate
    '''
    L = len(mpo)

    # Lp = np.ones([1])
    # Lp_list = [Lp, None, None, ...]

    for i in range(site):
        Lp = Lp_cache[i]
        if Lp_cache[i+1] is None:
            Lp = np.tensordot(Lp, mpo[i], axes=(0, 1))
            Lp = np.trace(Lp, axis1=0, axis2=2)
            Lp_cache[i+1] = Lp
        else:
            pass

    # Rp = np.ones([1])
    # Rp_list = [Rp, None, None, ...]

    for i in range(L-1, site+1, -1):
        Rp = Rp_cache[i]
        if Rp_cache[i-1] is None:
            Rp = np.tensordot(mpo[i], Rp, axes=(2, 0))
            Rp = np.trace(Rp, axis1=0, axis2=2)
            Rp_cache[i-1] = Rp

    L_env = Lp_cache[site]
    # L_env = Lp_list[site]
    R_env = Rp_cache[site+1]
    # R_env = Rp_list[L-2-site]

    left_site = np.tensordot(L_env, mpo[site], axes=(0, 1))  #[p1, r, q1]
    right_site = np.tensordot(mpo[site+1], R_env, axes=(2, 0))  #[p2, l, q2]
    M = np.tensordot(left_site, right_site, axes=(1, 1))  #[p1, q1, p2, q2]
    M = M.transpose([0, 2, 1, 3])  #lower_p1, lower_p2, upper_q1, upper_q2
    M = M.reshape([4, 4])

    ### For detailed explanation of the formula, see function var_gate
    U, _, Vd = misc.svd(M, full_matrices=False)
    new_gate = np.dot(U, Vd).conj()
    new_gate = new_gate.reshape([2, 2, 2, 2])

    return new_gate, Lp_cache, Rp_cache

def var_gate(new_mps, site, mps_ket):
    '''
    Input:
        new_mps : the mps representation of the bra
        site : two site gate is applied on (site, site+1)
        mps_ket : the mps representation of the ket
    Goal:
        max  Re<new_mps | gate | mps_ket>
        where gate is actting on (site, site+1)
    return:
        new_gate
    '''
    L = len(new_mps)
    Lp = np.ones([1, 1])
    Lp_list = [Lp]

    for i in range(L):
        Lp = np.tensordot(Lp, mps_ket[i], axes=(0, 1))
        Lp = np.tensordot(Lp, new_mps[i].conj(), axes=([0, 1], [1,0]))
        Lp_list.append(Lp)

    Rp = np.ones([1, 1])
    Rp_list = [Rp]

    for i in range(L-1, -1, -1):
        Rp = np.tensordot(mps_ket[i], Rp, axes=(2, 0))
        Rp = np.tensordot(Rp, new_mps[i].conj(), axes=([0, 2], [0, 2]))
        Rp_list.append(Rp)

    L_env = Lp_list[site]
    R_env = Rp_list[L-2-site]

    theta_top = np.tensordot(new_mps[site].conj(), new_mps[site + 1].conj(), axes=(2,1)) # p l, q r
    theta_bot = np.tensordot(mps_ket[site], mps_ket[site + 1],axes=(2,1))

    M = np.tensordot(L_env, theta_bot, axes=([0], [1])) #l, p, q, r
    M = np.tensordot(M, R_env, axes=([3], [0])) #l, p, q, r
    M = np.tensordot(M, theta_top, axes=([0, 3], [1, 3])) #lower_p, lower_q, upper_p, upper_q
    M = M.reshape([4, 4])

    ##### The formula should work for the first layer; where the unitary there has redundant
    ##### degree of freedom.
    # M_copy = M.reshape([2, 2, 2, 2]).copy()
    # M_copy = M_copy[:, 0, :, :]
    # U, _, Vd = misc.svd(M_copy.reshape([2, 4]), full_matrices=False)
    # new_gate = np.dot(U, Vd).reshape([2, 2, 2])
    # new_gate_ = onp.random.rand(2, 2, 2, 2) * (1+0j)
    # new_gate_[:, 0, :, :] = new_gate.conj()
    # # return new_gate_

    ######################################################################
    # new_gate = polar(M)
    # We are maximizing Re[\sum_ij A_ij W_ij ] with W^\dagger W = I
    # Re[\sum_ij A_ij W_ij] = Re Tr[ WA^T], A=USV^dagger, A^T = V*SU^T
    # W = (UV^\dagger)* = U* V^T   gives optimal results.
    # Re Tr[ WA^T] = Tr[S]
    ######################################################################
    U, _, Vd = misc.svd(M, full_matrices=False)
    new_gate = np.dot(U, Vd).conj()
    new_gate = new_gate.reshape([2, 2, 2, 2])

    return new_gate

def apply_gate(A_list, gate, idx, move, no_trunc=False, chi=None, normalized=False):
    '''
    [modification inplace]
    Goal:
        Apply gate on the MPS A_list
    Input:
        A_list: the list of tensors
        gate: the gate to apply
        idx: the gate is applying on (idx, idx+1)
        move: the direction to combine tensor after SVD
        no_trunc: if True, then even numerical zeros are not truncated
        chi: the truncation bond dimension provided.

    Return:
        trunc_error
    '''
    d1, chi1, chi2 = A_list[idx].shape
    d2, chi2, chi3 = A_list[idx + 1].shape

    theta = np.tensordot(A_list[idx], A_list[idx + 1],axes=(2,1))  # [d1, chi1, d2, chi3]
    theta = np.tensordot(gate, theta, axes=([2,3],[0,2]))  # [i',j',i,j] [i, D1, j, D2] -> [i',j',D1, D2]
    theta = np.reshape(np.transpose(theta,(0,2,1,3)),(d1*chi1, d2*chi3))  # [i',D1,j',D2]

    # [TODO] Remove the code below: old convention should be removed
    # theta = np.tensordot(gate, theta, axes=([0,1],[0,2]))  # [i,j,i',j'] [i, D1, j, D2] -> [i',j',D1, D2]
    # theta = np.reshape(np.transpose(theta,(0,2,1,3)),(d1*chi1, d2*chi3))  # [i',D1,j',D2]

    X, Y, Z = misc.svd(theta, full_matrices=0)

    if no_trunc:
        chi2 = np.size(Y)
    else:
        chi2 = np.sum(Y>1e-14)
        if chi is not None:
            chi2 = np.amin([chi2, chi])


    arg_sorted_idx = (np.argsort(Y)[::-1])[:chi2]
    trunc_idx = (np.argsort(Y)[::-1])[chi2:]
    trunc_error = np.sum(Y[trunc_idx] ** 2) / np.sum(Y**2)
    Y = Y[arg_sorted_idx]  # chi2
    if normalized:
        Y = Y / np.linalg.norm(Y)

    X = X[: ,arg_sorted_idx]  # (d1*chi1, chi2)
    Z = Z[arg_sorted_idx, :]  # (chi2, d2*chi3)

    if move == 'right':
        X=np.reshape(X, (d1, chi1, chi2))
        A_list[idx] = X.reshape([d1, chi1, chi2])
        A_list[idx + 1] = np.dot(np.diag(Y), Z).reshape([chi2, d2, chi3]).transpose([1, 0, 2])
    elif move == 'left':
        A_list[idx + 1] = np.transpose(Z.reshape([chi2, d2, chi3]), [1, 0, 2])
        A_list[idx] = np.dot(X, np.diag(Y)).reshape([d1, chi1, chi2])
    else:
        raise

    ## Sometimes both scipy and numpy seems to breakdown for SVD.
    ## Should due to the lapack or blas problem on workstation.

    # for t in A_list:
    #     try:
    #         assert np.isfinite(t).all()
    #     except:
    #         import pdb;pdb.set_trace()

    return trunc_error

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
    L = int(np.log2(total_dim))
    theta = np.reshape(state, [(2**idx), 4, 2**(L-(idx+2))])
    gate = np.reshape(gate, [4, 4])

    theta = np.tensordot(gate, theta, [1, 1])  ## [ij] [..., j, ...] --> [i, ..., ...]
    state = (np.transpose(theta, [1, 0, 2])).flatten()

    # [TODO] Remove the code below: old convention should be removed
    # theta = np.tensordot(theta, gate, [1, 0])  ## [..., j, ...] [ji] --> [..., ..., i]
    # state = (np.transpose(theta, [0, 2, 1])).flatten()
    return state

def apply_gate_mpo(A_list, gate, idx, pos, move, no_trunc=False, chi=None, normalized=False):
    '''
    [modification inplace]
    Goal:
        Apply gate on the MPO A_list
    Input:
        A_list: the list of tensors; convention [phys_up, left, right, phys_down]
        gate: the gate to apply (U[23,01]), [left_down, right_down, left_up, right_up]

        idx: the gate is applying on (idx, idx+1)
        pos: the position the gate is apply, need to be either up or down.
        move: the direction to combine tensor after SVD
        no_trunc: if True, then even numerical zeros are not truncated
        chi: the truncation bond dimension provided.

    Return:
        trunc_error
    '''
    assert pos in ['up', 'down']

    p1, chi1, chi2, q1 = A_list[idx].shape
    p2, chi2, chi3, q2 = A_list[idx + 1].shape

    theta = np.tensordot(A_list[idx], A_list[idx + 1],axes=(2,1))  #[p1, chi1, q1, p2, chi3, q2]
    if pos == 'up':
        theta = np.tensordot(gate, theta, axes=([0,1],[0,3]))  #[p3, p4, chi1, q1, chi3, q2]
        theta = np.reshape(np.transpose(theta,(0,2,3,1,4,5)), (p1*chi1*q1, p2*chi3*q2))
    else:
        theta = np.tensordot(gate, theta, axes=([2,3],[2,5]))  #[q3, q4, p1, chi1, p2, chi3]
        theta = np.reshape(np.transpose(theta,(2,3,0,4,5,1)), (p1*chi1*q1, p2*chi3*q2))

    X, Y, Z = misc.svd(theta, full_matrices=0)

    if no_trunc:
        chi2 = np.size(Y)
    else:
        chi2 = np.sum(Y>1e-14)
        if chi is not None:
            chi2 = np.amin([chi2, chi])


    arg_sorted_idx = (np.argsort(Y)[::-1])[:chi2]
    trunc_idx = (np.argsort(Y)[::-1])[chi2:]
    trunc_error = np.sum(Y[trunc_idx] ** 2) / np.sum(Y**2)
    Y = Y[arg_sorted_idx]  # chi2
    if normalized:
        Y = Y / np.linalg.norm(Y)

    X = X[: ,arg_sorted_idx]  # (p1*chi1*q1, chi2)
    Z = Z[arg_sorted_idx, :]  # (chi2, p2*chi3*q2)

    if move == 'right':
        A_list[idx] = np.reshape(X, (p1, chi1, q1, chi2)).transpose([0, 1, 3, 2])
        A_list[idx + 1] = np.dot(np.diag(Y), Z).reshape([chi2, p2, chi3, q2]).transpose([1, 0, 2, 3])
    elif move == 'left':
        A_list[idx + 1] = np.transpose(Z.reshape([chi2, p2, chi3, q2]), [1, 0, 2, 3])
        A_list[idx] = np.dot(X, np.diag(Y)).reshape([p1, chi1, q1, chi2]).transpose([0, 1, 3, 2])
    else:
        raise

    return trunc_error

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

def apply_U_all(A_list, U_list, cache=False, no_trunc=False, chi=None):
    # Make this function as an application of apply_gate only.
    #[TODO] Write also applying gates backward (L-2, L-1), (L-3, L-2), ...?
    '''
    Goal:
        apply a list of two site gates in U_list according to the order to sites
        [(0, 1), (1, 2), (2, 3), ... ]
    Input:
        A_list: the MPS representation to which the U_list apply
                To make truncation, A_list should be in right canonical form.
                If no truncation made, this does not matter.

        U_list: a list of two site unitary gates
        cache: indicate whether the intermediate state should be store.

        no_trunc: indicate whether truncation should take place
        chi: truncate to bond dimension chi, if chi is given. Must make sure no_trunc is True.
    Output:
        if cache is True, we will return a list_A_list,
        which gives the list of mps of length L, which corresponds to
        applying 0, 1, 2, ... L-1 gates.
    '''
    L = len(A_list)
    if cache:
        list_A_list = []
        list_A_list.append([A.copy() for A in A_list])

    tot_trunc_err = 0.
    for i in range(L-1):
        gate = U_list[i]
        trunc_error = apply_gate(A_list, gate, i, move='right', no_trunc=no_trunc,
                                 chi=chi, normalized=True)

        tot_trunc_err = tot_trunc_err + trunc_error

        if cache:
            list_A_list.append([A.copy() for A in A_list])
        else:
            pass

    if cache:
        return list_A_list, tot_trunc_err
    else:
        return A_list, tot_trunc_err

def apply_U_all_mpo(A_list, U_list, cache=False, no_trunc=False, chi=None, normalized=False):
    '''
    Goal:
        apply a list of two site gates in U_list according to the order to sites
        [(0, 1), (1, 2), (2, 3), ... ]
    Input:
        A_list: the list of tensors of MPO to which the U_list apply
                To make truncation, A_list should be in right canonical form.
                If no truncation made, this does not matter.

        U_list: a list of two site unitary gates
        cache: indicate whether the intermediate state should be store.

        no_trunc: indicate whether truncation should take place
        chi: truncate to bond dimension chi, if chi is given. Must make sure no_trunc is True.
    Output:
        if cache is True, we will return a list_A_list,
        which gives the list of mps of length L, which corresponds to
        applying 0, 1, 2, ... L-1 gates.
    '''
    L = len(A_list)
    if cache:
        list_A_list = []
        list_A_list.append([A.copy() for A in A_list])

    tot_trunc_err = 0.
    for i in range(L-1):
        gate = U_list[i]
        trunc_error = apply_gate_mpo(A_list, gate, i, pos='up', move='right',
                                     no_trunc=no_trunc,
                                     chi=chi, normalized=normalized)

        tot_trunc_err = tot_trunc_err + trunc_error

        if cache:
            list_A_list.append([A.copy() for A in A_list])
        else:
            pass

    if cache:
        return list_A_list, tot_trunc_err
    else:
        return A_list, tot_trunc_err

def apply_U(A_list, U_list, onset):
    '''
    There are two subset of gate.
    onset indicate whether we are applying even (0, 2, 4, ...)
    or odd (1, 3, 5, ...) gates
    '''
    L = len(A_list)

    Ap_list = [None for i in range(L)]
    if L % 2 == 0:
        if onset == 1:
            Ap_list[0] = A_list[0]
            Ap_list[L-1] = A_list[L-1]
        else:
            pass
    else:
        if onset == 0:
            Ap_list[L-1] = A_list[L-1]
        else:
            Ap_list[0] = A_list[0]


    bound = L-1
    for i in range(onset,bound,2):
        d1,chi1,chi2 = A_list[i].shape
        d2,chi2,chi3 = A_list[i+1].shape

        theta = np.tensordot(A_list[i],A_list[i+1],axes=(2,1))
        theta = np.tensordot(U_list[i],theta,axes=([0,1],[0,2]))
        theta = np.reshape(np.transpose(theta,(0,2,1,3)),(d1*chi1, d2*chi3))

        X, Y, Z = misc.svd(theta,full_matrices=0)
        chi2 = np.sum(Y>1e-14)

        # piv = np.zeros(len(Y), onp.bool)
        # piv[(np.argsort(Y)[::-1])[:chi2]] = True

        # Y = Y[piv]; invsq = np.sqrt(sum(Y**2))
        # X = X[:,piv]
        # Z = Z[piv,:]

        arg_sorted_idx = (np.argsort(Y)[::-1])[:chi2]
        Y = Y[arg_sorted_idx]
        X = X[: ,arg_sorted_idx]
        Z = Z[arg_sorted_idx, :]

        X=np.reshape(X, (d1, chi1, chi2))
        Ap_list[i]   = X.reshape([d1, chi1, chi2])
        Ap_list[i+1] = np.dot(np.diag(Y), Z).reshape([chi2, d2, chi3]).transpose([1, 0, 2])

    return Ap_list

def var_A(A_list, Ap_list, sweep='left'):
    '''
    ______________ Ap_list = (U|psi>)^\dagger
    |  |  |  |  |

    |  |  |  |  |
    -------------- A_list  = | phi >
    '''
    L = len(A_list)
    # dtype = A_list[0].dtype
    if sweep == 'left':
        Lp = np.ones([1, 1])
        Lp_list = [Lp]

        for i in range(L):
            Lp = np.tensordot(Lp, A_list[i], axes=(0, 1))  #[(1d),1u], [2p,(2l),2r]
            Lp = np.tensordot(Lp, np.conj(Ap_list[i]), axes=([0, 1], [1,0])) #[(1u),(1p),1r], [(2p),(2l),2r]
            Lp_list.append(Lp)

        Rp = np.ones([1, 1])

        A_list_new = [[] for i in range(L)]
        for i in range(L - 1, -1, -1):
            Rp = np.tensordot(Ap_list[i].conj(), Rp, axes=(2, 1))  #[1p,1l,(1r)] [2d,(2u)]
            theta = np.tensordot(Lp_list[i],Rp, axes=(1,1)) #[1d,(1u)], [2p,(2l),2r]
            theta = theta.transpose(1,0,2)  #[lpr]->[plr]
            A_list_new[i] = polar(theta).conj()
            Rp = np.tensordot(A_list_new[i], Rp, axes=([0,2], [0,2]))

        final_overlap = np.einsum('ijk,ijk->', A_list_new[0], theta)
        return A_list_new, final_overlap
    elif sweep == 'right':
        Rp = np.ones([1, 1])
        Rp_list = [Rp]
        for idx in range(L-1, -1, -1):
            Rp = np.tensordot(A_list[idx], Rp, axes=(2, 0))
            Rp = np.tensordot(Rp, np.conj(Ap_list[idx]), axes=([0, 2], [0, 2]))
            Rp_list.append(Rp)

        Lp = np.ones([1, 1])
        A_list_new = [[] for i in range(L)]
        for idx in range(L):
            Lp = np.tensordot(Lp, Ap_list[idx].conj(), axes=(1, 1))
            theta = np.tensordot(Lp, Rp_list[L-1-idx], axes=([2], [1]))
            theta = np.transpose(theta, [1,0,2])
            A_list_new[idx] = polar(theta).conj()
            ## d,ci1,chi2 = theta.shape
            ## Y,D,Z = misc.svd(theta.reshape(d*chi1,chi2), full_matrices=False)
            ## A_list_new[idx] = np.dot(Y,Z).reshape([d,chi1,chi2])
            # print("overlap : ", np.einsum('ijk,ijk->', theta, polar(theta).conj()))
            Lp = np.tensordot(A_list_new[idx], Lp, axes=([0, 1], [1, 0]))

        final_overlap = np.einsum('ijk,ijk->', A_list_new[L-1], theta)
        return A_list_new, final_overlap
    else:
        raise NotImplementedError



