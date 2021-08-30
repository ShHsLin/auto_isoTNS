import numpy as np
import misc
import circuit_func
try:
    raise
    import tcl.tcl
    einsum = tcl.tcl.einsum
except:
    einsum = np.einsum

def init_mps(L, chi, d):
    '''
    Return MPS in AAAAAA form, i.e. left canonical form
    such that \sum A^\dagger A = I
    '''
    A_list = []
    for i in range(L):
        chi1 = np.min([d**np.min([i,L-i]),chi])
        chi2 = np.min([d**np.min([i+1,L-i-1]),chi])
        try:
            A_list.append(circuit_func.polar(0.5 - np.random.uniform(size=[d,chi1,chi2])))
            # A_list.append(polar(np.random.uniform(size=[d,chi1,chi2])))
        except:
            A_list.append(circuit_func.polar(0.5 - onp.random.uniform(size=[d,chi1,chi2])))
            # A_list.append(polar(onp.random.uniform(size=[d,chi1,chi2])))

    return A_list

def plrq_2_plr(A_list):
    new_A_list = []
    for a in A_list:
        p_dim, l_dim, r_dim, q_dim = a.shape
        new_A_list.append(np.transpose(a, [0,3,1,2]).reshape([p_dim*q_dim, l_dim, r_dim]))

    return new_A_list

def plr_2_plrq(A_list, p_dim=2, q_dim=2):
    new_A_list = []
    for a in A_list:
        pq_dim, l_dim, r_dim = a.shape
        new_A_list.append(np.reshape(a, [p_dim, q_dim, l_dim, r_dim]).transpose([0, 2, 3, 1]))

    return new_A_list

def lpr_2_plr(A_list):
    return [np.transpose(a, [1,0,2]) for a in A_list]

def plr_2_lpr(A_list):
    return [np.transpose(a, [1,0,2]) for a in A_list]

def MPS_dot_left_env(mps_up, mps_down, site_l, cache_env_list=None):
    '''
    # Complex compatible
    Goal:
        Contract and form the left environment of site_l
    Input:
        mps_up : up (should actually add conjugate !!! )
        mps_down : down
        site_l : Index convention starting from 0
    Output:
        left_environment

    |-----------------  -----
    | left                |
    | environment         |site_l
    |-----------------  -----
    0,...,(site_l-1)
    '''
    dtype = mps_up[0].dtype
    assert dtype == mps_down[0].dtype

    if site_l == 0:
        return np.eye(1, dtype=dtype)

    left_env = np.eye(1, dtype=dtype)
    for idx in range(0, site_l):
        left_env = einsum('ij,ikl->jkl', left_env, mps_up[idx].conjugate())
        left_env = einsum('ijk,ijl->kl', left_env, mps_down[idx])
        if not (cache_env_list is None):
            cache_env_list[idx] = left_env

    return left_env


def MPS_dot_right_env(mps_up, mps_down, site_l, cache_env_list=None):
    '''
    # Complex compatible
    Goal:
        Contract and form the right environment of site_l
    Input:
        mps_up : up (should actually add conjugate !!! )
        mps_down : down
        site_l : Index convention starting from 0
        cache_env_list : List with length L
    Output:
        right_environment
        cache_env_list : with list[site_l] = contraction including site_l

      -----       -----------------|
        |          right           |
        |site_l    environment     |
      -----       -----------------|
                    (site_l+1),...,L-1
    '''
    L = len(mps_up)
    dtype = mps_up[0].dtype
    assert dtype == mps_down[0].dtype
    if site_l == L - 1:
        return np.eye(1, dtype=dtype)

    right_env = np.eye(1, dtype=dtype)
    for idx in range(L - 1, site_l, -1):
        right_env = einsum('kli, ij ->klj', mps_up[idx].conjugate(),
                              right_env)
        right_env = einsum('klj,mlj->km', right_env, mps_down[idx])
        if not (cache_env_list is None):
            cache_env_list[idx] = right_env

    return right_env


def MPS_dot(mps_1, mps_2):
    '''
    # Complex compatible
    Return inner product of two MPS, with mps_1 taking complex_conjugate
    <mps_1 | mps_2 >
    '''
    L = len(mps_1)
    mps_temp = einsum('ijk,ijl->kl', mps_1[0].conjugate(), mps_2[0])
    for idx in range(1, L):
        mps_temp = einsum('ij,ikl->jkl', mps_temp, mps_1[idx].conjugate())
        mps_temp = einsum('ijk,ijl->kl', mps_temp, mps_2[idx])

    return mps_temp[0, 0]


def MPS_compression_variational(mps_trial, mps_target, max_iter=30, tol=1e-4,
                                verbose=0):
    '''
    Variational Compression on MPS with mps_trial given.
    Input:
        mps_trial: MPS for optimization
            The input should be in right canonical form BBBBBB and
            it should be normalized.
        mps_target: The target to approximate.
            It is not necessary in canonical form and not necessarily
            normalized.

    Output:
        trunc_err
        modification mps_trial inplace still in right canonical form
    '''
    L = len(mps_trial)
    # Check normalization
    if np.abs(MPS_dot(mps_trial, mps_trial) - 1.) > 1e-8:
        print(('mps_comp_var not normalized', MPS_dot(mps_trial, mps_trial)))
        raise
    elif np.abs(MPS_dot(mps_target, mps_target) - 1.) > 1e-8:
        print(('mps_comp_var not normalized', MPS_dot(mps_target, mps_target)))
        mps_target[-1] /= np.sqrt(MPS_dot(mps_target, mps_target))
    else:
        pass
        # all normalized

    conv = False
    num_iter = 0
    old_trunc_err = 1.
    while (num_iter < max_iter and not conv):
        num_iter += 1
        # Creat cache of environment
        cache_env_list = [None] * L
        MPS_dot_right_env(mps_trial,
                          mps_target,
                          0,
                          cache_env_list=cache_env_list)

        # site = 0
        right_env = cache_env_list[1]
        left_env = np.eye(1)
        update_tensor = einsum('ij,jkl->ikl', left_env, mps_target[0])
        update_tensor = einsum('ikl,ml->ikm', update_tensor, right_env)
        mps_trial[0] = update_tensor
        # svd to shift central site
        l_dim, d, r_dim = mps_trial[0].shape
        U, s, Vh = np.linalg.svd(mps_trial[0].reshape((l_dim * d, r_dim)),
                                 full_matrices=False)
        rank = s.size
        s /= np.linalg.norm(s)
        mps_trial[0] = U.reshape((l_dim, d, rank))
        mps_trial[1] = einsum('ij,jkl->ikl',
                              np.diag(s).dot(Vh), mps_trial[1])
        # update env
        left_env = einsum('ijk,ijl->kl', mps_trial[0].conjugate(),
                             mps_target[0])
        cache_env_list[0] = left_env
        cache_env_list[1] = None

        for site in range(1, L - 1):
            right_env = cache_env_list[site + 1]
            left_env = cache_env_list[site - 1]
            update_tensor = einsum('ij,jkl->ikl', left_env,
                                   mps_target[site])
            update_tensor = einsum('ikl,ml->ikm', update_tensor, right_env)
            mps_trial[site] = update_tensor
            # svd to shift central site
            l_dim, d, r_dim = mps_trial[site].shape
            U, s, Vh = np.linalg.svd(mps_trial[site].reshape(
                (l_dim * d, r_dim)),
                                     full_matrices=False)
            rank = s.size
            s /= np.linalg.norm(s)
            mps_trial[site] = U.reshape((l_dim, d, rank))
            mps_trial[site + 1] = einsum('ij,jkl->ikl',
                                         np.diag(s).dot(Vh),
                                         mps_trial[site + 1])
            # update env
            left_env = einsum('ij,ikl->jkl', left_env,
                              mps_trial[site].conjugate())
            left_env = einsum('ijk,ijl->kl', left_env, mps_target[site])
            cache_env_list[site] = left_env
            cache_env_list[site + 1] = None

        # site = L-1
        right_env = np.eye(1)
        left_env = cache_env_list[L - 2]
        update_tensor = einsum('ij,jkl->ikl', left_env, mps_target[L - 1])
        update_tensor = einsum('ikl,ml->ikm', update_tensor, right_env)
        mps_trial[L - 1] = update_tensor
        # svd to shift central site
        l_dim, d, r_dim = mps_trial[L - 1].shape
        U, s, Vh = np.linalg.svd(mps_trial[L - 1].reshape((l_dim, d * r_dim)),
                                 full_matrices=False)
        rank = s.size
        s /= np.linalg.norm(s)
        mps_trial[L - 1] = Vh.reshape((rank, d, r_dim))
        mps_trial[L - 2] = einsum('ijk,kl->ijl', mps_trial[L - 2],
                                  U.dot(np.diag(s)))
        # No update for left_env

        # site = L-1
        # But update for right_env
        right_env = einsum('ijk,ljk->il', mps_trial[L - 1].conjugate(),
                           mps_target[L - 1])
        cache_env_list[L - 1] = right_env
        cache_env_list[L - 2] = None

        for site in range(L - 2, 0, -1):
            right_env = cache_env_list[site + 1]
            left_env = cache_env_list[site - 1]
            update_tensor = einsum('ij,jkl->ikl', left_env,
                                   mps_target[site])
            update_tensor = einsum('ikl,ml->ikm', update_tensor, right_env)
            mps_trial[site] = update_tensor
            # svd to shift central site
            l_dim, d, r_dim = mps_trial[site].shape
            U, s, Vh = np.linalg.svd(mps_trial[site].reshape(
                (l_dim, d * r_dim)),
                                     full_matrices=False)
            rank = s.size
            s /= np.linalg.norm(s)
            mps_trial[site] = Vh.reshape((rank, d, r_dim))
            mps_trial[site - 1] = einsum('ijk,kl->ijl', mps_trial[site - 1],
                                         U.dot(np.diag(s)))
            # update env
            right_env = einsum('kli, ij ->klj', mps_trial[site].conjugate(),
                               right_env)
            right_env = einsum('klj,mlj->km', right_env, mps_target[site])
            cache_env_list[site] = right_env
            cache_env_list[site - 1] = None

        # site = 0
        trunc_err = 1. - np.square(np.abs(MPS_dot(mps_trial, mps_target)))
        if verbose:
            print(('var_trunc_err = ', trunc_err))

        if np.abs(old_trunc_err - trunc_err) < 1e-6:
            conv = True

        old_trunc_err = trunc_err

    return trunc_err

def MPS_2_state(mps):
    '''
    Goal:
        Return the full tensor representation (vector) of the state
    Input:
        MPS: in [p,l,r] form
    '''
    Vec = mps[0][:,0,:]
    for idx in range(1, len(mps)):
        Vec = einsum('pa,qal->pql', Vec, mps[idx])
        dim_p, dim_q, dim_l = Vec.shape
        Vec = Vec.reshape([dim_p * dim_q, dim_l])

    return Vec.flatten()

def state_2_MPS(psi, L, chimax, eps=1e-15):
    '''
    Input:
        psi: the state
        L: the system size
        chimax: the maximum bond dimension
    '''
    psi_aR = np.reshape(psi, (1, 2**L))
    Ms = []
    for n in range(1, L+1):
        chi_n, dim_R = psi_aR.shape
        assert dim_R == 2**(L-(n-1))
        psi_LR = np.reshape(psi_aR, (chi_n*2, dim_R//2))
        M_n, lambda_n, psi_tilde = misc.svd(psi_LR, full_matrices=False)

        chimax_current = np.amin([chimax,
                                  np.sum((lambda_n / np.linalg.norm(lambda_n)) > eps)])

        if len(lambda_n) > chimax_current:
            keep = np.argsort(lambda_n)[::-1][:chimax]
            M_n = M_n[:, keep]
            lambda_n = lambda_n[keep]
            psi_tilde = psi_tilde[keep, :]

        chi_np1 = len(lambda_n)
        M_n = np.reshape(M_n, (chi_n, 2, chi_np1))
        Ms.append(M_n)
        psi_aR = lambda_n[:, np.newaxis] * psi_tilde[:,:]
    assert psi_aR.shape == (1, 1)
    return lpr_2_plr(Ms)

def MPO_2_operator(mpo):
    '''
    Goal:
        Return the full operator (2**L, 2**L)
    Input:
        MPO: in [p, l, r, q] form
    '''
    Op = mpo[0][:, 0, :, :]
    for idx in range(1, len(mpo)):
        Op = einsum('paq,PalQ->pPlqQ', Op, mpo[idx])
        dim_p, dim_P, dim_l, dim_q, dim_Q = Op.shape
        Op = Op.reshape([dim_p*dim_P, dim_l, dim_q*dim_Q])

    assert Op.shape[1] == 1
    return Op[:,0,:]

def operator_2_MPO(op, L, chimax):
    '''
    Input:
        op: the operator in matrix of size (2**L, 2**L)
        L: system size
        chimax: the maximum bond dimension
    Return:
        MPO in [p, l, r, q] form
    '''
    op_aR = np.reshape(op, (1, 2**L, 2**L))
    Ms = []
    for n in range(1, L+1):
        chi_n, dim_R1, dim_R2 = op_aR.shape
        assert dim_R1 == 2**(L-(n-1))
        op_LR = np.reshape(op_aR, [chi_n*2, dim_R1//2, 2, dim_R2//2])
        op_LR = np.transpose(op_LR, [0, 2, 1, 3]).reshape([chi_n*4, (dim_R1//2) * (dim_R2//2)])
        M_n, lambda_n, op_tilde = misc.svd(op_LR, full_matrices=False)
        if len(lambda_n) > chimax:
            keep = np.argsort(lambda_n)[::-1][:chimax]
            M_n = M_n[:, keep]
            lambda_n = lambda_n[keep]
            op_tilde = op_tilde[keep, :]

        chi_np1 = len(lambda_n)
        M_n = np.reshape(M_n, (chi_n, 2, 2, chi_np1))
        Ms.append(M_n.transpose([1, 0, 3, 2]))
        op_aR = lambda_n[:, np.newaxis] * op_tilde[:,:]
        op_aR = op_aR.reshape([chi_np1, (dim_R1//2), (dim_R2//2)])

    assert op_aR.shape == (1, 1, 1)
    Ms[-1] = Ms[-1] * op_aR[0, 0, 0]
    return Ms

def overlap(psi1, psi2):
    '''
    psi1 is not taken complex conjugate beforehand.
    psi1 : with dimension [p, l, r]
    psi2 : with dimension [p, l, r]
    '''
    N = np.ones([1,1]) # a ap 
    L = len(psi1)
    for i in np.arange(L):
        N = np.tensordot(N, np.conj(psi1[i]), axes=(1,1))  # a (ap), p (lp) rp -> a, p, rp
        N = np.tensordot(N,psi2[i], axes=([0,1],[1,0])) # (a) (p) rp, (p) (l) r -> rp r
        N = np.transpose(N, [1,0])

    assert N.size == 1
    N = np.trace(N)
    return(N)

def expectation_values_1_site(A_list, Op_list, check_norm=True):
    if check_norm:
        assert np.isclose(overlap(A_list, A_list), 1.)
    else:
        pass

    L = len(A_list)
    Lp = np.ones([1, 1])
    Lp_list = [Lp]

    for i in range(L):
        Lp = np.tensordot(Lp, A_list[i], axes=(0, 1)) # ap i b
        Lp = np.tensordot(Lp, np.conj(A_list[i]), axes=([0, 1], [1,0])) # b bp
        Lp_list.append(Lp)

    Rp = np.ones([1, 1])

    Op_per_site = np.zeros([L], dtype=np.complex)
    for i in range(L - 2, -2, -1):
        Rp = np.tensordot(np.conj(A_list[i+1]), Rp, axes=(2, 1)) #[p,l,r] [d,u] -> [p,l,d]
        Op = np.tensordot(Op_list[i+1], Rp, axes=(1, 0)) #[p,q], [q,l,d] -> [p,l,d]
        Op = np.tensordot(Op, A_list[i+1], axes=([0,2], [0,2])) #[p,ul,d], [p,dl,rr] -> [ul, dl]
        Op = np.tensordot(Lp_list[i+1], Op, axes=([0,1], [1,0]))
        Op_per_site[i+1] = Op[None][0]

        Rp = np.tensordot(A_list[i+1],Rp,axes=([0,2], [0,2]))

    return Op_per_site

def expectation_values(A_list, H_list, check_norm=True):
    if check_norm:
        assert np.isclose(np.abs(overlap(A_list, A_list)), 1.)
    else:
        pass

    L = len(A_list)
    Lp = np.ones([1, 1])
    Lp_list = [Lp]

    for i in range(L):
        Lp = np.tensordot(Lp, A_list[i], axes=(0, 1)) # ap i b
        Lp = np.tensordot(Lp, np.conj(A_list[i]), axes=([0, 1], [1,0])) # b bp
        Lp_list.append(Lp)

    Rp = np.ones([1, 1])

    E_list = []
    for i in range(L - 2, -1, -1):
        Rp = np.tensordot(np.conj(A_list[i+1]), Rp, axes=(2, 1)) #[p,l,r] [d,u] -> [p,l,d]
        E = np.tensordot(np.conj(A_list[i]), Rp, axes=(2, 1)) #[p,l,r] [q,L,R] -> [p,l,q,R]
        E = np.tensordot(H_list[i],E, axes=([0,1], [0,2])) # [p,q, r,s] , [p,l,q,R] -> [r,s, l,R]
        E = np.tensordot(A_list[i+1],E, axes=([0,2], [1,3])) # [s, L, R], [r,s, l,R] -> [L, r, l]
        E = np.tensordot(A_list[i],E, axes=([0,2], [1,0])) # [r, ll, L] [L, r, l] -> [ll, l]
        E = np.tensordot(Lp_list[i],E, axes=([0,1],[0,1]))
        Rp = np.tensordot(A_list[i+1],Rp,axes=([0,2], [0,2]))
        E_list.append(E[None][0])

    return E_list

def right_canonicalize(A_list, no_trunc=False, chi=None, normalized=True):
    '''
    Bring mps in right canonical form, assuming the input mps is in
    left canonical form already.

    modification in place
    '''
    L = len(A_list)
    tot_trunc_err = 0.
    for i in range(L-1, 0, -1):
        d1, chi1, chi2 = A_list[i].shape
        X, Y, Z = misc.svd(np.reshape(np.transpose(A_list[i], [1, 0, 2]), [chi1, d1 * chi2]),
                                full_matrices=0)

        if no_trunc:
            chi1 = np.size(Y)
        else:
            chi1 = np.sum((Y/np.linalg.norm(Y))>1e-14)

        if chi is not None:
            chi1 = np.amin([chi1, chi])

        trunc_idx = (np.argsort(Y)[::-1])[chi1:]
        trunc_error = np.sum(Y[trunc_idx] ** 2) / np.sum(Y ** 2)
        tot_trunc_err = tot_trunc_err + trunc_error

        arg_sorted_idx = (np.argsort(Y)[::-1])[:chi1]
        Y = Y[arg_sorted_idx]
        if normalized:
            Y = Y / np.linalg.norm(Y)

        X = X[:, arg_sorted_idx]
        Z = Z[arg_sorted_idx, :]

        A_list[i] = np.transpose(Z.reshape([chi1, d1, chi2]), [1, 0, 2])
        R = np.dot(X, np.diag(Y))
        new_A = np.tensordot(A_list[i-1], R, axes=([2], [0]))  #[p, 1l, (1r)] [(2l), 2r]
        A_list[i-1] = new_A

    if normalized:
        A_list[0] = A_list[0] / np.linalg.norm(A_list[0])

    return A_list, tot_trunc_err

def left_canonicalize(A_list, no_trunc=False, chi=None, normalized=True):
    '''
    Bring mps in left canonical form, assuming the input mps is in
    right canonical form already.

    modification in place
    '''
    L = len(A_list)
    tot_trunc_err = 0
    for i in range(L-1):
        d1, chi1, chi2 = A_list[i].shape
        X, Y, Z = misc.svd(np.reshape(A_list[i], [d1 * chi1, chi2]),
                                full_matrices=0)

        if no_trunc:
            chi2 = np.size(Y)
        else:
            chi2 = np.sum((Y/np.linalg.norm(Y))>1e-14)

        if chi is not None:
            chi2 = np.amin([chi2, chi])

        trunc_idx = (np.argsort(Y)[::-1])[chi2:]
        trunc_error = np.sum(Y[trunc_idx] ** 2) / np.sum(Y ** 2)
        tot_trunc_err = tot_trunc_err + trunc_error

        arg_sorted_idx = (np.argsort(Y)[::-1])[:chi2]
        Y = Y[arg_sorted_idx]
        if normalized:
            Y = Y / np.linalg.norm(Y)

        X = X[: ,arg_sorted_idx]
        Z = Z[arg_sorted_idx, :]

        A_list[i] = X.reshape([d1, chi1, chi2])
        R = np.dot(np.diag(Y), Z)
        new_A = np.tensordot(R, A_list[i+1], axes=([1], [1]))  #[1l,(1r)],[p, (2l), 2r]
        A_list[i+1] = np.transpose(new_A, [1, 0, 2])

    if normalized:
        A_list[-1] = A_list[-1] / np.linalg.norm(A_list[-1])

    return A_list, tot_trunc_err

def get_entanglement(A_list):
    '''
    Goal:
        Compute the bibpartite entanglement at each cut.
    Input:
        mps in left canonical form
    Output:
        list of bipartite entanglement [(0,1...), (01,2...), (012,...)]
    '''
    L = len(A_list)
    copy_A_list = [A.copy() for A in A_list]
    ent_list = [None] * (L-1)
    for i in range(L-1, 0, -1):
        d1, chi1, chi2 = copy_A_list[i].shape
        X, Y, Z = misc.svd(np.reshape(np.transpose(copy_A_list[i], [1, 0, 2]), [chi1, d1 * chi2]),
                                full_matrices=0)

        chi1 = np.sum(Y>1e-14)

        arg_sorted_idx = (np.argsort(Y)[::-1])[:chi1]
        Y = Y[arg_sorted_idx]
        X = X[: ,arg_sorted_idx]
        Z = Z[arg_sorted_idx, :]

        copy_A_list[i]   = np.transpose(Z.reshape([chi1, d1, chi2]), [1, 0, 2])
        R = np.dot(X, np.diag(Y))
        new_A = np.tensordot(copy_A_list[i-1], R, axes=([2], [0]))  #[p, 1l, (1r)] [(2l), 2r]
        copy_A_list[i-1] = new_A

        bi_ent = -(Y**2).dot(np.log(Y**2))
        ent_list[i-1] = bi_ent

    return ent_list

def get_renyi_n_entanglement(A_list, n):
    '''
    Goal:
        Compute the renyi-n entanglement at each cut.
    Input:
        mps in left canonical form
    Output:
        list of bipartite entanglement [(0,1...), (01,2...), (012,...)]
    '''
    L = len(A_list)
    copy_A_list = [A.copy() for A in A_list]
    ent_list = [None] * (L-1)
    for i in range(L-1, 0, -1):
        d1, chi1, chi2 = copy_A_list[i].shape
        X, Y, Z = misc.svd(np.reshape(np.transpose(copy_A_list[i], [1, 0, 2]), [chi1, d1 * chi2]),
                                full_matrices=0)

        chi1 = np.sum(Y>1e-14)

        arg_sorted_idx = (np.argsort(Y)[::-1])[:chi1]
        Y = Y[arg_sorted_idx]
        X = X[: ,arg_sorted_idx]
        Z = Z[arg_sorted_idx, :]

        copy_A_list[i]   = np.transpose(Z.reshape([chi1, d1, chi2]), [1, 0, 2])
        R = np.dot(X, np.diag(Y))
        new_A = np.tensordot(copy_A_list[i-1], R, axes=([2], [0]))  #[p, 1l, (1r)] [(2l), 2r]
        copy_A_list[i-1] = new_A

        bi_ent = np.log(np.sum(Y**(2*n))) / (1-n)
        ent_list[i-1] = bi_ent

    return ent_list

