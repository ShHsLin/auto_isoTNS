""" Exact diagonalization code to find the ground state of
a 1D quantum Ising model."""

import sys, pickle
import scipy
import scipy.sparse as sparse
import numpy as np
import scipy.sparse.linalg.eigen.arpack as arp
import matplotlib.pyplot as plt
import mps_func


def Op_expectation(Op, site_i, vector, L):
    full_Op = scipy.sparse.kron(scipy.sparse.eye(2 ** (site_i)), Op)
    full_Op = scipy.sparse.kron(full_Op, scipy.sparse.eye(2 ** (L - site_i - 1)))
    full_Op = scipy.sparse.csr_matrix(full_Op)
    return vector.conjugate().dot(full_Op.dot(vector))

def gen_spin_operators(L):
    """" Returns the spin operators sigma_x and sigma_z for L sites """
    sx = sparse.csr_matrix(np.array([[0.,1.],[1.,0.]]))
    sy = sparse.csr_matrix(np.array([[0.,-1.j],[1.j,0.]]))
    sz = sparse.csr_matrix(np.array([[1.,0.],[0.,-1.]]))

    d = 2
    sx_list = []
    sy_list = []
    sz_list = []

    for i_site in range(L):
            if i_site==0:
                    X=sx
                    Y=sy
                    Z=sz
            else:
                    X= sparse.csr_matrix(np.eye(d))
                    Y= sparse.csr_matrix(np.eye(d))
                    Z= sparse.csr_matrix(np.eye(d))

            for j_site in range(1,L):
                    if j_site==i_site:
                            X=sparse.kron(X,sx, 'csr')
                            Y=sparse.kron(Y,sy, 'csr')
                            Z=sparse.kron(Z,sz, 'csr')
                    else:
                            X=sparse.kron(X,np.eye(d),'csr')
                            Y=sparse.kron(Y,np.eye(d),'csr')
                            Z=sparse.kron(Z,np.eye(d),'csr')
            sx_list.append(X)
            sy_list.append(Y)
            sz_list.append(Z)

    return sx_list, sy_list, sz_list

def gen_hamiltonian(sx_list, sy_list, sz_list, L):
    """" Generates the Hamiltonian """
    H_xx = sparse.csr_matrix((2**L,2**L))
    H_yy = sparse.csr_matrix((2**L,2**L))
    H_zz = sparse.csr_matrix((2**L,2**L))
    H_x = sparse.csr_matrix((2**L,2**L))
    H_y = sparse.csr_matrix((2**L,2**L))
    H_z = sparse.csr_matrix((2**L,2**L))

    for i in range(L-1):
            H_xx = H_xx + sx_list[i]*sx_list[np.mod(i+1,L)]
            H_yy = H_yy + sy_list[i]*sy_list[np.mod(i+1,L)]
            H_zz = H_zz + sz_list[i]*sz_list[np.mod(i+1,L)]

    for i in range(L):
            H_x = H_x + sx_list[i]
            H_y = H_y + sy_list[i]
            H_z = H_z + sz_list[i]

    return H_xx, H_yy, H_zz, H_x, H_y, H_z

def get_H_Ising(g, h, J, L, change_basis=False):
    sx_list, sy_list, sz_list  = gen_spin_operators(L)
    H_xx, H_yy, H_zz, H_x, H_y, H_z = gen_hamiltonian(sx_list, sy_list, sz_list, L)
    H = -J*H_xx - g*H_z - h*H_x
    if change_basis:
        H = -J*H_zz + g*H_x + h*H_z

    return H

def get_H_XXZ(g, J, L):
    sx_list, sy_list, sz_list  = gen_spin_operators(L)
    H_xx, H_yy, H_zz, H_x, H_y, H_z = gen_hamiltonian(sx_list, sy_list, sz_list, L)
    H = J * (H_xx + H_yy + g * H_zz)
    return H

def get_E_Ising_exact(g, h, J, L):
    H = get_H_Ising(g, h, J, L)
    e = arp.eigsh(H,k=1,which='SA',return_eigenvectors=False)
    return(e)

def get_E_XXZ_exact(g,J,L):
    H = get_H_XXZ(g, J, L)
    e = arp.eigsh(H,k=1,which='SA',return_eigenvectors=False)
    return(e)

def get_E_exact(g, J, L, H):
    raise NotImplementedError
    if H == 'TFI':
        return get_E_Ising_exact(g, h, J, L)
    elif H == 'XXZ':
        return get_E_XXZ_exact(g, J, L)
    else:
        raise


def global_quench(L, J, g, h):
    print(" Perform global Quench ")
    # Solving -J szsz + g sx + h sz

    H = get_H_Ising(g, h, J, L)

    ed_dt = 0.05
    H = np.array(H.todense())
    exp_iHdt = scipy.linalg.expm(-1.j * ed_dt * H)

    total_time = 49
    num_real = 1

    psi = np.zeros([2**L])
    psi[0] = 1.
    # x_state = np.array([1, 1]) / np.sqrt(2)
    # psi = x_state.copy()
    # for i in range(L-1):
    #     psi = np.kron(psi, x_state)

    sz_list = []

    dt_list = [0.5, 0.1, 0.01, 0.001]
    order_list = ['4th'] # , '2nd', '4th']
    F_dir_list = {(i,j):[] for i in order_list for j in dt_list}
    steps = int(total_time / ed_dt)

    for i in range(steps+1):
        sx = np.array([[0, 1],[1., 0]])
        sz = np.array([[1., 0.], [0., -1]])
        sz_mid = Op_expectation(sz, L//2, psi, L)
        # print("Sz at i={0} , {1:.6f} ".format(L//2, sz_mid))
        sz_list.append(sz_mid)

        # dir_head = '/space/ga63zuh/qTEBD/4_gate_counts/'
        dir_head = '/space/ga63zuh/qTEBD/tenpy_tebd/'

        if i % 10 ==0:
            print("<E(%.2f)> : " % (i*0.05), psi.conj().T.dot(H.dot(psi)))
            if i > 1 and i <steps:
                for dt in dt_list:
                    for order in order_list:
                        try:
                            wf_mps = pickle.load(open(dir_head + 'data_tebd_dt%e/1d_TFI_g1.4000_h0.1000/L11/wf_chi32_%s/T%.1f.pkl' % (dt, order, i*0.05), 'rb'))
                            psi_tebd = mps_func.MPS_2_state(wf_mps)
                            overlap = np.conjugate(psi).dot(psi_tebd)
                            print("overlap < psi | psi_tebd > = ", overlap, " F = ", np.square(np.abs(overlap)))
                            F_dir_list[(order, dt)].append((i*0.05, 1-np.square(np.abs(overlap))))
                        except Exception as e:
                            print(e)


        psi = exp_iHdt.dot(psi)


    plt.subplot(3,1,1)
    plt.plot(np.arange(steps+1)*0.05, sz_list, '--', label='exact')
    for dt in dt_list:
        for order in order_list:
            try:
                tebd_sz = np.load(dir_head + 'data_tebd_dt%e/1d_TFI_g1.4000_h0.1000/L11/mps_chi32_%s_sz_array.npy' % (dt, order))
                tebd_dt = np.load(dir_head + 'data_tebd_dt%e/1d_TFI_g1.4000_h0.1000/L11/mps_chi32_%s_dt.npy' % (dt, order))
                plt.plot(tebd_dt, tebd_sz[:, L//2], '--', label='tebd %s %.3f 2nd' % (order, dt))
            except Exception as e:
                print(e)


    plt.legend()


    plt.subplot(3,1,2)
    for dt in dt_list:
        if dt == 0.1:
            for order in order_list:
                try:
                    tebd_sz = np.load(dir_head + 'data_tebd_dt%e/1d_TFI_g1.4000_h0.1000/L11/mps_chi32_%s_sz_array.npy' % (dt, order))
                    tebd_dt = np.load(dir_head + 'data_tebd_dt%e/1d_TFI_g1.4000_h0.1000/L11/mps_chi32_%s_dt.npy' % (dt, order))
                    per_steps = int(dt / ed_dt)
                    tebd_len = len(sz_list[::2])
                    plt.semilogy(tebd_dt[:tebd_len], np.abs(tebd_sz[:tebd_len, L//2]-sz_list[::2]), label='%s %.3f' % (order, dt))
                except Exception as e:
                    print(e)


        elif dt == 0.01 or dt == 0.001:
            for order in order_list:
                try:
                    tebd_sz = np.load(dir_head + 'data_tebd_dt%e/1d_TFI_g1.4000_h0.1000/L11/mps_chi32_%s_sz_array.npy' % (dt, order))
                    tebd_dt = np.load(dir_head + 'data_tebd_dt%e/1d_TFI_g1.4000_h0.1000/L11/mps_chi32_%s_dt.npy' % (dt, order))
                    per_steps = int(ed_dt / (tebd_dt[1]-tebd_dt[0]))
                    plt.semilogy(np.arange(steps+1)*ed_dt, np.abs(tebd_sz[:per_steps*steps+1:per_steps, L//2]-sz_list), label='%s %.3f' % (order, dt))
                except Exception as e:
                    print(e)

            plt.ylabel('$| \Delta <Sz> |$')
            plt.legend()


    plt.subplot(3,1,3)
    for dt in dt_list:
        for order in order_list:
            if len(F_dir_list[(order, dt)]) != 0:
                plt.semilogy(*list(zip(*F_dir_list[(order, dt)])), 'x', markersize=4, label='<exact | mps(%s, %.3f) >' % (order, dt))
                # t_array=np.arange(0,total_time,0.5)
                # plt.semilogy(t_array, t_array*(dt)**3, '--')

    # tebd_err = np.load(dir_head + 'data_tebd/1d_TFI_g1.4000_h0.1000/L11/mps_chi32_2nd_error.npy')
    # plt.plot(tebd_dt, tebd_err, label='$1-\prod_i \mathcal{F}_i$')
    plt.legend()
    plt.ylabel('$1 - \mathcal{F}$')

    plt.show()


if __name__ == '__main__':
    L, J, g, h = sys.argv[1:]
    L, J, g, h = int(L), float(J), float(g), float(h)
    N = L
    print("python 1dIsing L=%d, J=%f, g=%f, h=%f" % (L, J, g, h))

    global_quench(L, J, g, h)



