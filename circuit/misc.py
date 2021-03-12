import numpy as np
import pickle
import scipy.linalg
import os

def nparray_2_dict(A_array):
    A_dict = {int(row[0]) : row[1] for row in A_array}
    return A_dict

def dict_2_nparray(A_dict):
    num_data = len(A_dict)
    A_array = np.zeros([num_data, 2], dtype=float)
    for idx, key in enumerate(A_dict.keys()):
        A_array[idx, 0] = key
        A_array[idx, 1] = A_dict[key]

    return A_array

def load_array(path):
    B_array = np.loadtxt(path, delimiter=',')
    return B_array.reshape([-1, 2])

def save_array(path, A_array):
    np.savetxt(path, A_array, delimiter=',', fmt='%d, %.16e' )

def svd(theta, compute_uv=True, full_matrices=True):
    """SVD with gesvd backup"""
    try:
        return scipy.linalg.svd(theta,
                                compute_uv=compute_uv,
                                full_matrices=full_matrices)
    except:
        print("*gesvd*")
        return scipy.linalg.svd(theta,
                                compute_uv=compute_uv,
                                full_matrices=full_matrices,
                                lapack_driver='gesvd')

def save_circuit_simple(dir_path, circuit, data_dict):
    '''
    Goal:
        new version of saving circuit.
        circuit depth, circuit type, the optimization proceduce
        should be include in dir_path.

        all data are now combined in data_dict to avoid cluttering.

    Input:
        dir_path
        circuit
        data_dict
    '''
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    filename = 'circuit.pkl'
    path = dir_path + filename
    pickle.dump(circuit, open(path, 'wb'))

    filename = 'data_dict.pkl'
    path = dir_path + filename
    pickle.dump(data_dict, open(path, 'wb'))
    return

def load_circuit_simple(dir_path):
    filename = 'circuit.pkl'
    path = dir_path + filename
    circuit = pickle.load(open(path, 'rb'))

    filename = 'data_dict.pkl'
    path = dir_path + filename
    data_dict = pickle.load(open(path, 'rb'))
    return circuit, data_dict


def save_circuit(dir_path, depth, N_iter, order,
                 circuit, E_list, t_list, update_error_list,
                 Sz_array, ent_array, num_iter_array, running):
    '''
    If running:
        Saving data to $dir_path/XXX_tmp.npy
    else:
        Saving data to $dir_path/XXX.npy
        Removing data from $dir_path/XXX_tmp.npy

    '''
    if running:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        filename = 'circuit_depth%d_Niter%d_%s_circuit_tmp.pkl' % (depth, N_iter, order)
        path = dir_path + filename
        pickle.dump(circuit, open(path, 'wb'))

        filename = 'circuit_depth%d_Niter%d_%s_energy_tmp.npy' % (depth, N_iter, order)
        path = dir_path + filename
        np.save(path, np.array(E_list))

        filename = 'circuit_depth%d_Niter%d_%s_dt_tmp.npy' % (depth, N_iter, order)
        path = dir_path + filename
        np.save(path, np.array(t_list))

        filename = 'circuit_depth%d_Niter%d_%s_error_tmp.npy' % (depth, N_iter, order)
        path = dir_path + filename
        np.save(path, np.array(update_error_list))

        filename = 'circuit_depth%d_Niter%d_%s_sz_array_tmp.npy' % (depth, N_iter, order)
        path = dir_path + filename
        np.save(path, Sz_array)

        filename = 'circuit_depth%d_Niter%d_%s_ent_array_tmp.npy' % (depth, N_iter, order)
        path = dir_path + filename
        np.save(path, ent_array)

        filename = 'circuit_depth%d_Niter%d_%s_Niter_array_tmp.npy' % (depth, N_iter, order)
        path = dir_path + filename
        np.save(path, num_iter_array)
    else:  # running=False, the simulation ends
        filename = 'circuit_depth%d_Niter%d_%s_circuit.pkl' % (depth, N_iter, order)
        path = dir_path + filename
        pickle.dump(circuit, open(path, 'wb'))
        tmp_filename = filename[:-4] + '_tmp' + filename[-4:]
        os.remove(dir_path + tmp_filename)

        filename = 'circuit_depth%d_Niter%d_%s_energy.npy' % (depth, N_iter, order)
        path = dir_path + filename
        np.save(path, np.array(E_list))
        tmp_filename = filename[:-4] + '_tmp' + filename[-4:]
        os.remove(dir_path + tmp_filename)

        filename = 'circuit_depth%d_Niter%d_%s_dt.npy' % (depth, N_iter, order)
        path = dir_path + filename
        np.save(path, np.array(t_list))
        tmp_filename = filename[:-4] + '_tmp' + filename[-4:]
        os.remove(dir_path + tmp_filename)

        filename = 'circuit_depth%d_Niter%d_%s_error.npy' % (depth, N_iter, order)
        path = dir_path + filename
        np.save(path, np.array(update_error_list))
        tmp_filename = filename[:-4] + '_tmp' + filename[-4:]
        os.remove(dir_path + tmp_filename)

        filename = 'circuit_depth%d_Niter%d_%s_sz_array.npy' % (depth, N_iter, order)
        path = dir_path + filename
        np.save(path, Sz_array)
        tmp_filename = filename[:-4] + '_tmp' + filename[-4:]
        os.remove(dir_path + tmp_filename)

        filename = 'circuit_depth%d_Niter%d_%s_ent_array.npy' % (depth, N_iter, order)
        path = dir_path + filename
        np.save(path, ent_array)
        tmp_filename = filename[:-4] + '_tmp' + filename[-4:]
        os.remove(dir_path + tmp_filename)

        filename = 'circuit_depth%d_Niter%d_%s_Niter_array.npy' % (depth, N_iter, order)
        path = dir_path + filename
        np.save(path, num_iter_array)
        tmp_filename = filename[:-4] + '_tmp' + filename[-4:]
        os.remove(dir_path + tmp_filename)

    return

def check_circuit(dir_path, depth, N_iter, order):
    filename = 'circuit_depth%d_Niter%d_%s_circuit.pkl' % (depth, N_iter, order)
    path = dir_path + filename
    return os.path.exists(path)

def load_circuit(dir_path, depth, N_iter, order, tmp=True):
    if tmp:
        filename = 'circuit_depth%d_Niter%d_%s_circuit_tmp.pkl' % (depth, N_iter, order)
        path = dir_path + filename
        circuit = pickle.load(open(path, 'rb'))

        filename = 'circuit_depth%d_Niter%d_%s_energy_tmp.npy' % (depth, N_iter, order)
        path = dir_path + filename
        E_list = np.load(path).tolist()

        filename = 'circuit_depth%d_Niter%d_%s_dt_tmp.npy' % (depth, N_iter, order)
        path = dir_path + filename
        t_list = np.load(path).tolist()

        filename = 'circuit_depth%d_Niter%d_%s_error_tmp.npy' % (depth, N_iter, order)
        path = dir_path + filename
        update_error_list = np.load(path).tolist()

        filename = 'circuit_depth%d_Niter%d_%s_sz_array_tmp.npy' % (depth, N_iter, order)
        path = dir_path + filename
        Sz_array = np.load(path)

        filename = 'circuit_depth%d_Niter%d_%s_ent_array_tmp.npy' % (depth, N_iter, order)
        path = dir_path + filename
        ent_array = np.load(path)

        filename = 'circuit_depth%d_Niter%d_%s_Niter_array_tmp.npy' % (depth, N_iter, order)
        path = dir_path + filename
        num_iter_array = np.load(path)

        running_idx = len(t_list)
        return (running_idx, circuit, E_list, t_list, update_error_list,
                Sz_array, ent_array, num_iter_array)
    else:
        filename = 'circuit_depth%d_Niter%d_%s_circuit.pkl' % (depth, N_iter, order)
        path = dir_path + filename
        circuit = pickle.load(open(path, 'rb'))

        filename = 'circuit_depth%d_Niter%d_%s_energy.npy' % (depth, N_iter, order)
        path = dir_path + filename
        E_list = np.load(path).tolist()

        filename = 'circuit_depth%d_Niter%d_%s_dt.npy' % (depth, N_iter, order)
        path = dir_path + filename
        t_list = np.load(path).tolist()

        filename = 'circuit_depth%d_Niter%d_%s_error.npy' % (depth, N_iter, order)
        path = dir_path + filename
        update_error_list = np.load(path).tolist()

        filename = 'circuit_depth%d_Niter%d_%s_sz_array.npy' % (depth, N_iter, order)
        path = dir_path + filename
        Sz_array = np.load(path)

        filename = 'circuit_depth%d_Niter%d_%s_ent_array.npy' % (depth, N_iter, order)
        path = dir_path + filename
        ent_array = np.load(path)

        filename = 'circuit_depth%d_Niter%d_%s_Niter_array.npy' % (depth, N_iter, order)
        path = dir_path + filename
        num_iter_array = np.load(path)

        running_idx = len(t_list)
        return (running_idx, circuit, E_list, t_list, update_error_list,
                Sz_array, ent_array, num_iter_array)


