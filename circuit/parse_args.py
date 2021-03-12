import argparse
import sys

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    """
    parse arguments
    """
    parser = argparse.ArgumentParser(description='Quantum Circuit Simulation')
    parser.add_argument('--L', dest='L',
                        help='system size. Default: 10',
                        default=10, type=int)
    parser.add_argument('--H', dest='H',
                        help='Hamiltonian, now we have TFI, XXZ'
                        'Default: TFI',
                        default='TFI', type=str)
    parser.add_argument('--g', dest='g',
                        help='coupling parameter g, only used in TFI'
                        'Default: 1.4',
                        default=1.4, type=float)
    parser.add_argument('--h', dest='h',
                        help='coupling parameter h, only used in TFI'
                        'Default: 0.',
                        default=0., type=float)
    parser.add_argument('--delta', dest='delta',
                        help='coupling parameter delta, only used in XXZ'
                        'Default: 1.',
                        default=1., type=float)

    parser.add_argument('--chi', dest='chi',
                        help='bond dimension of the mps'
                        'Default: 2',
                        default=2, type=int)
    parser.add_argument('--depth', dest='depth',
                        help='depth of the circuit'
                        'Default: 2',
                        default=2, type=int)
    parser.add_argument('--N_iter', dest='N_iter',
                        help='(maximum) number of iteration in the optimization'
                        'Default: 1',
                        default=1, type=int)
    parser.add_argument('--order', dest='order',
                        help='order in the trotterization used in time evolution'
                        'option: 1st, 2nd. Default: 1st',
                        default='1st', type=str)
    parser.add_argument('--T', dest='T',
                        help='the time T. Setting for target states or ending time evolution'
                        'Default: 0.',
                        default=0., type=float)
    parser.add_argument('--brickwall', dest='brickwall',
                        help='Whether or not using brickwall'
                        'type in 1 for true, 0 for false'
                        'Default: False',
                        default=0, type=int,
                       )



    if len(sys.argv) == 1:
        pass
        # parser.print_help()
        # sys.exit(1)

    args = parser.parse_args()
    return args
