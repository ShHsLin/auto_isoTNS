import numpy as np
import pandas as pd
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import ticker ## to control the number of tick in plot
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import FormatStrFormatter

def setup(fontsize):
    params = {'legend.fontsize':fontsize,#'xx-large',
                        # 'figure.figsize': (15,6),
                        'axes.labelsize': fontsize,
                        'axes.titlesize': fontsize,
                        'xtick.labelsize': fontsize,#'xx-large',
                        'ytick.labelsize': fontsize,#'xx-large',
                        'figure.autolayout':  False, #True,
                        'mathtext.fontset': u'cm',
                        'font.family': u'serif',
                        'font.serif': u'Times New Roman',
                        'pgf.texsystem':'pdflatex',
                        'text.usetex': True,
                        # 'text.latex.unicode': False,
                        # 'text.dvipnghack' : True
                       }

    pylab.rcParams.update(params)

