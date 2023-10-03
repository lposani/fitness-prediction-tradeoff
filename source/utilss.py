from .imports import *
from scipy.optimize import newton


def mkdir(filename):
    if not os.path.exists(filename):
        os.makedirs(filename)


def rmdir(path):
    shutil.rmtree(path)


def mkparent(filename):
    slash = [pos for pos, char in enumerate(filename) if char == '/']
    save_folder = filename[:slash[-1]]
    mkdir(save_folder)


def parentdir(filename):
    slash = [pos for pos, char in enumerate(filename) if char == '/']
    parent = filename[:slash[-1]]
    return parent


def contacts_to_matrix(contacts, N):
    C = np.zeros((N, N))
    for c in contacts:
        C[c[0], c[1]] = 1
    return C


def gauge_and_pcut_moments(av, corr, gaug = 'dissensus', p_cut = 0):
    N = np.shape(av)[0]
    A = np.shape(av)[1]

    if gaug == 'consensus':
        gauge = np.argmax(av, 1)
    if gaug == 'dissensus':
        gauge = np.argmin(av, 1)

    for i in range(N):
        av[i][gauge[i]] = 0
        for j in range(N):
            for a in range(A):
                corr[i, j, gauge[i], a] = 0
                corr[i, j, a, gauge[j]] = 0

    a_plus = np.where(av > p_cut, av, 0)
    a_minus = np.where(av <= p_cut, av, 0)
    a_m = np.sum(a_minus, 1)


def contacts_dictionary(contacts, N):
    contact_dict = {}
    for i in range(N):
        contact_dict[i] = []
        for c in contacts:
            if c[0]==i:
                contact_dict[i].append(c[1])
            if c[1]==i:
                contact_dict[i].append(c[0])
        contact_dict[i] = np.asarray(contact_dict[i])
        contact_dict[i].sort()
    return contact_dict


def spearNaN(a, b):
    mask = (np.isnan(a)==0) & (np.isnan(b)==0)
    R, p = spearmanr(a[mask], b[mask])
    return R, p


def conditional_load(path, function, overwrite=False):
    if os.path.exists(path) and overwrite==False:
        print("[conditional load]\t loading data from %s" % path)
        data = np.loadtxt(path)
    else:
        data = function()
        print("[conditional load]\t saving data to %s" % path)
        np.savetxt(path, data)
    return data

def hard_cutoff_weights(msa, wt, D):
    B, N = msa.shape
    # D = int(d*N)
    dists = np.sum(wt != msa, 1)
    W = np.asarray(dists <= D, dtype=float)
    return W


def p_to_ast(p):
    t = 'ns'
    if p<0.05:
        t = '*'
    if p<0.01:
        t='**'
    if p<0.001:
        t='***'
    return t


from math import floor, log10
def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    if exponent is None:
        exponent = int(floor(log10(abs(num))))
    coeff = round(num / float(10**exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits

    return r"${0:.{2}f}\cdot10^{{{1:d}}}$".format(coeff, exponent, precision)


def compute_weighted_averages(msa, W='auto'):
    if W=='auto':
        W = np.ones(len(msa))

    L = len(msa[0])
    columns = np.transpose(msa)
    Z = np.sum(W)
    averages = np.zeros((L, 21))

    for a in range(21):
        averages[:, a] = np.dot(columns == a, W) / Z

    return averages
