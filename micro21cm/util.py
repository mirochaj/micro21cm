"""

util.py

Authors: Jordan Mirocha and Julian B. Munoz
Affiliation: McGill University and Harvard-Smithsonian Center for Astrophysics
Created on: Wed 16 Dec 2020 16:16:41 EST

Description:

"""

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import griddata
from scipy.fft import fftn as scipy_fftn
from scipy.fft import ifftn as scipy_ifftn

try:
    import progressbar
    pb = True
except ImportError:
    pb = False

labels = \
{
 'R': r'$R / [h^{-1} \ \mathrm{Mpc}]$',
 'k': r'$k / [h \ \mathrm{cMpc}^{-1}]$',
 'pofk': r'$P(k)$',
 'cf21': r'$\xi_{21}(R)$',
 'delta_sq': r'$\overline{\delta T_b}^2 \Delta^2(k) \ [\mathrm{mK}^2]$',
 'delta_sq_long': r'$\overline{\delta T_b}^2 k^3 \left(\frac{P(k)}{2\pi^2}\right) \ [\mathrm{mK}^2]$',
 'delta_sq_xx': r'$\Delta_{xx}^2(k)$',
 'delta_sq_xd': r'$k^3 |P_{x\delta}| / 2 \pi^2$',
 'Ts': r'$T_S / \mathrm{K}$',
 'Q': r'$Q$',
 'Q_ion': r'$Q_{\mathrm{ion}}$',
 'Q_hot': r'$Q_{\mathrm{hot}}$',
 'xHI': r'$x_{\mathrm{HI}}$',
 'bsd': r'$dn /d\log R$',
 'bsd_normed': r'$Q^{-1} V dn/d\log R$',
}

def get_cf_from_ps(R, f_ps, kmin=1e-4, kmax=5000., rtol=1e-5, atol=1e-5):

    cf = np.zeros_like(R)
    for i, RR in enumerate(R):

        # Split the integral into an easy part and a hard part
        kcrit = 1. / RR

        # Re-normalize integrand to help integration
        norm = 1. / f_ps(kmax)

        # Leave sin(k*R) out -- that's the 'weight' for scipy.
        integrand = lambda kk: norm * 4 * np.pi * kk**2 * f_ps(kk) / kk / RR
        integrand_full = lambda kk: integrand(kk) * np.sin(kk * RR)

        # Do the easy part of the integral
        cf[i] = quad(integrand_full, kmin, kcrit,
            epsrel=rtol, epsabs=atol, limit=10000, full_output=1)[0] / norm

        # Do the hard part of the integral using Clenshaw-Curtis integration
        cf[i] += quad(integrand, kcrit, kmax,
            epsrel=rtol, epsabs=atol, limit=10000, full_output=1,
            weight='sin', wvar=RR)[0] / norm

    # Our FT convention
    cf /= (2 * np.pi)**3

    return cf

def get_ps_from_cf(k, f_cf, Rmin=1e-2, Rmax=1e3, rtol=1e-5, atol=1e-5):

    ps = np.zeros_like(k)
    for i, kk in enumerate(k):

        # Split the integral into an easy part and a hard part
        Rcrit = 1. / kk

        # Re-normalize integrand to help integration
        norm = 1. / f_cf(Rmax)

        # Leave sin(k*R) out -- that's the 'weight' for scipy.
        integrand = lambda RR: norm * 4 * np.pi * RR**2 * f_cf(RR) / kk / RR
        integrand_full = lambda RR: integrand(RR) * np.sin(kk * RR)

        # Do the easy part of the integral
        ps[i] = quad(integrand_full, Rmin, Rcrit,
            epsrel=rtol, epsabs=atol, limit=10000, full_output=1)[0] / norm

        # Do the hard part of the integral using Clenshaw-Curtis integration
        ps[i] += quad(integrand, Rcrit, Rmax,
            epsrel=rtol, epsabs=atol, limit=10000, full_output=1,
            weight='sin', wvar=kk)[0] / norm

    return ps

def get_filter(box, R=10, kernel='tophat'):
    _fil_ = np.zeros_like(box)
    if kernel == 'tophat':
        dim = box.shape[0]
        # Bin edges
        _xe = _ye = _ze = np.arange(0, dim+1, 1)
        _xc = _yc = _zc = bin_e2c(_xe)

        _xx, _yy, _zz = np.meshgrid(_xc, _yc, _zc, indexing="ij")

        x0 = y0 = z0 = int(0.5 * dim)

        _RR = np.sqrt((_xx - x0)**2 + (_yy - y0)**2 + (_zz - z0)**2)

        ok = np.abs(_RR) < R

        _fil_[ok==1] = 1. / float(ok.sum())

    else:
        raise NotImplemented('help')

    return _fil_

def smooth_box(box, R=2, kernel='tophat', periodic=False):
    """
    Smooth a box with some kernel.
    """

    ##
    # Setup smoothing filter first
    _fil_ = get_filter(box, R=R, kernel=kernel)

    _box = scipy_fftn(box)
    _fil = scipy_fftn(_fil_)

    return scipy_ifftn(_box * _fil)

class ProgressBar(object):
    def __init__(self, maxval, name='micro21cm', use=True, width=80):
        self.maxval = maxval
        self.use = use
        self.width = width

        self.has_pb = False
        if pb and use:
            self.widget = ["{!s}: ".format(name), progressbar.Percentage(),
                ' ', \
              progressbar.Bar(marker='#'), ' ', \
              progressbar.ETA(), ' ']

    def start(self):
        if pb and self.use:
            self.pbar = progressbar.ProgressBar(widgets=self.widget,
                max_value=self.maxval, redirect_stdout=False,
                term_width=self.width).start()
            self.has_pb = True

    def update(self, value):
        if self.has_pb:
            self.pbar.update(value)

    def finish(self):
        if self.has_pb:
            self.pbar.finish()




def get_cmd_line_kwargs(argv):

    cmd_line_kwargs = {}

    for arg in argv[1:]:
        try:
            pre, post = arg.split('=')
        except ValueError:
            # To deal with parameter values that have an '=' in them.
            pre = arg[0:arg.find('=')]
            post = arg[arg.find('=')+1:]

        # Need to do some type-casting
        if post.isdigit():
            cmd_line_kwargs[pre] = int(post)
        elif post.isalpha():
            if post == 'None':
                cmd_line_kwargs[pre] = None
            elif post in ['True', 'False']:
                cmd_line_kwargs[pre] = True if post == 'True' else False
            else:
                cmd_line_kwargs[pre] = str(post)
        elif post[0] == '[':
            vals = post[1:-1].split(',')
            cmd_line_kwargs[pre] = np.array([float(val) for val in vals])
        else:
            try:
                cmd_line_kwargs[pre] = float(post)
            except ValueError:
                # strings with underscores will return False from isalpha
                cmd_line_kwargs[pre] = str(post)

    return cmd_line_kwargs

def split_by_sign(x, y):
    """
    Split apart an array into its positive and negative chunks.
    """

    splitter = np.diff(np.sign(y))

    if np.all(splitter == 0):
        ych = [y]
        xch = [x]
    else:
        splits = np.atleast_1d(np.argwhere(splitter != 0).squeeze()) + 1
        ych = np.split(y, splits)
        xch = np.split(x, splits)

    return xch, ych

def CTfit(z):
#fit to CT=\delta T_g/\delta_matter, error below 3% for z=6-50
    return (0.58 - 0.005*(z-10.))


def Tgadiabaticfit(z):
#fit to Tgas(z) adiabatically cooling in LCDM, in K. Good to 3% in z=6-50 (for exponent=2 good within 10%)
    return 9.5 * ((1+z)/(21.))**1.95

def bin_e2c(bins):
    """
    Convert bin edges to bin centers.
    """
    dx = np.diff(bins)
    assert np.allclose(np.diff(dx), 0), "Binning is non-uniform!"
    dx = dx[0]

    return 0.5 * (bins[1:] + bins[:-1])

def bin_c2e(bins):
    """
    Convert bin centers to bin edges.
    """
    dx = np.diff(bins)
    assert np.allclose(np.diff(dx), 0), "Binning is non-uniform!"
    dx = dx[0]

    return np.concatenate(([bins[0] - 0.5 * dx], bins + 0.5 * dx))

def get_error_2d(x, y, z, bins, nu=[0.95, 0.68], weights=None, method='raw'):
    """
    Find 2-D contour given discrete samples of posterior distribution.

    Parameters
    ----------
    x : np.ndarray
        Array of samples in x.
    y : np.ndarray
        Array of samples in y.
    bins : np.ndarray, (2, Nsamples)

    method : str
        'raw', 'nearest', 'linear', 'cubic'


    """

    if method == 'raw':
        nu, levels = _error_2D_crude(z, nu=nu)
    else:

        # Interpolate onto new grid
        grid_x, grid_y = np.meshgrid(bins[0], bins[1])
        points = np.array([x, y]).T
        values = z

        grid = griddata(points, z, (grid_x, grid_y), method=method)

        # Mask out garbage points
        mask = np.zeros_like(grid, dtype='bool')
        mask[np.isinf(grid)] = 1
        mask[np.isnan(grid)] = 1
        grid[mask] = 0

        nu, levels = _error_2D_crude(grid, nu=nu)

    return nu, levels

def _error_2D_crude(L, nu=[0.95, 0.68]):
    """
    Integrate outward at "constant water level" to determine proper
    2-D marginalized confidence regions.

    ..note:: This is fairly crude -- the "coarse-ness" of the resulting
        PDFs will depend a lot on the binning.

    Parameters
    ----------
    L : np.ndarray
        Grid of likelihoods.
    nu : float, list
        Confidence intervals of interest.

    Returns
    -------
    List of contour values (relative to maximum likelihood) corresponding
    to the confidence region bounds specified in the "nu" parameter,
    in order of decreasing nu.
    """

    if type(nu) in [int, float]:
        nu = np.array([nu])

    # Put nu-values in ascending order
    if not np.all(np.diff(nu) > 0):
        nu = nu[-1::-1]

    peak = float(L.max())
    tot = float(L.sum())

    # Counts per bin in descending order
    Ldesc = np.sort(L.ravel())[-1::-1]

    Lencl_prev = 0.0

    # Will correspond to whatever contour we're on
    j = 0

    # Some preliminaries
    contours = [1.0]
    Lencl_running = []

    # Iterate from high likelihood to low
    for i in range(1, Ldesc.size):

        # How much area (fractional) is contained in bins at or above the current level?
        Lencl_now = L[L >= Ldesc[i]].sum() / tot

        # Keep running list of enclosed (integrated) likelihoods
        Lencl_running.append(Lencl_now)

        # What contour are we on?
        Lnow = Ldesc[i]

        # Haven't hit next contour yet
        if Lencl_now < nu[j]:
            pass
        # Just passed a contour
        else:

            # Interpolate to find contour more precisely
            Linterp = np.interp(nu[j], [Lencl_prev, Lencl_now],
                [Ldesc[i-1], Ldesc[i]])

            # Save relative to peak
            contours.append(Linterp / peak)

            j += 1

            if j == len(nu):
                break

        Lencl_prev = Lencl_now

    # Return values that match up to inputs
    return nu[-1::-1], np.array(contours[-1::-1])
