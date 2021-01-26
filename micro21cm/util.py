"""

util.py

Author: Jordan Mirocha
Affiliation: McGill University
Created on: Wed 16 Dec 2020 16:16:41 EST

Description:

"""

import numpy as np
from scipy.integrate import quad

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
 'delta_sq': r'$\Delta^2(k) \ [\mathrm{mK}^2]$',
 'delta_sq_long': r'$k^3 \left(\frac{P(k)}{2\pi^2}\right) \ [\mathrm{mK}^2]$',
 'Ts': r'$T_S / \mathrm{K}$',
 'Q': r'$Q$',
 'Q_ion': r'$Q_{\mathrm{ion}}$',
 'Q_hot': r'$Q_{\mathrm{hot}}$',
 'xHI': r'$x_{\mathrm{HI}}$',
}

def get_cf_from_ps(R, f_ps, kmin=1e-3, kmax=5000., rtol=1e-5, atol=1e-5):

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
