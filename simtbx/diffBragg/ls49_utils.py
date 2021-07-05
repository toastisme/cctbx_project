from scipy.signal import savgol_filter
from simtbx.diffBragg import utils
import numpy as np
from scipy.special import voigt_profile
from scipy.interpolate import interp1d

import re
import os

RAYONIX_DIR = "/global/project/projectdirs/lcls/asmit/for_derek/rayonix_files_with_ts_refined_on_JF1M"

def get_tstamp(name):
    s = re.search("201805[0-9]+", name)
    return name[s.start(): s.end()]


def broaden_Fe_theo(shift=7.5):
    en, fp, fdp = np.loadtxt("Fe.dat").T
    en += shift
    grain = 0.05
    x = np.arange(-20,20+grain,grain)
    theo_en = np.arange(en[0], en[-1], grain)

    fp_grain = interp1d(en, fp)(theo_en)
    fdp_grain = interp1d(en, fdp)(theo_en)

    V = voigt_profile(x, 2, 1.61)
    fp_broad = np.convolve(V / V.sum(), fp_grain, 'same')
    fdp_broad = np.convolve(V / V.sum(), fdp_grain, 'same')

    return theo_en, fp_broad, fdp_broad


def fprime_ala_scherrell(en,fdp, theo_data=None, shift=7.5):
    if theo_data is not None:
        theo_en, fp_broad, fdp_broad = theo_data
    else:
        theo_en, fp_broad, fdp_broad = broaden_Fe_theo(shift=shift)

    fdp_broad_at_en = interp1d(theo_en, fdp_broad)(en)
    fp_broad_at_en = interp1d(theo_en, fp_broad)(en)
    delta_fdp = fdp - fdp_broad_at_en
    delta_fp = utils.f_prime(delta_fdp)
    fp = delta_fp + fp_broad_at_en
    return fp


