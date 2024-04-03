#!/usr/bin/python
""" Functions for defining robust weights for fitting problems """

import numpy as np

MIN_POSITIVE_SIGNAL = 0.0001

def weights_method_wls_m_est(data, pred_sig, design_matrix, leverages, idx, total_idx, last_robust, m_est="gm"):
    """
    M-estimator weights for weight-least-squares problem.
    """
    # check if M-estimator is valid (defined in this function)
    if m_est not in ["gm", "cauchy"]:
        raise ValueError("unknown M-estimator choice")

    # min 4 iters: (1) WLS (2) WLS with M-weights (3) clean OLS (3) clean WLS
    if total_idx < 4:
        raise ValueError("require 4+ iterations")

    p, N = design_matrix.shape[-1], data.shape[-1]
    if N <= p: raise ValueError("Fewer data points than parameters.")
    factor = 1.4826 * np.sqrt(N / (N - p))

    # cut-off for outlier detection
    cutoff = 3

    # handle potential zeros
    pred_sig[pred_sig < MIN_POSITIVE_SIGNAL] = MIN_POSITIVE_SIGNAL

    # calculate quantities needed for C and w
    log_pred_sig = np.log(pred_sig)
    residuals = data - pred_sig
    log_data = np.log(data)
    log_residuals = log_data - log_pred_sig
    z = pred_sig * log_residuals

    # IRLLS paper eq9 corrected (weights for log_residuals^2 are pred_sig^2)
    C = factor * np.median(np.abs(z - np.median(z, axis=-1)[..., None]), axis=-1)[..., None]
    C[C == 0] = np.nanmedian(C)  # C could be 0, if all signals = min_signal

    if m_est == "gm":
        w = (C/pred_sig)**2 / ((C/pred_sig)**2 + log_residuals**2)**2
    if m_est == "cauchy":
        w = C**2 / ((C/pred_sig)**2 + log_residuals**2)

    robust = None

    if idx == total_idx -1:  # OLS without outliers
        leverages[np.isclose(leverages, 1.0)] = 0.99  # avoids rare issues
        HAT_factor = np.sqrt(1 - leverages)
        cond_a = (residuals > +cutoff*C*HAT_factor) | (log_residuals < -cutoff*C*HAT_factor/pred_sig)
        #cond_b = (log_residuals > +cutoff*C*HAT_factor/pred_sig) | (residuals < -cutoff*C*HAT_factor)
        cond = cond_a #| cond_b
        robust = (cond == False)
        w[robust==0] = 0.0
        w[robust==1] = 1.0

    if idx == total_idx:  # WLS without outliers
        robust = last_robust
        w[robust==0] = 0.0
        w[robust==1] = pred_sig[robust==1]**2

    w[np.isinf(w)] = 0
    w[np.isnan(w)] = 0

    return w, robust


# define specific M-estimators
weights_method_wls_gm = lambda *args: weights_method_wls_m_est(*args, "gm")
weights_method_wls_cauchy = lambda *args: weights_method_wls_m_est(*args, "cauchy")


def weights_method_nlls_m_est(data, pred_sig, design_matrix, leverages, idx, total_idx, last_robust, m_est="gm"):
    """
    M-estimator weights for non-linear least squares problem.
    """
    # check if M-estimator is valid (defined in this function)
    if m_est not in ["gm", "cauchy"]:
        raise ValueError("unknown M-estimator choice")

    # min 3 iters: (1) NLLS (2) NLLS with M-weights (3) clean NLLS
    if total_idx < 3:
        raise ValueError("require 3+ iterations")

    p, N = design_matrix.shape[-1], data.shape[-1]
    if N <= p: raise ValueError("Fewer data points than parameters.")
    factor = 1.4826 * np.sqrt(N / (N - p))

    # cut-off for outlier detection
    cutoff = 3

    # handle potential zeros
    pred_sig[pred_sig < MIN_POSITIVE_SIGNAL] = MIN_POSITIVE_SIGNAL

    # calculate quantities needed for C and w
    log_pred_sig = np.log(pred_sig)
    residuals = data - pred_sig
    log_data = np.log(data)
    log_residuals = log_data - log_pred_sig
    z = pred_sig * log_residuals

    C = factor * np.median(np.abs(residuals - np.median(residuals)[..., None]), axis=-1)[..., None]
    C[C == 0] = np.nanmedian(C)  # C could be 0, if all signals = min_signal

    if m_est == "gm":
        w = C**2 / (C**2 + residuals**2)**2
    if m_est == "cauchy":
        w = C**2 / (C**2 + residuals**2)

    robust = None

    if idx == total_idx:  # the user should be able to specify things to do on the last iteration

        leverages[np.isclose(leverages, 1.0)] = 0.99  # avoids rare issues
        HAT_factor = np.sqrt(1 - leverages)
        cond_a = (residuals > +cutoff*C*HAT_factor) | (log_residuals < -cutoff*C*HAT_factor/pred_sig)
        # FIXME: for NLLS weighting, perhaps we should use *only* the +/- 3 sigma rule?
        #cond_b = (log_residuals > +cutoff*C*HAT_factor/pred_sig) | (residuals < -cutoff*C*HAT_factor)
        cond = cond_a #| cond_b
        robust = (cond == False)

        w[robust==0] = 0.0
        w[robust==1] = 1.0

    w[np.isinf(w)] = 0
    w[np.isnan(w)] = 0

    return w, robust


# define specific M-estimators
weights_method_nlls_gm = lambda *args: weights_method_nlls_m_est(*args, "gm")
weights_method_nlls_cauchy = lambda *args: weights_method_nlls_m_est(*args, "cauchy")

