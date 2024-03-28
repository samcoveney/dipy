#!/usr/bin/python
""" Functions for defining robust weights for fitting problems """

import numpy as np

MIN_POSITIVE_SIGNAL = 0.0001

# NOTE: this is the weights method specifically for WLS! May need to rename...
def weights_method_wls_m_est(data, pred_sig, design_matrix, leverages, idx, total_idx, last_robust, m_est="gmm"):
    """
    M-estimator weights for weight-least-squares DTI problem.
    """
    # NOTE: would be preferable to not need design_matrix and leverages... figure this out
    # NOTE: could provide 'adjacency' optionally, in which case the C estimates could be pooled
    #       in which cases, we really want a function that returns a lambda function, given the appropriate arguments to the first function

    cutoff = 3 # could be an input

    # handle potential zeros
    pred_sig[pred_sig < MIN_POSITIVE_SIGNAL] = MIN_POSITIVE_SIGNAL
    #data[data < MIN_POSITIVE_SIGNAL] = MIN_POSITIVE_SIGNAL # NOTE: should already have been filtered

    # calculate quantities needed for C and w
    #log_pred_sig = np.dot(design_matrix, params.T).T
    #pred_sig = np.exp(log_pred_sig)
    log_pred_sig = np.log(pred_sig) # NOTE: inefficient to recalc but who cares
    residuals = data - pred_sig
    log_data = np.log(data)  # Waste to recalc, but I want to hand things to 'weight_method'
    log_residuals = log_data - log_pred_sig
    z = pred_sig * log_residuals

    p = design_matrix.shape[-1]
    N = data.shape[-1]
    if N <= p: raise ValueError("Fewer data points than parameters.")
    factor = 1.4826 * np.sqrt(N / (N - p))

    C = factor * np.median(np.abs(z - np.median(z, axis=-1)[..., None]), axis=-1)[..., None]  # NOTE: IRLS eq9 correction

    # NOTE: C can be zero! If all signals were set to min_signal, for example
    C[C == 0] = np.nanmedian(C)  # does this make sense if using a mask?
    #print(np.nanmedian(C))
    #print(np.sum(np.isinf(C)))

    if m_est == "gm":
        w = (C/pred_sig)**2 / ((C/pred_sig)**2 + log_residuals**2)**2
    if m_est == "cauchy":
        w = 1 / ((C/pred_sig)**2 + log_residuals**2)  # NOTE: double check these from first principles for WLS for DTI

    robust = None

    if idx >= total_idx - 1:  # the user should be able to specify things to do on the last iteration

        if idx < total_idx:  # OLS without outliers
            leverages[np.isclose(leverages, 1.0)] = 0.9999  # NOTE: maybe still needed
            HAT_factor = np.sqrt(1 - leverages)
            #HAT_factor = 1  # NOTE: hack while testing!!! Needs putting back to normal
            cond_a = (residuals > +cutoff*C*HAT_factor) | (log_residuals < -cutoff*C*HAT_factor/pred_sig)
            #cond_b = (log_residuals > +cutoff*C*HAT_factor/pred_sig) | (residuals < -cutoff*C*HAT_factor)
            cond = cond_a #| cond_b
            robust = (cond == False)
            w[robust==0] = 0.0
            w[robust==1] = 1.0
        else:  # WLS without outliers
            robust = last_robust
            w[robust==0] = 0.0
            w[robust==1] = pred_sig[robust==1]**2

#    import IPython as ipy
#    ipy.embed()
    print("inf:", np.isinf(w).sum())
    print("nan:", np.isnan(w).sum())  # getting a lot of NaN in the weights if not using a mask! Why would that be?
    w[np.isinf(w)] = 0
    w[np.isnan(w)] = 0

    return w, robust  # NOTE: could return estimate of noise level, that is a very general thing to want to return for robust fitting


# define specific M-estimators
weights_method_wls_gm = lambda *args: weights_method_wls_m_est(*args, "gm")
weights_method_wls_cauchy = lambda *args: weights_method_wls_m_est(*args, "cauchy")


# NOTE: this is the weights method specifically for WLS! May need to rename...
def weights_method_nlls_m_est(data, pred_sig, design_matrix, leverages, idx, total_idx, last_robust, m_est="gmm"):
    """
    M-estimator weights for weight-least-squares DTI problem.
    """
    # NOTE: would be preferable to not need design_matrix and leverages... figure this out
    # NOTE: could provide 'adjacency' optionally, in which case the C estimates could be pooled
    #       in which cases, we really want a function that returns a lambda function, given the appropriate arguments to the first function

    cutoff = 3 # could be an input

    # handle potential zeros
    pred_sig[pred_sig < MIN_POSITIVE_SIGNAL] = MIN_POSITIVE_SIGNAL

    # calculate quantities needed for C and w
    #log_pred_sig = np.dot(design_matrix, params.T).T
    #pred_sig = np.exp(log_pred_sig)
    log_pred_sig = np.log(pred_sig) # NOTE: inefficient to recalc but who cares
    residuals = data - pred_sig
    log_data = np.log(data)  # Waste to recalc, but I want to hand things to 'weight_method'
    log_residuals = log_data - log_pred_sig
    z = pred_sig * log_residuals

    p = design_matrix.shape[-1]
    N = data.shape[-1]
    if N <= p: raise ValueError("Fewer data points than parameters.")
    factor = 1.4826 * np.sqrt(N / (N - p))

    C = factor * np.median(np.abs(residuals - np.median(residuals)[..., None]), axis=-1)[..., None]

    if m_est == "gm":
        w = C**2 / (C**2 + residuals**2)**2
    if m_est == "cauchy":
        w = 1 / (C**2 + residuals**2)  # NOTE: double check these from first principles for WLS for DTI

    robust = None

    if idx == total_idx:  # the user should be able to specify things to do on the last iteration

        # NOTE: not sure we want to run this in both the second to last and the last...
        leverages[np.isclose(leverages, 1.0)] = 0.9999
        #HAT_factor = np.sqrt(1 - leverages)
        HAT_factor = 1  # NOTE: hack while testing!!! Needs putting back to normal
        cond_a = (residuals > +cutoff*C*HAT_factor) | (log_residuals < -cutoff*C*HAT_factor/pred_sig)
        #cond_b = (log_residuals > +cutoff*C*HAT_factor/pred_sig) | (residuals < -cutoff*C*HAT_factor)
        cond = cond_a #| cond_b
        robust = (cond == False)

        w[robust==0] = 0.0
        w[robust==1] = 1.0

    return w, robust  # NOTE: could return estimate of noise level, that is a very general thing to want to return for robust fitting


# define specific M-estimators
weights_method_nlls_gm = lambda *args: weights_method_nlls_m_est(*args, "gm")
weights_method_nlls_cauchy = lambda *args: weights_method_nlls_m_est(*args, "cauchy")

