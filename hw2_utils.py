import numpy as np
from scipy.optimize import minimize


# *** PLOTTING UTILITIES ***


def latex_float(v, pm=None, precision=3):
    """Converts a float (with optional error) into a nicely formatted LaTeX
    string for matplotlib labels."""
    fmt = "{:." + str(int(precision)) + "e}"
    base, exp = fmt.format(v).split("e")

    if pm is not None:
        fmt_f = fmt.replace("e", "f")
        pm_base = fmt_f.format(pm / 10 ** int(exp))
        return f"$({base} \\pm {pm_base}) \\times 10^" + "{" + f"{int(exp):d}" + "}$"

    else:
        return f"${base} \\times 10^" + "{" + f"{int(exp):d}" + "}$"


# *** MODELS ***


def null_model(parameters, t_i):
    """The null model: no sinusoid, just a DC level.

    parameters: [D]
    """
    D = parameters[0]
    return np.ones_like(t_i) * D


def one_signal_model(parameters, t_i):
    """The one signal model: DC level & a sinusoid with a 100-day period.

    parameters: [D, A1, A2]
    """
    D, A1, A2 = parameters[:3]
    T_0 = 100
    return D + A1 * np.sin(2 * np.pi * t_i / T_0) + A2 * np.cos(2 * np.pi * t_i / T_0)


def two_signal_model(parameters, t_i):
    """The two signal model. DC level, a 100-day sinusoid, and a second sinusoid
    of unknown frequency.

    parameters: [D, A1, A2, A3, A4, f]
    """
    D, A1, A2, A3, A4, f = parameters[:6]
    T_0 = 100
    sine_1 = A1 * np.sin(2 * np.pi * t_i / T_0) + A2 * np.cos(2 * np.pi * t_i / T_0)
    sine_2 = A3 * np.sin(2 * np.pi * f * t_i) + A4 * np.cos(2 * np.pi * f * t_i)
    return D + sine_1 + sine_2


# *** Log-likelihood functions ***


def logl(parameters, model, data):
    """Computes the log-likelihood function for the given model with given
    parameters on the dataset.

    data = [t_i, y_i, sigma_i]
    """
    t_i, y_i, sigma_i = data
    return -0.5 * np.sum(np.square((y_i - model(parameters, t_i)) / sigma_i))


def nll(*args):
    """Computes the negative log-likelihood function."""
    return -1 * logl(*args)


def logl_profile(params, prof_params, model, data):
    """Computes the log-likelihood, but splits the parameters to be profiled
    away from the other parameters (so that, e.g., they don't get optimized by
    the scipy optimizer). Note that the profile parameters are concatenated
    onto the parameters, so they must come at the end of the parameter list
    in the model!

    data: [t_i, y_i, sigma_i]
    """
    return logl([*params, *prof_params], model, data)


def nll_profile(*args):
    """Negative log-likelihood function with profile parameters."""
    return -1 * logl_profile(*args)


def max_likelihood_estimate(model, params, data, n_attempts=100):
    """Performs maximum-likelihood estimate by minimize the negative log-likelihood.

    model: the model to use
    params: initial search value of the params
    data: [t_i, y_i, sigma_i]

    Returns the best fit parameters and the log-likelihood value.
    """
    if not isinstance(params, np.ndarray):
        params = np.array(params)

    for _ in range(n_attempts):
        res = minimize(
            nll,
            x0=params,
            args=(model, data),
            method="Nelder-Mead",
        )
        if res.status != 0:
            continue
        break
    else:
        raise RuntimeError(
            "Max-likelihood estimate failed to converge on 100 consecutive attempts."
        )

    return res.x, -res.fun


def max_likelihood_estimate_profile(model, params, prof_params, data, n_attempts=100):
    """Performs maximum-likelihood estimate by minimize the negative log-likelihood.

    model: the model to use
    params: initial search value of the params
    prof_params: profile parameters, can be an empty list
    data: [t_i, y_i, sigma_i]

    Returns the best fit parameters and the log-likelihood value.
    """
    if not isinstance(params, np.ndarray):
        params = np.array(params)

    for _ in range(n_attempts):
        res = minimize(
            nll_profile,
            x0=params,
            args=(prof_params, model, data),
            method="Nelder-Mead",
        )
        if res.status != 0:
            continue
        break
    else:
        raise RuntimeError(
            "Max-likelihood estimate failed to converge on 100 consecutive attempts."
        )

    return res.x, -res.fun


# *** ERROR APPROXIMATION ***


def bootstrap(data, func, value, n_trials=100):
    """
    data: [t_i, y_i, sigma_i]
    func: a function of the resampled data
    value: the nominal value of `func`
    """
    N = len(data[0])
    rng = np.random.default_rng()

    if not isinstance(value, np.ndarray):
        value = np.array(value)
    value = value.reshape((1, -1))

    trials = np.zeros((n_trials, value.shape[1]))

    for i in range(n_trials):
        idx = rng.choice(N, size=(N,), replace=True)
        resampled = [x[idx] for x in data]
        trials[i] = func(resampled)

    return np.sqrt(np.average(np.square(trials - value), axis=0))
