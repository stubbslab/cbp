import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
from tqdm import tqdm
from scipy.optimize import curve_fit
import scipy as sp
from scipy import interpolate
import pwlf
from astropy.io import fits
from astropy import constants as const
from astropy import units as u
from scipy.signal import savgol_filter

def set_dataset_from_fitsheader(dataset, header):
    dataset.expnum = header["EXPNUM"]
    if "LASERNPULSES" in header:
        dataset.npulses = header["lasernpulses"]
    else:
        dataset.npulses = 0
    if "LASERNBURST" in header:
        dataset.nbursts = header["lasernburst"]
    else:
        dataset.nbursts = 0
    if "LASERWAVELENGTH" in header:
        dataset.wavelength = header["laserwavelength"]
    else:
        dataset.wavelength = 0
    if "LASERQSW" in header:
        dataset.laserqsw = header["LASERQSW"]
    else:
        dataset.laserqsw = "MAX"
    if "laserwheelfilter" in header:
        dataset.laserwheelfilter = header["laserwheelfilter"]
    else:
        dataset.laserwheelfilter = "EMPTY"
    if "filterwheelfilter" in header:
        dataset.filterwheelfilter = header["filterwheelfilter"]
    else:
        dataset.filterwheelfilter = "EMPTY"


def groupBurstStartGuesses(ungrouped_guess_indeces, min_sep=20, verbose=0):
    """
    authors: Sasha Brownsberger (sashab@alumni.stanford.edu)

    Modifications:
    2021/07/04 - 2021/07/14: First written.

    Description:
    Takes an array of indeces of an array (integers) and sorts them into "groups" of nearby indeces.  For an
        index to be placed in a group, it must be within min_sep distance from any other component of that group.

    This function is used to separate distinct peaks above some detection threshold.
    """
    all_guesses_grouped = 0
    current_index = 0
    current_group = []
    guess_groups = []
    while not (all_guesses_grouped):
        if len(current_group) == 0:
            current_group = [ungrouped_guess_indeces[current_index]]
        elif ungrouped_guess_indeces[current_index] - current_group[-1] < min_sep:
            current_group = current_group + [ungrouped_guess_indeces[current_index]]
        else:
            guess_groups = guess_groups + [current_group]
            current_group = [ungrouped_guess_indeces[current_index]]
        if current_index == len(ungrouped_guess_indeces) - 1:
            guess_groups = guess_groups + [current_group]
            all_guesses_grouped = 1
        else:
            current_index = current_index + 1
    if verbose:
        print('guess_groups = ' + str(guess_groups))
    return guess_groups


def findTransitions(arr, n_target_transitions, max_n_sig_to_identify=14, min_n_sig_to_identify=1, pos_or_neg=1,
                    verbose=0, max_iterations=20, min_data_sep=10, show_process=0):
    """
    authors: Sasha Brownsberger (sashab@alumni.stanford.edu)

    Modifications:
    2021/07/04 - 2021/07/14: First written.

    Description:
    Takes an array of indeces of an array (integers) and sorts them into "groups" of nearby indeces.  For an
        index to be placed in a group, it must be within min_sep distance from any other component of that group.

    This function is used to separate distinct peaks above some detection threshold.
    """
    if verbose:
        print('Finding transitions...')
    transitions = []
    low_n_sig = min_n_sig_to_identify
    high_n_sig = max_n_sig_to_identify
    arr_med = np.median(arr)
    arr_std = np.std(arr)
    if verbose:
        print('[arr_med, arr_std] = ' + str([arr_med, arr_std]))
    if verbose:
        print('n_target_transitions = ' + str(n_target_transitions))
    iter_number = 0
    while len(transitions) != n_target_transitions and iter_number <= max_iterations:
        current_n_sig = (high_n_sig + low_n_sig) / 2
        if show_process:
            f, axarr = plt.subplots(1, 1)
            axarr.plot(arr, c='k', alpha=0.75)
            axarr.axhline(arr_med + arr_std * current_n_sig, color='cyan')
            axarr.axhline(arr_med - arr_std * current_n_sig, color='cyan')
            plt.draw()
            plt.pause(0.5)
        if pos_or_neg == 1:
            transition_indeces = [i for i in range(len(arr)) if arr[i] > arr_med + arr_std * current_n_sig]
        else:
            transition_indeces = [i for i in range(len(arr)) if arr[i] < arr_med - arr_std * current_n_sig]
        if verbose:
            print('transition_indeces = ' + str(transition_indeces))
            print('current_n_sig = ' + str(current_n_sig))
        if len(transition_indeces) >= n_target_transitions:
            # print ('turn_on_guess_indeces = ' + str(turn_on_guess_indeces))
            transitions = groupBurstStartGuesses(transition_indeces, verbose=verbose, min_sep=min_data_sep)
        else:
            transitions = [arr[index] for index in transition_indeces]
        if verbose:
            print('transitions = ' + str(transitions))
        if len(transitions) > n_target_transitions:
            low_n_sig = current_n_sig
        if len(transitions) < n_target_transitions:
            high_n_sig = current_n_sig
        iter_number = iter_number + 1
        if len(transitions) != n_target_transitions and iter_number > max_iterations:
            current_n_sig = -1
            transitions = [-1 for i in range(n_target_transitions)]
            if verbose:
                print(f'Failed to find transitions before exceeding max allowed iterations {max_iterations}.')
        if show_process:
            plt.close('all')
    if verbose:
        print('Returning: ' + str([current_n_sig, transitions]))
    return [current_n_sig, transitions]


def calculate_npulses(wl):
    """
    Calculate the number of pulses to maximize and flatten the signal for each wavelength in nanometer
    """

    # Load a previous file to find the turnout point between the two regimes of the laser

    charges_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SC_photocurrents_over_PD_charges.csv')
    data_charges = np.loadtxt(charges_path, skiprows=1, delimiter=",")
    l_wl, SC_charges, PD_charges = data_charges.T

    # Interpolate to fill with potentials empty values
    l_wl_full = np.arange(l_wl[0], l_wl[-1] + 1)
    f_PD_charges = sp.interpolate.interp1d(l_wl, PD_charges)
    PD_charges = f_PD_charges(l_wl_full)

    f_SC_charges = sp.interpolate.interp1d(l_wl, SC_charges)
    SC_charges = f_SC_charges(l_wl_full)

    PD_charges_per_pulse = PD_charges / 1000

    diff = []
    for i in range(len(PD_charges)):
        diff.append(PD_charges[i] - PD_charges[i - 1])

    turnout = np.array(np.where(diff == np.max(diff))).squeeze()
    # Calculate the mean before the turnout point to flatten the signal with this mean
    charges_mean = np.mean(PD_charges[:turnout])

    # Calculate the number of pulses for the wavelength wanted
    npulses = charges_mean / PD_charges_per_pulse[l_wl_full == wl]

    if npulses <= 1:
        npulses = 1
    elif npulses >= 1000:
        npulses = 1000

    return int(npulses)


def estimate_noise(x, y, length=20):
    """
    Estimate the RMS after subtraction of line fit on first data points, length is defined by legnth parameter.

    Parameters
    ----------
    x: array_like
        The abscissa data.
    y: array_like
        The ordinate data.
    length: int, optional
        Number of data points to make the estimate (default: 20).

    Returns
    -------
    noise: float
        The RMS after line fit subtraction.
    """
    pval = np.polyfit(x[:length], y[:length], deg=1)
    res = y[:length] - np.polyval(pval, x[:length])
    return np.std(res)


def fit_line_pieces(x, y, err=None, delta_chi2=10, min_len=2, deg=1):
    """Fit pieces of lines on piecewise linear data, starting from the left with a minimum of 2 points.
    The delta_chi2 argument set the level to break the fitting loop and start a new line.

    Parameters
    ----------
    x: array_like
        The abscissa data.
    y: array_like
        The ordinate data.
    err: array_like, optional
        The uncertainties on data (default: None).
    delta_chi2: float, optional
        The threshold value from which a new line is fitted to next data points (default: 10).

    Returns
    -------
    pvals: array_like
        List of linear coefficient for each piece of line.
    indices: array_like
        List of the indices for each piece of line.

    """
    pvals = []
    indices_all = []
    startpoint = 0
    endpoint = 0
    counter = 0
    if type(min_len) is int:
        min_len = [min_len]
    while endpoint < len(x) - 1 and counter <= len(x):
        chi2 = 0
        # start a line with at least three points
        if counter > len(min_len) - 1:
            start_index = min(startpoint + np.min(min_len), x.size - 2)
        else:
            start_index = min(startpoint + min_len[counter], x.size - 2)
        for index in range(start_index, len(x)):
            indices_tmp = np.arange(startpoint, index, dtype=int)
            pval_tmp = np.polyfit(x[indices_tmp], y[indices_tmp], deg=deg)
            # compute the chi square
            if err is None:
                stand_in_err = np.std(y[startpoint:index])
                # print ('[startpoint, index] = ' + str([startpoint, index]))
                # print ('stand_in_err = ' + str(stand_in_err))
                chi2_tmp = np.sum(((y[indices_tmp] - np.polyval(pval_tmp, x[indices_tmp])) / stand_in_err) ** 2)
            else:
                chi2_tmp = np.sum(((y[indices_tmp] - np.polyval(pval_tmp, x[indices_tmp])) / err[indices_tmp]) ** 2)
            # print ('[chi2, chi2_tmp, len(indices_tmp), (chi2_tmp - chi2) / len(indices_tmp), delta_chi2] = ' + str([chi2, chi2_tmp, len(indices_tmp), (chi2_tmp - chi2) / len(indices_tmp), delta_chi2] ))
            # initialisation
            if chi2 == 0:
                chi2 = chi2_tmp
            # check the new chi square
            if abs(chi2_tmp - chi2) / len(indices_tmp) < delta_chi2 and index < len(x) - 1:
                # continue
                chi2 = chi2_tmp
                pval = pval_tmp
                indices = indices_tmp
            else:
                # break the fit
                if index == len(x) - 1:
                    # last point: keep the current fit
                    indices_tmp = np.append(indices_tmp, index)
                    indices_all.append(indices_tmp)
                    pvals.append(pval_tmp)
                else:
                    # save previous fit
                    pvals.append(pval)
                    indices_all.append(indices)
                endpoint = indices_all[-1][-1]
                startpoint = endpoint + 1
                break
        counter += 1
    return pvals, indices_all


def get_photodiode_total_charge(time, charge, delta_chi2=10, err=None, plot=False, min_len=2, deg=1):
    r"""Estimate the total charge accumulated in the charge sequence and
    the leakage currents at beginning and end of the sequence.

    Fit pieces of lines on charge data, starting from the left with a minimum of 2 points.
    The delta_chi2 argument set the level to break the fitting loop and start a new line.

    This measures the leakage current at the beginning of the sequence $i_K^{(1)}$ and at the end $i_K^{(2)}$.
    The beginning of the charge $t_1$ is considered to be the last point of the first step while the end of the
    charge $t_2$ is defined as the first point of the last step. The accumulated charge $\Delta q_{\rm mes}$ is then
        $$\Delta q_{\rm mes} = q(t_2) - q(t_1)$$
    But because of the leakage current, charges are lost all along the process at a constant rate defined by the
    leakage current. As both leakage currents $i_K^{(1)}$ and $i_K^{(2)}$ varies with the accumulated charge but
    are nearly the same, we define the Keithley capacitor charge as:
        $$\Delta q = \Delta q_{\rm mes}- \frac{1}{2} \left( i_K^{(1)}+i_K^{(2)}\right) (t_2-t_1)$$
    If this correction is not done, we differences between the runs on $\Delta q_{\rm mes}$ because leakage
    currents are not the same along the data acquisition.
    With the correction, the charge measurement is reproducible.

    The error budget is the following:
        - $\Delta q_{\rm mes}$ can be underestimated if the true $t_1$ is a time step $\delta t$ after $t_1$ and
        the true $t_2$ is $\delta t$ before
    $$q(t_2 - \delta t) - q(t_1 + \delta t) \approx \Delta q_{\rm mes} - i_K^{(2)} \delta t - i_K^{(1)} \delta t$$
    $$ \sigma_{\Delta q_{\rm mes}} = - \delta t \left(i_K^{(2)} + i_K^{(1)} \right) $$
        - a first uncertainty on the charge correction comes from the estimate of $t_2$ and $t_1$:
    $$ \frac{1}{2} \left(i_K^{(1)}+i_K^{(2)}\right) (t_2-\delta t -t_1 - \delta t ) = \frac{1}{2} \left(i_K^{(1)}
       +i_K^{(2)}\right) (t_2-t_1) - \delta t \left(i_K^{(2)} + i_K^{(1)} \right)$$
    $$ \sigma_{\delta q^{(t)}} = - \delta t \left(i_K^{(2)} + i_K^{(1)} \right) $$
        - a first uncertainty on the charge correction comes from the estimate of the leakage currents:
    $$ \sigma_{\delta q^{(i_K)}} = \frac{1}{2} \left \vert i_K^{(1)}-i_K^{(2)}\right\vert (t_2-t_1) $$
    The final uncertainty is estimated as:
    $$\sigma_{\Delta q} =  \left\vert\sigma_{\Delta q_{\rm mes}}\right\vert + \left\vert\sigma_{\delta q^{(t)}}
       \right\vert +  \left\vert\sigma_{\delta q^{(i_K)}} \right\vert$$

    Parameters
    ----------
    time: array_like
        Array of timestamps in seconds.
    charge: array_like
        Array of charges in Coulomb.
    err: array_like, optional
        Array of charge uncertainties in Coulomb (default: None).
    delta_chi2: float, optional
        The threshold value from which a new line is fitted to next data points (default: 10).
    plot: bool, optional
        If True, plot the results (default: False).

    Returns
    -------
    total_charge: float
        The total charge in Keithley units.
    total_charge_err: float
        Uncertainty on the total charge in Keithley units.
    i_k1: float
        Leakage current at the beginning of the sequence in Keithley units.
    i_k2: float
        Leakage current at the end of the sequence in Keithley units.

    """
    # fit lines at beginning and end of the charge sequence
    pvals, indices = fit_line_pieces(time, charge, delta_chi2=delta_chi2, err=err, min_len=min_len, deg=deg)
    pval1, pval2 = pvals[0], pvals[-1]
    indices1, indices2 = indices[0], indices[-1]
    i_k1 = pval1[0]
    i_k2 = pval2[0]
    # charges at the beginning and end of charge
    q1 = np.polyval(pval1, time[indices1[-1]])  # take charge at last point of first step
    q2 = np.polyval(pval2, time[indices2[0]])  # take charge at first point of first step
    total_charge = q2 - q1
    # times of beginning and end of charge
    t1 = time[indices1[-1]]
    t2 = time[indices2[0]]
    # charge correction by leakage currents
    total_charge -= 0.5 * (i_k1 + i_k2) * (t2 - t1)
    # uncertainties
    delta_t = np.mean(np.gradient(time))
    total_charge_err = abs(2 * delta_t * (i_k1 + i_k2)) + 0.5 * abs(i_k1 - i_k2) * (t2 - t1)
    res = np.copy(charge)
    for k, pval in enumerate(pvals):
        res[indices[k]] -= np.polyval(pval, time[indices[k]])
    total_charge_err = np.sqrt(total_charge_err ** 2 + np.var(res))
    if plot:
        fig, ax = plt.subplots(2, 1, sharex="all")
        ax[0].plot(time, charge, "r+")
        ax[0].plot(time, np.polyval(pval1, time))
        ax[0].plot(time, np.polyval(pval2, time))
        for pval in pvals:
            ax[0].plot(time, np.polyval(pval, time))
        if np.sign(charge[-1]) > 0:
            ax[0].set_ylim(0.8 * np.min(charge), 1.2 * np.max(charge))
        else:
            ax[0].set_ylim(1.2 * np.min(charge), 0.8 * np.max(charge))
        for ind in indices:
            ax[0].axvline(time[ind[-1]], color="k", linestyle="-.")
            ax[0].axvline(time[ind[0]], color="k", linestyle="-.")
        ax[0].set_title(f"Total charge: {total_charge:.4g} +/- {total_charge_err:.4g} [C]")
        ax[0].axhline(q1, color="k", linestyle="--")
        ax[0].axhline(q2, color="k", linestyle="--")
        ax[0].axvline(t1, color="k", linestyle="--")
        ax[0].axvline(t2, color="k", linestyle="--")
        ax[0].set_xlabel("Time [ms]")
        ax[0].set_ylabel(f"PD Charge [C]")
        ax[0].grid()
        ax[1].plot(time, res, "r+")
        ax[1].set_xlabel("Time [ms]")
        ax[1].set_ylabel(f"Difference [C]")
        ax[1].grid()
        fig.tight_layout()
        plt.show()
    return total_charge, total_charge_err, i_k1, i_k2, pvals, indices


def get_solarcell_total_charge(time, charge, time_breaks, err=None, plot=False):
    r"""Estimate the total charge accumulated in the charge sequence and
    the leakage currents at beginning and end of the sequence for the solar cell.

    A continuous piecewise linear function is fitted on the data. The break times must be given in seconds.

    Parameters
    ----------
    time: array_like
        Array of timestamps in seconds.
    charge: array_like
        Array of charges in Coulomb.
    time_breaks: array_like
        List of guessed instant at which a break is expected in seconds.
    err: array_like, optional
        Array of charge uncertainties in Coulomb (default: None).
    plot: bool, optional
        If True, plot the results (default: False).

    Returns
    -------
    total_charge: float
        The total charge in Keithley units.
    total_charge_err: float
        Uncertainty on the total charge in Keithley units.
    i_k1: float
        Leakage current at the beginning of the sequence in Keithley units.
    i_k2: float
        Leakage current at the end of the sequence in Keithley units.

    """
    # fit lines at beginning and end of the charge sequence
    # pvals, indices = fit_line_pieces(time, charge, delta_chi2=delta_chi2, err=err, min_len=min_len, deg=deg)
    # print("first pval", len(pvals), pvals)
    # compute durations of dark and signal parts
    # bounds = [[0.9 * t, 1.1 * t] for t in time_breaks]
    t_bound = 0.1
    bounds = [[t-t_bound, t+t_bound] for t in time_breaks]
    # fit the data for 2*nbursts+1 line segments, continuous
    my_pwlf = pwlf.PiecewiseLinFit(time, charge, weights=1 / err, degree=1)
    my_pwlf.fit_guess(time_breaks, bounds=bounds, epsilon=1e-4)
    # compute uncertainties
    my_pwlf.p_values(method='non-linear', step_size=1e-4)
    betas = my_pwlf.beta
    breaks = my_pwlf.fit_breaks
    beta_errs = np.copy(my_pwlf.se[:my_pwlf.beta.size])
    t_step = np.median(np.gradient(time))
    break_errs = np.array([0] + list(my_pwlf.se[my_pwlf.beta.size:]) + [0]) + t_step
    # convert fitted result into numpy pval form
    pvals = [[betas[1], betas[0] - betas[1] * breaks[0]]]
    pvals_err = [
        [beta_errs[1], np.sqrt(beta_errs[0] ** 2 + (beta_errs[1] * breaks[0]) ** 2 + (betas[1] * break_errs[0]) ** 2)]]
    for k in range(2, len(my_pwlf.beta)):
        pvals.append([pvals[-1][0] + betas[k],
                      pvals[-1][1] - betas[k] * breaks[k - 1]])
        pvals_err.append([np.sqrt(pvals_err[-1][0] ** 2 + beta_errs[k] ** 2), np.sqrt(
            pvals_err[-1][1] ** 2 + (beta_errs[k] * breaks[k - 1]) ** 2 + (betas[k] * break_errs[k - 1]) ** 2)])
    # compute index ranges for each segment
    indices = []
    for k in range(1, len(my_pwlf.fit_breaks)):
        indices.append(np.arange(np.argmin(np.abs(time - breaks[k - 1])), np.argmin(np.abs(time - breaks[k]))))
    # compute total charge
    pval1, pval2 = pvals[0], pvals[-1]
    i_k1 = pval1[0]
    i_k2 = pval2[0]
    # times of beginning and end of charge
    t1 = my_pwlf.fit_breaks[1]
    t2 = my_pwlf.fit_breaks[-2]
    # charge correction by leakage currents
    total_charge = 0
    total_charge_err2 = 0
    for n in range(5):
        i1, i2 = pvals[2 * n][0], pvals[2 * n + 2][0]
        t1, t2 = breaks[2 * n + 1], breaks[2 * n + 2]
        total_charge += np.polyval(pvals[2 * n + 2], t2) - np.polyval(pvals[2 * n], t1)
        total_charge -= 0.5 * (i1 + i2) * (t2 - t1)
        # uncertainty on the leakage current value in signal region
        # + uncertainties on the currents + uncertainties on the break times
        total_charge_err2 += (0.5 * abs(i2 - i1) * (t2 - t1)) ** 2 \
                             + (0.5 * pvals_err[2 * n][0] * (t2 - t1)) ** 2 \
                             + (0.5 * pvals_err[2 * n + 2][0] * (t2 - t1)) ** 2 \
                             + (0.5 * (i1 + i2) * break_errs[2 * n + 1]) ** 2 \
                             + (0.5 * (i1 + i2) * break_errs[2 * n + 2]) ** 2
    res = np.copy(charge[:-1])
    for k, pval in enumerate(pvals):
        res[indices[k]] -= np.polyval(pval, time[indices[k]])
    total_charge_err = np.sqrt(
        total_charge_err2 + 2 * np.var(res))  # count 2x the res RMS because we compute a difference q2-q1
    if plot:
        fig, ax = plt.subplots(2, 1, sharex="all")
        ax[0].plot(time, charge, "r+")
        ylim = ax[0].get_ylim()
        ax[0].plot(time, np.polyval(pval1, time))
        ax[0].plot(time, np.polyval(pval2, time))
        for pval in pvals:
            ax[0].plot(time, np.polyval(pval, time))
        ax[0].set_ylim(ylim)
        for brk in my_pwlf.fit_breaks[1:-1]:
            ax[0].axvline(brk, color="k", linestyle="-.")
            ax[1].axvline(brk, color="k", linestyle="-.")
        ax[0].set_title(f"Total charge: {total_charge:.4g} +/- {total_charge_err:.4g} [C]")
        q1 = np.polyval(pval1, t1)  # take charge at last point of first step
        q2 = np.polyval(pval2, t2)  # take charge at first point of first step
        ax[0].axhline(q1, color="k", linestyle="--")
        ax[0].axhline(q2, color="k", linestyle="--")
        ax[0].axvline(t1, color="k", linestyle="--")
        ax[0].axvline(t2, color="k", linestyle="--")
        ax[0].set_xlabel("Time [s]")
        ax[0].set_ylabel(f"SC Charge [C]")
        ax[0].grid()
        ax[1].plot(time[:-1], res, "r+")
        ax[1].set_xlabel("Time [s]")
        ax[1].set_ylabel(f"Difference [C]")
        ax[1].grid()
        fig.tight_layout()
        plt.show()
    return total_charge, total_charge_err, i_k1, i_k2, pvals, indices


def get_info_from_filename(filename):
    words = filename.split("_")
    wavelength = -1
    qsw = "undef"
    flt = "EMPTY"
    for w in words:
        if "Wave" in w:
            wavelength = float(w[4:])
        if "QSW" in w:
            qsw = w[3:min(13, len(w))]
        if "Filter" in w:
            flt = w[6:min(16, len(w))]
    npulses = calculate_npulses(wavelength)
    return wavelength, qsw, npulses, flt


class SolarCellPDDataSet:

    def __init__(self, hdu=None, filename="", npulses=-1, wavelength=-1., nbursts=1, qsw="undef"):
        self.filename = filename
        self.npulses = npulses
        self.wavelength = wavelength
        self.nbursts = nbursts
        self.keithley_err = 5e-12
        self.units = "C"
        self.laserqsw = qsw
        self.dt = 0.13
        if filename != "":
            self.data = self.load(filename)
        elif hdu is not None:
            self.data = self.load_hdu(hdu)

    def __str__(self):
        txt = f"{self.nbursts} bursts containing {self.npulses} laser pulses at wavelength {self.wavelength} nm."
        return txt

    def load(self, filename):
        if ".csv" in filename:
            a = np.loadtxt(filename, skiprows=1, delimiter=",")
            self.data = {"time": a.T[0], "charge": a.T[1]}
        elif ".fits" in filename:
            header = fits.getheader(filename, ext=0)
            self.data = fits.getdata(filename, extname="KEITHLEY", cache=False, memmap=False)
            set_dataset_from_fitsheader(self, header)
        else:
            raise ValueError(f"Unknown file extension for {filename}. Must be .csv or .fits.")
        self.data["time"] -= 2
        self.dt = np.mean(np.gradient(self.data["time"]))
        return self.data

    def load_hdu(self, hdu):
        self.data = hdu["KEITHLEY"].data
        set_dataset_from_fitsheader(self, hdu[0].header)
        self.data["time"] -= 2
        self.dt = np.mean(np.gradient(self.data["time"]))
        return self.data

    def plot_data_set(self, ax):
        ax.plot(self.data["time"], self.data["charge"], "r+")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"PD Charge [{self.units}]")
        ax.grid()


class SolarCellSCDataSet:

    def __init__(self, hdu=None, filename="", npulses=-1, wavelength=-1., nbursts=1, qsw="undef"):
        self.npulses = npulses
        self.wavelength = wavelength
        self.nbursts = nbursts
        self.filename = filename
        self.units = "C"
        self.laserqsw = qsw
        self.dt = 2e-3
        if filename != "":
            self.data = self.load(filename)
        elif hdu is not None:
            self.data = self.load_hdu(hdu)

    def load(self, filename):
        if ".csv" in filename:
            a = np.loadtxt(filename, skiprows=1, delimiter=",")
            self.data = {"time": a.T[0], "charge": a.T[1]}
        elif ".fits" in filename:
            header = fits.getheader(filename, ext=0)
            self.data = fits.getdata(filename, extname="SOLARCELL", cache=False, memmap=False)
            set_dataset_from_fitsheader(self, header)
        else:
            raise ValueError(f"Unknown file extension for {filename}. Must be .csv or .fits.")
        self.data["time"] *= 1e-3
        self.dt = np.mean(np.gradient(self.data["time"]))
        return self.data

    def load_hdu(self, hdu):
        self.data = hdu["SOLARCELL"].data
        set_dataset_from_fitsheader(self, hdu[0].header)
        self.data["time"] *= 1e-3
        self.dt = np.mean(np.gradient(self.data["time"]))
        return self.data

    def plot_data_set(self, ax):
        ax.plot(self.data["time"], self.data["charge"], 'r+')
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"SC Charge [{self.units}]")
        ax.grid()


class SolarCellDataSet:

    def __init__(self, hdu=None, fits_filename="", pd_filename="", sc_filename="", npulses=-1, wavelength=-1.,
                 nbursts=1, qsw="undef", laserfilter="EMPTY", sleep=1, pulse_interval=1e-3):
        if hdu is not None:
            self.pd = SolarCellPDDataSet(hdu=hdu, npulses=npulses, wavelength=wavelength, nbursts=nbursts)
            self.sc = SolarCellSCDataSet(hdu=hdu, npulses=npulses, wavelength=wavelength, nbursts=nbursts)
            set_dataset_from_fitsheader(self, hdu[0].header)
        elif fits_filename != "":
            self.pd = SolarCellPDDataSet(filename=fits_filename, npulses=npulses, wavelength=wavelength,
                                         nbursts=nbursts)
            self.sc = SolarCellSCDataSet(filename=fits_filename, npulses=npulses, wavelength=wavelength,
                                         nbursts=nbursts)
            header = fits.getheader(fits_filename, ext=0)
            set_dataset_from_fitsheader(self, header)
        else:
            self.pd = SolarCellPDDataSet(filename=pd_filename, npulses=npulses, wavelength=wavelength, nbursts=nbursts,
                                         qsw=qsw)
            self.sc = SolarCellSCDataSet(filename=sc_filename, npulses=npulses, wavelength=wavelength, nbursts=nbursts,
                                         qsw=qsw)
            self.npulses = npulses
            self.wavelength = wavelength
            self.nbursts = nbursts
            self.laserqsw = qsw
            self.laserwheelfilter = laserfilter
            self.filename = pd_filename
        self.sleep = sleep  # sleeping times in acquisition code in seconds
        self.pulse_interval = pulse_interval  # interval between two laser pulses in seconds

    def __str__(self):
        txt = f"SolarCellDataSet with nbursts={self.nbursts} and npulses={self.npulses}\n"
        txt += f"Laser set wavelength: {self.wavelength}nm\n"
        txt += f"Laser set QSW: {self.laserqsw}\n"
        txt += f"Laser filter wheel: {self.laserwheelfilter}\n"
        return txt

    def plot_data_set(self):
        fig, ax = plt.subplots(2, 1, sharex="all")
        self.pd.plot_data_set(ax[0])
        self.sc.plot_data_set(ax[1])

    def get_time_breaks(self):
        """
        in seconds
        """
        charge_deriv2 = np.gradient(np.gradient(savgol_filter(self.sc.data["charge"], 41, polyorder=1)))
        cur, indeces = findTransitions(charge_deriv2, n_target_transitions=self.nbursts, pos_or_neg=1, min_data_sep=20)
        cur2, indeces2 = findTransitions(charge_deriv2, n_target_transitions=5, pos_or_neg=-1, min_data_sep=20)
        bad_fit=False
        if cur != -1 and cur2 != -1:
            indeces = [ind[0] for ind in indeces]
            indeces2 = [ind[0] for ind in indeces2]
            transitions = np.union1d(indeces, indeces2)
            output = self.sc.data["time"][transitions]
            g = output[:-1]-output[1:]
            durations = [g[i] + g[i+1] for i in range(0,g.size-1,2)] + [g[i] + g[i+1] for i in range(1,g.size-1,2)]
            if np.std(durations) > 0.1:
                bad_fit=True
        else:
            bad_fit=True
        if bad_fit:
            durations = [self.sleep]
            for b in range(self.nbursts):
                durations.append(self.npulses * self.pulse_interval)
                durations.append(2 * self.sleep - self.npulses * self.pulse_interval)
            output = np.cumsum(durations)[:-1]
        return output


class SolarCellRun:

    def __init__(self, directory_path, tag="", nbursts=5):
        self.directory_path = directory_path
        self.nbursts = nbursts
        self.data_sets = []
        self.filenames = os.listdir(directory_path)
        self.filenames.sort()
        if tag != "":
            self.filenames = [f for f in self.filenames if tag in f]
        # self.filenames = [f for f in self.filenames if ("Photodiode_fromKeithley" in f and ".csv" in f)]
        self.filenames = [f for f in self.filenames if ".fits" in f]
        self.names = ["laserfilter", "qsw", "set_nbursts", "set_npulses", "set_wl", "pd_charge_total",
                      "pd_charge_total_err",
                      "sc_charge_total", "sc_charge_total_err", "pd_ik1", "pd_ik2", "sc_ik1", "sc_ik2"]
        
    def load(self):
        for filename in tqdm(self.filenames[::]):
            if "~" in filename:
                continue
            if ".csv" in filename:
                tmp = filename.split(".")[0]
                wavelength, qsw, npulses, flt = get_info_from_filename(tmp)
                d = SolarCellDataSet(pd_filename=os.path.join(self.directory_path, filename),
                                     sc_filename=os.path.join(self.directory_path,
                                                              filename.replace("Photodiode_fromKeithley_",
                                                                               "SolarCell_fromB2987A_")),
                                     npulses=npulses, wavelength=wavelength, nbursts=self.nbursts, qsw=qsw,
                                     laserfilter=flt)
            elif ".fits" in filename:
                hdu = fits.open(os.path.join(self.directory_path, filename), memmap=False)
                d = SolarCellDataSet(hdu=hdu)
                d.filename = filename
                hdu.close()
            self.data_sets.append(d)
        self.laser_set_wavelengths = np.unique([d.wavelength for d in self.data_sets])
        self.laser_set_npulses = np.unique([d.npulses for d in self.data_sets])
        self.laser_set_nbursts = np.unique([d.nbursts for d in self.data_sets])
        self.ndata = len(self.data_sets)
        formats = ['U10', 'U10'] + ['i4', 'i4'] + ['f8'] * int(len(self.names) - 4)
        self.data = np.recarray((self.ndata,), names=self.names, formats=formats)

    def __str__(self):
        txt = f"SolarCell run with {len(self.data_sets)} data sets.\n"
        txt += f"Laser set wavelengths: {self.laser_set_wavelengths}\n"
        txt += f"Laser set nbursts: {self.laser_set_nbursts}\n"
        return txt

    def get_data_set(self, laser_set_wavelength, nbursts, others={}):
        try:
            for d in self.data_sets:
                if d.wavelength == laser_set_wavelength and nbursts == d.nbursts:
                    if others == {}:
                        return d
                    else:
                        valid = True
                        for key in others.keys():
                            if getattr(d, key) != others[key]:
                                valid = False
                                break
                        if valid:
                            return d
        except:
            raise FileNotFoundError(
                f"SolarCellDataSet with laser_set_wavelength={laser_set_wavelength} and nbursts={nbursts}"
                f" not in {self.directory_path}.")

    def process_data(self, id):
        d = self.data_sets[id]
        for name in self.names:
            self.data[name][id] = 0
        self.data["set_nbursts"][id] = d.nbursts
        self.data["set_npulses"][id] = d.npulses
        self.data["set_wl"][id] = d.wavelength
        self.data["qsw"][id] = d.laserqsw
        self.data["laserfilter"][id] = d.laserwheelfilter

        noise = estimate_noise(d.sc.data["time"], d.sc.data["charge"])
        err = noise * np.ones_like(d.sc.data["time"])
        time_breaks = d.get_time_breaks()
        # d.plot_data_set()
        try:
            charge_sc, charge_sc_err, i_k1, i_k2, pvals_sc, indices_sc = get_solarcell_total_charge(
                d.sc.data["time"], d.sc.data["charge"], time_breaks=time_breaks, err=err, plot=False)
            self.data["sc_charge_total"][id] = charge_sc
            self.data["sc_charge_total_err"][id] = charge_sc_err
            self.data["sc_ik1"][id] = i_k1
            self.data["sc_ik2"][id] = i_k2
        except np.linalg.LinAlgError:
            print("Error. Skip ", d.filename)
            pass
        # try:
        charge_pd, charge_pd_err, i_k1, i_k2, pvals_pd, indices_pd = get_photodiode_total_charge(
            d.pd.data["time"], d.pd.data["charge"], delta_chi2=10, min_len=2,
            err=0.5e-11 * np.ones_like(d.pd.data["time"]), plot=False)
        self.data["pd_charge_total"][id] = charge_pd
        self.data["pd_charge_total_err"][id] = charge_pd_err
        self.data["pd_ik1"][id] = i_k1
        self.data["pd_ik2"][id] = i_k2
        print(id, "done")
        # except:
        #    print("Skip ?", filename)
        #    pass

    def process_data_multi(self, id, catalog):
        d = self.data_sets[id]
        for name in self.names:
            self.data[name] = 0
        catalog["set_nbursts"][id] = d.nbursts
        catalog["set_npulses"][id] = d.npulses
        catalog["set_wl"][id] = d.wavelength
        catalog["qsw"][id] = d.laserqsw
        catalog["laserfilter"][id] = d.laserwheelfilter

        noise = estimate_noise(d.sc.data["time"], d.sc.data["charge"])
        err = noise * np.ones_like(d.sc.data["time"])
        time_breaks = d.get_time_breaks()
        # d.plot_data_set()
        try:
            charge_sc, charge_sc_err, i_k1, i_k2, pvals_sc, indices_sc = get_solarcell_total_charge(
                d.sc.data["time"], d.sc.data["charge"], time_breaks=time_breaks, err=err, plot=False)
            catalog["sc_charge_total"][id] = charge_sc
            catalog["sc_charge_total_err"][id] = charge_sc_err
            catalog["sc_ik1"][id] = i_k1
            catalog["sc_ik2"][id] = i_k2
        except np.linalg.LinAlgError:
            print("Error. Skip ", d.filename)
            pass
        # try:
        charge_pd, charge_pd_err, i_k1, i_k2, pvals_pd, indices_pd = get_photodiode_total_charge(
            d.pd.data["time"], d.pd.data["charge"], delta_chi2=10, min_len=2,
            err=0.5e-11 * np.ones_like(d.pd.data["time"]), plot=False)
        catalog["pd_charge_total"][id] = charge_pd
        catalog["pd_charge_total_err"][id] = charge_pd_err
        catalog["pd_ik1"][id] = i_k1
        catalog["pd_ik2"][id] = i_k2
        print(id, "done")
        # except:
        #    print("Skip ?", filename)
        #    pass

    def solarcell_characterization_test(self):
        #from joblib import Parallel, delayed
        #self.data_sets = Parallel(n_jobs=10, verbose=10, backend="threading")(
        #     delayed(self.process_data)(id) for id in tqdm(range(self.ndata)))
        from multiprocessing import Process, Manager, Pool
        manager = Manager()
        catalog = manager.dict()
        for name in self.names:
            catalog[name] = self.data[name]
        procs = []
        for id in tqdm(range(len(self.data_sets))):
            p = Process(target=self.process_data_multi, args=(id, catalog))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        self.data = catalog
        #with Pool(processes=4) as pool:
        #     pool.map(self.process_data, range(self.ndata))
            #pool.apply_async(self.process_data, range(self.ndata))
            #multiple_results = [pool.apply_async(self.process_data, (i)) for i in range(self.ndata)]
        #for id in tqdm(range(len(self.data_sets))):
        #    self.process_data(id)
        print(self.data["sc_charge_total"])
            
    def solarcell_characterization(self):
        for id, d in enumerate(tqdm(self.data_sets)):
            for name in self.names:
                self.data[name][id] = 0
            self.data["set_nbursts"][id] = d.nbursts
            self.data["set_npulses"][id] = d.npulses
            self.data["set_wl"][id] = d.wavelength
            self.data["qsw"][id] = d.laserqsw
            self.data["laserfilter"][id] = d.laserwheelfilter

            noise = estimate_noise(d.sc.data["time"], d.sc.data["charge"])
            err = noise * np.ones_like(d.sc.data["time"])
            time_breaks = d.get_time_breaks()
            # d.plot_data_set()
            try:
                charge_sc, charge_sc_err, i_k1, i_k2, pvals_sc, indices_sc = get_solarcell_total_charge(
                    d.sc.data["time"], d.sc.data["charge"], time_breaks=time_breaks, err=err, plot=False)
                self.data["sc_charge_total"][id] = charge_sc
                self.data["sc_charge_total_err"][id] = charge_sc_err
                self.data["sc_ik1"][id] = i_k1
                self.data["sc_ik2"][id] = i_k2
            except np.linalg.LinAlgError:
                print("Error. Skip ", d.filename)
                pass
            # try:
            charge_pd, charge_pd_err, i_k1, i_k2, pvals_pd, indices_pd = get_photodiode_total_charge(
                d.pd.data["time"], d.pd.data["charge"], delta_chi2=10, min_len=2,
                err=0.5e-11 * np.ones_like(d.pd.data["time"]), plot=False)
            self.data["pd_charge_total"][id] = charge_pd
            self.data["pd_charge_total_err"][id] = charge_pd_err
            self.data["pd_ik1"][id] = i_k1
            self.data["pd_ik2"][id] = i_k2
            # except:
            #    print("Skip ?", filename)
            #    pass

    def save(self, filename):
        np.save(filename, self.data)

    def load_from_file(self, filename):
        self.data = np.load(filename)
        self.names = ["laserfilter", "qsw", "set_nbursts", "set_npulses", "set_wl", "qsw", "pd_charge_total",
                      "pd_charge_total_err", "sc_charge_total", "sc_charge_total_err",
                      "pd_ik1", "pd_ik2", "sc_ik1", "sc_ik2"]
        self.laser_set_wavelengths = np.unique(self.data["set_wl"])
        self.laser_set_npulses = np.unique(self.data["set_npulses"])
        self.laser_set_nbursts = np.unique(self.data["set_nbursts"])
        self.ndata = len(self.data)

    def plot_summary(self):
        fig, ax = plt.subplots(3, 1, sharex="all")
        ax[0].errorbar(self.data["set_wl"], self.data["pd_charge_total"], yerr=self.data["pd_charge_total_err"],
                       linestyle="none",
                       label="total charge PD", markersize=3, marker="o")
        ax[1].errorbar(self.data["set_wl"], self.data["sc_charge_total"], yerr=self.data["sc_charge_total_err"],
                       linestyle="none",
                       label="total charge SC", markersize=3, marker="o")
        ratio = self.data["sc_charge_total"] / self.data["pd_charge_total"]
        ratio_err = ratio * np.sqrt((self.data["sc_charge_total_err"] / self.data["sc_charge_total"]) ** 2
                                    + (self.data["pd_charge_total_err"] / self.data["pd_charge_total"]) ** 2)
        ax[2].errorbar(self.data["set_wl"], self.data["sc_charge_total"] / self.data["pd_charge_total"], yerr=ratio_err,
                       linestyle="none",
                       label="current", markersize=3, marker="o")
        # ax[1].legend()
        # ax[2].legend()
        ax[0].grid()
        ax[1].grid()
        ax[2].grid()
        ax[0].set_ylabel("PD [C]")
        ax[1].set_ylabel("SC [C]")
        ax[2].set_ylabel("Ratio")
        ax[2].set_xlabel(r"$\lambda$ [nm]")
        plt.show()

    def get_SC_photons_over_PD_charges(self, plot=False):
        ref_data_root = "/data/STARDICE/cbp/solarcell/refCalData/"
        SC_QE_data_file = ref_data_root + 'SC_QE_from_mono_SC_ED_20210618_MultiDay.txt'
        PD_QE_data_file = ref_data_root + 'SM05PD1B_QE.csv'

        SC_QE = np.loadtxt(SC_QE_data_file, skiprows=1, delimiter=",").T
        PD_QE = np.loadtxt(PD_QE_data_file, skiprows=1, delimiter=",").T

        # PD_QE is in A/W units, convert it to e-/photons as SC_QE
        PD_QE[1] *= (const.h * const.c / (PD_QE[0] * 1e-9 * u.meter * const.e.value)).value

        SC_QE_f = interpolate.interp1d(SC_QE[0], SC_QE[1], bounds_error=False, fill_value=np.min(SC_QE[1]))
        PD_QE_f = interpolate.interp1d(PD_QE[0], PD_QE[1], bounds_error=False, fill_value=np.min(PD_QE[1]))

        tr = (self.data["sc_charge_total"] / SC_QE_f(self.data["set_wl"])) / np.abs(self.data["pd_charge_total"])
        tr_err = tr * np.sqrt((self.data["sc_charge_total_err"] / self.data["sc_charge_total"]) ** 2 + (
                    self.data["pd_charge_total_err"] / self.data["pd_charge_total"]) ** 2)
        if plot:
            double_Al_wavelength, double_Al_responsivity = np.loadtxt(os.path.join(ref_data_root, 'TwoBounceAl.dat'),
                                                                      delimiter=',', skiprows=1).T
            Al_wl, Al_CFHT = np.loadtxt(os.path.join(ref_data_root, 'CFHT_Primary_Transmission.dat'), delimiter=' ',
                                        skiprows=0).T
            Al_wl *= 0.1
            Al_CFHT *= Al_CFHT

            fig = plt.figure()
            plt.errorbar(self.data["set_wl"], tr, yerr=tr_err, marker='+', linestyle="none", label="data")
            theory = double_Al_responsivity / PD_QE_f(double_Al_wavelength)
            factor = np.sum(tr) / np.sum(theory)
            plt.plot(double_Al_wavelength, factor * theory, '-', label="theory")
            theory = Al_CFHT / PD_QE_f(Al_wl)
            factor = np.sum(tr) / np.sum(theory)
            plt.plot(Al_wl, factor * theory, '-', label="theory CFHT")
            plt.xlabel('Laser wavelength (nm)')
            plt.ylabel('SC photons / PD charges [$\gamma$/C]')
            plt.grid()
            plt.title("CBP transmission")
            plt.legend()
            plt.show()
        return self.data["set_wl"], tr, tr_err

    def get_SC_photons_over_PD_photons(self, plot=False):
        ref_data_root = "/data/STARDICE/cbp/solarcell/refCalData/"
        SC_QE_data_file = ref_data_root + 'SC_QE_from_mono_SC_ED_20210618_MultiDay.txt'
        PD_QE_data_file = ref_data_root + 'SM05PD1B_QE.csv'

        SC_QE = np.loadtxt(SC_QE_data_file, skiprows=1, delimiter=",").T
        PD_QE = np.loadtxt(PD_QE_data_file, skiprows=1, delimiter=",").T

        # PD_QE is in A/W units, convert it to e-/photons as SC_QE
        PD_QE[1] *= (const.h * const.c / (PD_QE[0] * 1e-9 * u.meter * const.e.value)).value

        SC_QE_f = interpolate.interp1d(SC_QE[0], SC_QE[1], bounds_error=False, fill_value=np.min(SC_QE[1]))
        PD_QE_f = interpolate.interp1d(PD_QE[0], PD_QE[1], bounds_error=False, fill_value=np.min(PD_QE[1]))

        tr = (self.data["sc_charge_total"] / SC_QE_f(self.data["set_wl"])) / np.abs(
            self.data["pd_charge_total"] / PD_QE_f(self.data["set_wl"]))
        tr_err = tr * np.sqrt((self.data["sc_charge_total_err"] / self.data["sc_charge_total"]) ** 2 + (
                    self.data["pd_charge_total_err"] / self.data["pd_charge_total"]) ** 2)
        if plot:
            double_Al_wavelength, double_Al_responsivity = np.loadtxt(os.path.join(ref_data_root, 'TwoBounceAl.dat'),
                                                                      delimiter=',', skiprows=1).T
            Al_wl, Al_CFHT = np.loadtxt(os.path.join(ref_data_root, 'CFHT_Primary_Transmission.dat'), delimiter=' ',
                                        skiprows=0).T
            Al_wl *= 0.1
            Al_CFHT *= Al_CFHT

            fig = plt.figure()
            plt.errorbar(self.data["set_wl"], tr, yerr=tr_err, marker='+', linestyle="none", label="data")
            theory = double_Al_responsivity
            factor = np.sum(tr) / np.sum(theory)
            plt.plot(double_Al_wavelength, factor * theory, '-', label="theory")
            theory = Al_CFHT
            factor = np.sum(tr) / np.sum(theory)
            plt.plot(Al_wl, factor * theory, '-', label="theory CFHT")
            plt.xlabel('Laser wavelength (nm)')
            plt.ylabel('SC photons / PD photons')
            plt.grid()
            plt.title("CBP transmission")
            plt.legend()
            plt.show()
        return self.data["set_wl"], tr, tr_err
