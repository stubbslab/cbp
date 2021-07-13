import cantrips as can
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy
import AstronomicalParameterArchive as apa

def fit_line_pieces(x, y, err=None, delta_chi2=10):
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
    while endpoint < len(x) - 1 and counter <= len(x):
        chi2 = 0
        # start a line with at least three points
        for index in range(startpoint + 2, len(x)):
            indices_tmp = np.arange(startpoint, index, dtype=int)
            pval_tmp = np.polyfit(x[indices_tmp], y[indices_tmp], deg=1)
            # compute the chi square
            if err is None:
                stand_in_err = np.std(y[startpoint:index])
                print ('[startpoint, index] = ' + str([startpoint, index]))
                print ('stand_in_err = ' + str(stand_in_err))
                chi2_tmp = np.sum(((y[indices_tmp] - np.polyval(pval_tmp, x[indices_tmp])) / stand_in_err) ** 2)
            else:
                chi2_tmp = np.sum(((y[indices_tmp] - np.polyval(pval_tmp, x[indices_tmp])) / err[indices_tmp]) ** 2)
            print ('[chi2, chi2_tmp, len(indices_tmp), (chi2_tmp - chi2) / len(indices_tmp), delta_chi2] = ' + str([chi2, chi2_tmp, len(indices_tmp), (chi2_tmp - chi2) / len(indices_tmp), delta_chi2] ))
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


def get_total_charge_and_leakage_currents(time, charge, delta_chi2=10, err=None, plot=False):
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
    pvals, indices = fit_line_pieces(time, charge, delta_chi2=delta_chi2, err=err)
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
    if plot:
        fig = plt.figure()
        plt.plot(time, charge, "r+")
        plt.plot(time, np.polyval(pval1, time))
        plt.plot(time, np.polyval(pval2, time))
        plt.title(f"Total charge: {total_charge:.4g} +/- {total_charge_err:.4g} [C]")
        plt.axhline(q1, color="k", linestyle="--")
        plt.axhline(q2, color="k", linestyle="--")
        plt.axvline(t1, color="k", linestyle="--")
        plt.axvline(t2, color="k", linestyle="--")
        plt.xlabel("Time [s]")
        plt.ylabel(f"Charge [C]")
        plt.grid()
        fig.tight_layout()
        plt.show()
    return total_charge, total_charge_err, i_k1, i_k2

def groupBurstStartGuesses(nominal_timestamps, ungrouped_guess_indeces, min_sep = 20, verbose = 0):
    all_guesses_grouped = 0
    current_index = 0
    current_group = []
    guess_groups = []
    while not(all_guesses_grouped):
        #print ('[current_group, current_index, start_guesses] = ' + str([current_group, current_index, start_guesses]))
        if len(current_group) == 0:
            current_group = [ungrouped_guess_indeces[current_index]]
        elif ungrouped_guess_indeces[current_index] - current_group[-1] < min_sep:
            current_group = current_group + [ungrouped_guess_indeces[current_index]]
        else:
            #new_guess = np.mean([nominal_timestamps[index] for index in current_group])
            guess_groups = guess_groups + [current_group]
            current_group = [ungrouped_guess_indeces[current_index]]
        if current_index == len(ungrouped_guess_indeces) - 1:
            #new_guess = np.mean([nominal_timestamps[index] for index in current_group])
            guess_groups = guess_groups + [current_group]
            all_guesses_grouped = 1
        else:
            current_index = current_index + 1
    if verbose: print ('guess_groups = ' + str(guess_groups ))
    return guess_groups

def findTransitions(arr, n_target_transitions, max_n_sig_to_identify = 14, min_n_sig_to_identify = 1, pos_or_neg = 1, verbose = 0, max_iterations = 20, min_data_sep = 10, show_process = 0):
    #print ('Finding transitions...')
    transitions = []
    low_n_sig = min_n_sig_to_identify
    high_n_sig = max_n_sig_to_identify
    arr_med = np.median(arr)
    arr_std = np.std(arr)
    if verbose: print ('[arr_med, arr_std] = ' + str([arr_med, arr_std]))
    if verbose: print ('n_target_transitions = ' + str(n_target_transitions))
    iter_number = 0
    while len(transitions) != n_target_transitions and iter_number <= max_iterations:
        current_n_sig = (high_n_sig + low_n_sig) / 2
        if show_process:
            f, axarr = plt.subplots(1,1)
            axarr.plot(arr, c = 'k', alpha = 0.75)
            axarr.axhline(arr_med + arr_std * current_n_sig, color = 'cyan')
            axarr.axhline(arr_med - arr_std * current_n_sig, color = 'cyan')
            plt.draw()
            plt.pause(0.5)
        if pos_or_neg == 1:
            transition_indeces = [ i for i in range(len(arr)) if arr[i] > arr_med + arr_std * current_n_sig  ]
        else:
            transition_indeces = [ i for i in range(len(arr)) if arr[i] < arr_med - arr_std * current_n_sig  ]
        if verbose:
            print ('transition_indeces = ' + str(transition_indeces))
            print ('current_n_sig = ' + str(current_n_sig))
        if len(transition_indeces) >= n_target_transitions:
            #print ('turn_on_guess_indeces = ' + str(turn_on_guess_indeces))
            transitions = groupBurstStartGuesses(arr, transition_indeces, verbose = verbose, min_sep = min_data_sep)
        else:
            transitions = [arr[index] for index in transition_indeces]
        if verbose:
            print ('transitions = ' + str(transitions))
        if len(transitions) > n_target_transitions:
            low_n_sig = current_n_sig
        if len(transitions) < n_target_transitions:
            high_n_sig = current_n_sig
        iter_number = iter_number + 1
        if  len(transitions) != n_target_transitions and iter_number > max_iterations:
            current_n_sig = -1
            transitions = [-1 for i in range(n_target_transitions)]
            print ('Failed to find transitions before exceeding max allowed iterations (' + str(max_iterations) + ').  This will be marked as a bad data point. ')
        if show_process:
            plt.close('all')
    if verbose:
        print ('Returning: ' + str([current_n_sig, transitions]))
    return [current_n_sig, transitions]


def syncSCDataToSteps(SC_data, ref_burst_times, start_n_sig_to_identify = 14, n_sig_id_step = 1 ):
    burst_starts = ref_burst_times[0]
    burst_ends = ref_burst_times[1]
    n_bursts = len(burst_starts)
    burst_seps = [burst_ends[i] - burst_ends[0] for i in range(0, n_bursts)]
    SC_std, SC_med = [np.std(np.gradient(SC_data[1])), np.std(np.gradient(SC_data[1]))]
    turn_on_guesses = []
    n_sig_id_burst = start_n_sig_to_identify + n_sig_id_step
    print ('len(burst_seps) = ' + str(len(burst_seps)))
    while len(turn_on_guesses) < len(burst_seps):
        n_sig_id_burst = n_sig_id_burst - n_sig_id_step
        turn_on_guess_indeces = [ i for i in range(len(SC_data[1])) if np.gradient(SC_data[1])[i] < SC_med - SC_std * n_sig_id_burst  ]
        if len(turn_on_guess_indeces) >= len(ref_burst_times):
            #print ('turn_on_guess_indeces = ' + str(turn_on_guess_indeces))
            turn_on_guesses = groupBurstStartGuesses(SC_data[0], turn_on_guess_indeces)
        print ('[turn_on_guesses, n_sig_id_burst] = ' + str([turn_on_guesses, n_sig_id_burst]))
    #print ('turn_on_guesses = ' + str(turn_on_guesses))
    turn_on_seps = [turn_on_guesses[i] - turn_on_guesses[0] for i in range(0, len(turn_on_guesses))]
    #print ('burst_seps = ' + str(burst_seps))
    #print ('turn_on_seps = ' + str(turn_on_seps))
    #print( np.polyfit(turn_on_seps, burst_seps, 1) )
    time_scaling = np.polyfit(turn_on_seps, burst_seps, 1)[0]
    SC_time_seps = [elem - SC_data[0][0] for elem in SC_data[0]]
    #print ('time_scaling = ' + str(time_scaling))
    true_SC_time_seps = np.array(SC_time_seps) * time_scaling

    return true_SC_time_seps

def pieceFitCharge(xs, ys, piecewise_parts_indeces, verbose = 0):
    piecewise_parts_indeces = [0] + piecewise_parts_indeces + [len(xs) - 1]
    piecewise_poly_fits = [[] for i in range(len(piecewise_parts_indeces) - 1)]
    piecewise_exp_fits = [[] for i in range(len(piecewise_parts_indeces) - 1)]
    piecewise_exp_fit = [[] for i in range(len(piecewise_parts_indeces) - 1)]
    paired_piece_fit_funct = lambda xs, transition_x, transition_y, left_slope, right_slope: (transition_x - np.where(xs < transition_x, xs, transition_x )) * -left_slope + transition_y + (np.where(xs >= transition_x, xs, transition_x ) - transition_x ) * right_slope
    centered_exp_funct = lambda xs_to_plot, x1, x0, A, alpha: x0 + xs_to_plot * x1 + A * np.exp(-xs_to_plot * alpha)
    if verbose:
        print ('len(xs) = ' + str(len(xs)))
        print ('piecewise_parts_indeces = ' + str(piecewise_parts_indeces))
        piecewise_seps = [piecewise_parts_indeces[i] - piecewise_parts_indeces[i-1] for i in range(1, len(piecewise_parts_indeces))]
        print ('piecewise_seps = ' + str(piecewise_seps))
        print ('np.where(np.array(piecewise_seps) == 0, 1, 0) = ' + str(np.where(np.array(piecewise_seps) == 0, 1, 0)))

        if np.any(np.where(np.array(piecewise_seps) == 0, 1, 0)):
            f, axarr = plt.subplots(1,1)
            axarr.scatter(xs, ys, marker = 'x', c = 'k')
            [axarr.axvline(xs[line], c = 'r')  for line in piecewise_parts_indeces ]
            plt.show()
    for i in range(len(piecewise_parts_indeces) - 2):
        paired_piecewise_xs = np.array(xs[piecewise_parts_indeces[i]:piecewise_parts_indeces[i+2]] )
        paired_piecewise_ys = np.array(ys[piecewise_parts_indeces[i]:piecewise_parts_indeces[i+2]] )
        init_paired_fit_guess = [xs[piecewise_parts_indeces[i+1]], ys[piecewise_parts_indeces[i+1]], (ys[piecewise_parts_indeces[i+1]] - ys[piecewise_parts_indeces[i]]) / (xs[piecewise_parts_indeces[i+1]] - xs[piecewise_parts_indeces[i]]), (ys[piecewise_parts_indeces[i+2]] - ys[piecewise_parts_indeces[i+1]]) / (xs[piecewise_parts_indeces[i+2]] - xs[piecewise_parts_indeces[i+1]])]
        try:
            paired_piece_fit = scipy.optimize.curve_fit(paired_piece_fit_funct, paired_piecewise_xs, paired_piecewise_ys, p0 = init_paired_fit_guess)[0]
            piecewise_parts_indeces[i + 1] = np.argmin(np.abs(paired_piecewise_xs - paired_piece_fit[0])) + piecewise_parts_indeces[i]
        except:
            piecewise_parts_indeces[i + 1] = np.argmin(np.abs(paired_piecewise_xs - init_paired_fit_guess[0])) + piecewise_parts_indeces[i]
    if verbose:
        print ('len(xs) = ' + str(len(xs)))
        print ('piecewise_parts_indeces = ' + str(piecewise_parts_indeces))
    for i in range(len(piecewise_parts_indeces) - 1):
        piecewise_xs = np.array(xs[piecewise_parts_indeces[i]:piecewise_parts_indeces[i+1]+1] )
        piecewise_ys = np.array(ys[piecewise_parts_indeces[i]:piecewise_parts_indeces[i+1]+1] )
        center_x = np.mean(piecewise_xs)
        if len(piecewise_xs) > 1:
            centered_xs = piecewise_xs - center_x
            if verbose:
                print ('centered_xs = ' + str(centered_xs))
                print ('piecewise_ys = ' + str(piecewise_ys))
            try:
                centered_lin_fit_terms = np.polyfit(centered_xs, piecewise_ys, 1)
            except np.linalg.LinAlgError:
                print ('[centered_xs, centered_ys] = ' + str([centered_xs, piecewise_ys] ))
                centered_lin_fit_terms = [0.0, 0.0 ]
            try:
                centered_poly_fit = np.poly1d(np.polyfit(centered_xs, piecewise_ys, 2))
            except np.linalg.LinAlgError:
                print ('[centered_xs, centered_ys] = ' + str([centered_xs, piecewise_ys] ))
                centered_poly_fit = np.poly1d([0.0] + list(centered_lin_fit_terms))
            if len(piecewise_ys) >= 4:
                centered_exp_fit = scipy.optimize.curve_fit(centered_exp_funct, centered_xs, piecewise_ys, p0 = [*centered_lin_fit_terms, 0.0, 0.0])[0]
            else:
                centered_exp_fit = [*centered_lin_fit_terms, 0.0, 0.0]
        else:
            centered_exp_fit = [0.0, 0.0, 0.0, 0.0]
            centered_poly_fit = np.poly1d([0.0, 0.0, 0.0])
        piecewise_poly_fit = lambda xs_to_plot, center_x = center_x, centered_poly_fit = centered_poly_fit: centered_poly_fit(np.array(xs_to_plot) - center_x)
        piecewise_poly_fits[i] = piecewise_poly_fit
        fit_params = centered_exp_fit
        piecewise_exp_fit = lambda xs_to_plot, center_x = center_x, fit_params = fit_params: centered_exp_funct(np.array(xs_to_plot) - center_x, *fit_params)
        piecewise_exp_fits[i] = piecewise_exp_fit
    median_abs_y = np.median(np.abs(ys))
    for i in range(len(piecewise_parts_indeces) - 1):
        piecewise_xs = np.array(xs[piecewise_parts_indeces[i]:piecewise_parts_indeces[i+1]] )
    for i in range(len(piecewise_parts_indeces) - 1):
        piecewise_xs = np.array(xs[piecewise_parts_indeces[i]:piecewise_parts_indeces[i+1]] )
        piecewise_ys = np.array(ys[piecewise_parts_indeces[i]:piecewise_parts_indeces[i+1]] )
    for i in range(len(piecewise_parts_indeces) - 1):
        piecewise_xs = np.array(xs[piecewise_parts_indeces[i]:piecewise_parts_indeces[i+1]] )
        piecewise_ys = np.array(ys[piecewise_parts_indeces[i]:piecewise_parts_indeces[i+1]] )

    if verbose:
        print ('piecewise_parts_indeces = ' + str(piecewise_parts_indeces))
    return [piecewise_parts_indeces, piecewise_poly_fits]

def calculatePhotoChargesGivenTransitionFits(transition_charges, bad_data_indeces, orig_data, time_seps, n_bursts, current_neg = 0):
    charges = [[transition_charges[j][2 * i + 2] - transition_charges[j][2 * i + 1] for i in range(n_bursts)] for j in range(len(orig_data)) if not (j in bad_data_indeces)]
    charge_times = [[time_seps[j][2 * i + 2] - time_seps[j][2 * i + 1] for i in range(n_bursts)] for j in range(len(orig_data)) if not (j in bad_data_indeces)]
    dark_charges = [[transition_charges[j][2 * i + 1] - transition_charges[j][2 * i] for i in range(n_bursts + 1)] for j in range(len(orig_data)) if not (j in bad_data_indeces)]
    dark_charge_times = [ [time_seps[j][2 * i + 1] - time_seps[j][2 * i] for i in range(n_bursts + 1)] for j in range(len(orig_data)) if not (j in bad_data_indeces) ]

    dark_currents = [[dark_charges[j][i] / dark_charge_times[j][i] for i in range(n_bursts + 1)] for j in range(len(dark_charges))]
    dark_charges_during_illumination = [[(dark_currents[j][i + 1] + dark_currents[j][i]) / 2 * charge_times[j][i] for i in range(n_bursts)] for j in range(len(charges)) ]

    photo_charges = [[(charges[j][i] - dark_charges_during_illumination[j][i]) * (-1 if current_neg else 1) for i in range(n_bursts)] for j in range(len(charges)) ]
    return photo_charges


def smoothArray(array, smoothing):
    if smoothing == 1:
        smoothed = array
    else:
        smoothed = [np.mean(array[max(0, i - smoothing // 2):min(len(array), i + smoothing // 2 + 1)]) for i in range(len(array))]
    #smoothed = np.convolve(array, smoothing_box, mode = 'same')
    return smoothed

def readInSCQEData(SC_QE_data_file):
    SC_QE_wavelengths, SC_QE_data = can.readInColumnsToList(SC_QE_data_file, delimiter = ', ', n_ignore = 1, convert_to_float = 1)
    SC_QE_wavelengths, SC_QE_data = can.safeSortOneListByAnother(SC_QE_wavelengths, [SC_QE_wavelengths, SC_QE_data])
    SC_QE_wavelength_correction = -5
    SC_QE_wavelengths = [wave + SC_QE_wavelength_correction for wave in SC_QE_wavelengths]
    SC_QE_data = np.array(SC_QE_data)
    SC_QE_smoothing_nm = 100
    smooth_locations = np.arange(wavelength_range[0] - 1, wavelength_range[1] + 2, 1)
    SC_QE_smoothed = [np.mean([SC_QE_data[i] for i in range(len(SC_QE_data)) if (np.abs(SC_QE_wavelengths[i] - j) < SC_QE_smoothing_nm / 2)]) for j in smooth_locations]
    SC_QE_interp = scipy.interpolate.interp1d(smooth_locations, SC_QE_smoothed)
    return SC_QE_interp

def readInPDQEData(PD_QE_data_file):
    astro_arch = apa.AstronomicalParameterArchive()
    PD_QE_wavelengths, PD_responsivity_data = can.readInColumnsToList(PD_QE_data_file, delimiter = ', ', n_ignore = 1, convert_to_float = 1)
    PD_QE_wavelengths, PD_responsivity_data = can.safeSortOneListByAnother(PD_QE_wavelengths, [PD_QE_wavelengths, PD_responsivity_data])
    electron_charge = astro_arch.getElectronCharge()
    Ne_per_C = 1 / electron_charge
    planck_h = astro_arch.getPlancksConstant()
    sol_nm_per_s = astro_arch.getc() * 10.0 ** 12
    phot_J_times_nm = planck_h * sol_nm_per_s
    Nphot_per_J_per_nm = 1.0 / phot_J_times_nm

    PD_QE_data = Ne_per_C / (Nphot_per_J_per_nm * np.array(PD_QE_wavelengths)) * np.array(PD_responsivity_data)
    PD_QE_interp = scipy.interpolate.interp1d(PD_QE_wavelengths, PD_QE_data)
    return PD_QE_interp



def measureDetectorCharges(wavelengths, data_files, timestamp_files, smoothing, max_val, photocharge_neg,
                           data_dir = '', read_in_data_clipping = 1,
                           max_std_to_identify_bursts = 14, min_std_to_identify_bursts = 0.25, min_data_sep_to_identify_bursts = 20,
                           show_progress = 1, progress_plot_extra_title = '', save_transitions_plot = 1, save_plot_prefix = None,
                           show_process = 0):
    bad_data_indeces = []

    data_sets = [can.readInColumnsToList(file, data_dir, delimiter = ', ', n_ignore = 1, convert_to_float = 1) for file in data_files]
    data_sets = [[arr[0][read_in_data_clipping:-read_in_data_clipping], arr[1][read_in_data_clipping:-read_in_data_clipping]] for arr in data_sets]
    uncorrected_times = [data_set[0] for data_set in data_sets]
    raw_charges = [data_set[1] for data_set in data_sets]


    sat_data_indeces = [i for i in range(len(data_sets)) if np.any(np.array(data_sets[i][1]) > max_val)  ]
    if len(sat_data_indeces) > 0:
        print ('The following files will be ignored because at least one value in the data array exceeds the max value of ' + str(max_SC_val) + ':')
        print ([data_files[i] for i in sat_data_indeces])
    bad_data_indeces = can.union([bad_data_indeces, sat_data_indeces])

    data_smoothed = [smoothArray(arr, smoothing) for arr in raw_charges]
    data_to_find_transitions = [np.gradient(np.gradient(arr)) for arr in data_smoothed]

    ref_times = [can.readInColumnsToList(file, data_dir, delimiter = ', ', n_ignore = 1, convert_to_float = 1) for file in ref_burst_timestamps_files]
    n_bursts = len(ref_times[0][0])

    burst_on_indeces = [[] for arr in data_smoothed]
    burst_off_indeces = [[] for arr in data_smoothed]
    for i in range(len(data_smoothed)):
        arr = data_to_find_transitions[i]
        burst_on_indeces[i] = findTransitions(arr[smoothing:-smoothing], n_bursts, max_n_sig_to_identify = max_std_to_identify_bursts, min_n_sig_to_identify = min_std_to_identify_bursts, pos_or_neg = (-1 if photocharge_neg else 1), min_data_sep = min_data_sep_to_identify_bursts, show_process = show_process, verbose = show_process)
        burst_off_indeces[i] = findTransitions(arr[smoothing:-smoothing], n_bursts, max_n_sig_to_identify = max_std_to_identify_bursts, min_n_sig_to_identify = min_std_to_identify_bursts, pos_or_neg = (1 if photocharge_neg else -1), min_data_sep = min_data_sep_to_identify_bursts, show_process = show_process, verbose = show_process)
        if i == 28:
            print ('burst_on_indeces = ' + str(burst_on_indeces[i]))
        burst_on_indeces[i] = [burst_on_indeces[i][0], [int(np.median(index_set)) + smoothing for index_set in burst_on_indeces[i][1]] ]
        if i == 28:
            print ('burst_on_indeces = ' + str(burst_on_indeces[i]))
        burst_off_indeces[i] = [burst_off_indeces[i][0], [int(np.median(index_set)) + smoothing  for index_set in burst_off_indeces[i][1]] ]

    failed_burst_id_indeces = [i for i in range(len(data_sets)) if (burst_on_indeces[i][0] == -1) or (burst_off_indeces[i][0] == -1)]
    if len(failed_burst_id_indeces ) > 0:
        print ('The following files failed to identify all ' + str(n_bursts) + ' bursts in the this data set.  We will mark it as a bad observation: ')
        print ([data_files[i] for i in failed_burst_id_indeces ])
        #print ('Here is what the problematic data look like: ' )
        #for index in failed_burst_id_indeces:
        #    plt.plot(data_to_find_transitions[index])
        #    plt.show()

    bad_data_indeces = can.union([bad_data_indeces, failed_burst_id_indeces])

    burst_off_before_burst_on = [[burst_off_indeces[i][1][j-1] > burst_on_indeces[i][1][j] for j in range(1, len(burst_off_indeces[i][1]))] for i in range(len(burst_on_indeces))]
    burst_on_before_next_burst_off = [[burst_on_indeces[i][1][j] > burst_off_indeces[i][1][j] for j in range(1, len(burst_off_indeces[i][1]))] for i in range(len(burst_on_indeces))]


    bursts_out_of_order_indeces = [i for i in range(len(burst_on_indeces)) if (np.any(burst_off_before_burst_on[i]) or np.any(burst_on_before_next_burst_off[i])) ]
    if len(bursts_out_of_order_indeces ) > 0:
        print ('The following files identified burst starts and ends out of order. We will mark it as a bad observation: ')
        print ([data_files[i] for i in bursts_out_of_order_indeces ])
        #print ('Here were the burst on and off indeces of those problematic files, and what the data look like: ')
        #for index in bursts_out_of_order_indeces:
        #    print ('index = ' + str(index))
        #    print ('burst_off_indeces[index] = ' + str(burst_off_indeces[index]))
        #    print ('burst_on_indeces[index] = ' + str(burst_on_indeces[index]))
        #    plt.plot(data_to_find_transitions[index])
        #    plt.show()
    bad_data_indeces = can.union([bad_data_indeces, bursts_out_of_order_indeces])

    corrected_times = [[] for i in range(len(uncorrected_times))]
    time_sep_fits = [[] for i in range(len(uncorrected_times))]
    for i in range(0, len(uncorrected_times)):
        if i == 28:
            print ('i = ' + str(i) )
            print ('data_files[i] = ' + str(data_files[i] ))
            print ('len(uncorrected_times[i]) = ' + str(len(uncorrected_times[i])))
            print ('burst_on_indeces[i][1] = ' + str(burst_on_indeces[i][1]))
        turn_on_times = [uncorrected_times[i][index] for index in burst_on_indeces[i][1]]
        deltaT_turn_on = [turn_on - turn_on_times[0] for turn_on in turn_on_times]
        if np.sum(deltaT_turn_on) == 0:
            time_sep_fit = [1.0, 0.0]
            print ('The following files did not identify usable burst times. We will mark it as a bad observation: ')
            print (data_files[i] )
            bad_data_indeces = can.union([bad_data_indeces, [i]])
        else:
            time_sep_fit = np.polyfit(deltaT_turn_on, [ref_time - ref_times[i][0][0] for ref_time in ref_times[i][0]], 1)
        time_sep_fits[i] = time_sep_fit
        true_seps = (np.array(uncorrected_times[i]) - uncorrected_times[i][0]) * time_sep_fit[0]
        corrected_times[i] = true_seps[:]

    good_wavelengths = [wavelengths[i] for i in range(len(wavelengths)) if not (i in bad_data_indeces)]

    piece_fits = [pieceFitCharge(corrected_times[i], raw_charges[i], can.flattenListOfLists([[burst_on_indeces[i][1][j], burst_off_indeces[i][1][j]] for j in range(len(burst_on_indeces[i][1])) ]) ) if not (i in bad_data_indeces) else [] for i in range(len(data_sets)) ]
    transition_indeces, piece_fit_functs = [[piece_fits[i][0] if not (i in bad_data_indeces) else can.flattenListOfLists([[burst_on_indeces[i][1][j], burst_off_indeces[i][1][j]] for j in range(len(burst_on_indeces[i][1])) ]) for i in range(len(piece_fits))],
                                                  [piece_fits[i][1] if not (i in bad_data_indeces) else [] for i in range(len(piece_fits))]  ]

    transition_times = [[ corrected_times[i][index] for index in transition_indeces[i]] for i in range(len(piece_fit_functs)) ]
    transition_charges = [[ piece_fit_functs[i][0](corrected_times[i][transition_indeces[i][0]]) ] + [piece_fit_functs[i][j](corrected_times[i][transition_indeces[i][j+1]]) for j in range(len(piece_fit_functs[i]))] if not (i in bad_data_indeces)
                              else  [raw_charges[i][transition_indeces[i][j+1]] for j in range(len(piece_fit_functs[i]))]
                              for i in range(len(piece_fit_functs)) ]
    transition_charges = [[transition_charges[i][j] for j in range(len(transition_charges[i]))] for i in range(len(transition_charges))]
    photo_charges = calculatePhotoChargesGivenTransitionFits(transition_charges, bad_data_indeces, data_sets, transition_times, n_bursts, current_neg = photocharge_neg)

    charges_by_wavelength_dict = {good_wavelengths[i]: photo_charges[i] for i in range(len(good_wavelengths))}

    print ('bad_data_indeces = ' + str(bad_data_indeces))
    for j in range(len(data_sets)):
        if not (j in bad_data_indeces):
            f, axarr = plt.subplots(2,1)
            axarr[0].plot(corrected_times[j], np.array(raw_charges[j]) * 10.0 ** 6.0, c = 'k')
            axarr[0].set_ylabel('Charge $(\mu C)$')
            axarr[0].set_title(progress_plot_extra_title + ': ' + data_files[j])
            axarr[0].scatter(transition_times[j], np.array(transition_charges[j]) * 10.0 ** 6.0 , marker = 'x', c = 'orange')
            #[axarr[0].axvline(SC_data[j][0][index], color = 'r') for index in SC_transition_indeces[j][1]]
            axarr[1].plot(corrected_times[j], np.array(data_to_find_transitions[j]) * 10.0 ** 6.0 * time_sep_fits[j][0] ** -2.0, c = 'k')
            axarr[1].set_xlabel(r'$\Delta t$ (s)')
            axarr[1].set_ylabel('Charge 2nd deriv $(\mu C \ s^{-2})$')
            for i in range(len(transition_indeces[j]) // 2):
                off_index = transition_indeces[j][2 * i]
                on_index = transition_indeces[j][2 * i + 1]
                axarr[0].axvline(corrected_times[j][on_index], color = 'g', linestyle = '--', alpha = 0.5)
                axarr[0].axvline(corrected_times[j][off_index], color = 'r', linestyle = '--', alpha = 0.5)
                axarr[1].axvline(corrected_times[j][on_index], color = 'g', linestyle = '--', alpha = 0.5)
                axarr[1].axvline(corrected_times[j][off_index], color = 'r', linestyle = '--', alpha = 0.5)
            #[axarr[1].axvline(PD_data[j][0][index], color = 'g') for index in PD_transition_indeces[j][1]]
            #[axarr[1].axvline(PD_data[j][0][index], color = 'r') for index in PD_transition_indeces[j][1]]
            axarr[0].set_title(progress_plot_extra_title  + r' data for $\lambda = $' + str(wavelengths[j]) + 'nm')
            plt.gcf().subplots_adjust(left=0.12)
            plt.draw()
            plt.pause(0.25)

            if save_transitions_plot and save_plot_prefix != None:
                plt.savefig(data_dir + save_plot_prefix + str(wavelengths[j]) + '.pdf')
            plt.close('all')

    return charges_by_wavelength_dict


if __name__=="__main__":
    SC_current_neg = 0
    PD_current_neg = 1
    wavelengths = np.arange(350, 670, 1).tolist() + np.arange(680, 1050, 15).tolist()
    wavelengths = np.arange(350, 1050, 5)
    #wavelengths = [400, 800]
    wavelength_range = [min(wavelengths),  max(wavelengths)]
    data_index_to_study = 0
    max_SC_val = 2e-6
    max_PD_val = 1e-3
    SC_smoothing = 10 #How many adjacent solar cell samples should we sum when looking for the bursts?
    PD_smoothing = 1 #How many adjacent photodiode samples should we sum when looking for the bursts?
    #n_points_ignore_transition_id = 4 #We identify transitions by looking at 2nd numerical derivative => need to ignore some data points.

    ref_data_root = '../refCalData/'
    data_root = '../data/CBP_throughput_calib_data/'
    file_suffixes = can.flattenListOfLists([['_Wave' + str(wave) + '_QSWMax_' + str(iris_setting) + '.csv' for wave in wavelengths  ] for iris_setting in ['5mmPin3_AmbLight' ] ])
    n_pulses = [100 for suffix in file_suffixes]
    data_dir = data_root + 'ut20210713/CBPOffTarget/'
    SC_data_files = [ 'SolarCell_fromB2987A' + suffix for suffix in file_suffixes ]
    PD_data_files = ['Photodiode_fromKeithley' + suffix for suffix in file_suffixes ]
    ref_burst_timestamps_files = ['LaserBurstTimes' + suffix for suffix in file_suffixes ]
    SC_QE_data_file = ref_data_root + 'SC_QE_from_mono_SC_ED_20210618_MultiDay.txt'
    PD_QE_data_file = ref_data_root + 'SM05PD1B_QE.csv'
    save_str = 'QSW293_5mmPin'

    SC_photo_charges_by_wavelength_dict = measureDetectorCharges(wavelengths, SC_data_files, ref_burst_timestamps_files, SC_smoothing, max_SC_val, progress_plot_extra_title = 'SC', photocharge_neg = SC_current_neg, data_dir = data_dir, save_transitions_plot = 1, save_plot_prefix = save_str + '_SC_Readout_with_burst_ids_', show_process = 0, min_data_sep_to_identify_bursts = 10)
    SC_good_wavelengths = list(SC_photo_charges_by_wavelength_dict.keys())
    PD_photo_charges_by_wavelength_dict = measureDetectorCharges(wavelengths, PD_data_files, ref_burst_timestamps_files, PD_smoothing, max_PD_val, progress_plot_extra_title = 'PD', photocharge_neg = PD_current_neg, data_dir = data_dir, save_transitions_plot = 1, save_plot_prefix = save_str + '_PD_Readout_with_burst_ids_', show_process = 0, min_data_sep_to_identify_bursts = 3)
    PD_good_wavelengths = list(PD_photo_charges_by_wavelength_dict.keys())
    shared_good_wavelengths = can.intersection(SC_good_wavelengths, PD_good_wavelengths)
    print ('SC_good_wavelengths = ' + str(SC_good_wavelengths))
    print ('PD_good_wavelengths = ' + str(PD_good_wavelengths))
    print ('shared_good_wavelengths = ' + str(shared_good_wavelengths))

    SC_QE_interp = readInSCQEData(SC_QE_data_file)
    PD_QE_interp = readInPDQEData(PD_QE_data_file)

    SC_QE_plot = plt.plot(wavelengths, SC_QE_interp(wavelengths), c = 'k')[0]
    PD_QE_plot = plt.plot(wavelengths, PD_QE_interp(wavelengths), c = 'r')[0]
    plt.xlabel('Wavelength (nm)')
    plt.ylabel(r'SC QE (e-/$\gamma$)')
    plt.legend([SC_QE_plot, PD_QE_plot], ['SC QE (measured)', 'PD QE (from ThorLabs)'])
    plt.savefig(data_root + save_str + 'PD_and_SC_QEs.pdf')
    plt.close('all' )

    double_Al_wavelength, Double_Al_responsivity = can.readInColumnsToList(data_root + 'TwoBounceAl.dat', delimiter = ', ', n_ignore = 1, convert_to_float = 1)
    al_responsivity = plt.plot(double_Al_wavelength, Double_Al_responsivity, c = 'k')[0]
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Responsivity')
    plt.legend([al_responsivity], ['2 Reflections off Aluminum'])
    plt.savefig(data_root + save_str + 'TwoBounceAl.pdf')
    plt.close('all' )

    SC_over_PD_charge_ratio = [ np.mean(np.array(SC_photo_charges_by_wavelength_dict[wave]) / np.array(PD_photo_charges_by_wavelength_dict[wave])) for wave in shared_good_wavelengths ]
    #SC_photons_over_PD_electrons = SC_over_PD_charge_ratio / SC_QE_interp([wavelengths[i] for i in range(len(wavelengths)) if not(i in bad_data_indeces)])
    SC_photons_over_PD_electrons = SC_over_PD_charge_ratio / SC_QE_interp([wave for wave in shared_good_wavelengths])
    #SC_phot_over_PD_phot = SC_over_PD_charge_ratio / SC_QE_interp([wavelengths[i] for i in range(len(wavelengths)) if not(i in bad_data_indeces)]) * PD_QE_interp([wavelengths[i] for i in range(len(wavelengths)) if not(i in bad_data_indeces)])
    SC_phot_over_PD_phot = SC_over_PD_charge_ratio / SC_QE_interp([wave for wave in shared_good_wavelengths]) * PD_QE_interp([wave for wave in shared_good_wavelengths])


    min_SC_charge, max_SC_charge = [np.min([np.median(SC_photo_charges_by_wavelength_dict[wave]) for wave in SC_good_wavelengths]), np.max([np.median(SC_photo_charges_by_wavelength_dict[wave]) for wave in SC_good_wavelengths])]
    min_PD_charge, max_PD_charge = [np.min([np.median(PD_photo_charges_by_wavelength_dict[wave]) for wave in PD_good_wavelengths]), np.max([np.median(PD_photo_charges_by_wavelength_dict[wave]) for wave in PD_good_wavelengths])]
    f, axarr = plt.subplots(2,1, sharex = True, figsize = [10, 6] )
    #axarr[0].plot([wavelengths[i] for i in range(len(wavelengths)) if not(i in bad_data_indeces)], [np.median(SC_charges[i]) / max_SC_charge for i in range(len(SC_charges)) if not(i in bad_data_indeces)])
    #axarr[1].plot([wavelengths[i] for i in range(len(wavelengths)) if not(i in bad_data_indeces)], [np.median(PD_charges[i]) / max_PD_charge for i in range(len(PD_charges)) if not(i in bad_data_indeces)])
    axarr[0].plot(SC_good_wavelengths, [np.median(SC_photo_charges_by_wavelength_dict[wave]) / max_SC_charge for wave in SC_good_wavelengths ])
    axarr[1].plot(PD_good_wavelengths, [np.median(PD_photo_charges_by_wavelength_dict[wave]) / max_PD_charge for wave in PD_good_wavelengths ])
    axarr[1].set_xlabel('Laser wavelength (nm)')
    axarr[0].set_ylabel('Normalized SC charge/burst')
    axarr[0].set_yscale('log')
    axarr[1].set_ylabel('Normalized PD charge/burst')
    axarr[1].set_yscale('log')
    plt.savefig(data_dir + save_str + 'PD_and_SC_normalized_photocharges.pdf')
    plt.gcf().subplots_adjust(left=0.12)
    plt.show()
    can.saveListsToColumns([SC_good_wavelengths, [np.median(SC_photo_charges_by_wavelength_dict[wave]) for wave in SC_good_wavelengths]], save_str + 'SC_charges.csv', data_dir, sep = ', ', header = 'Wavelength(nm), SC charge (e-), PD charge (e-)')
    can.saveListsToColumns([PD_good_wavelengths, [np.median(PD_photo_charges_by_wavelength_dict[wave]) for wave in PD_good_wavelengths]], save_str + 'PD_charges.csv', data_dir, sep = ', ', header = 'Wavelength(nm), SC charge (e-), PD charge (e-)')

    #plt.plot([wavelengths[i] for i in range(len(wavelengths)) if not(i in bad_data_indeces)], SC_phot_over_PD_phot, c ='k')
    plt.plot(shared_good_wavelengths, SC_phot_over_PD_phot, color ='k', marker = 'x')
    plt.xlabel('Laser wavelength (nm)')
    plt.ylabel('SC Int. Charge / PD Int. Charge')
    plt.ylim([0, 20000])
    plt.savefig(data_dir + save_str + 'SC_charge_over_PD_charge.pdf')
    plt.close('all')

    #plt.plot([wavelengths[i] for i in range(len(wavelengths)) if not(i in bad_data_indeces)], SC_photons_over_PD_electrons, c ='k')
    plt.plot(shared_good_wavelengths, SC_photons_over_PD_electrons, color ='k', marker = 'x')
    plt.xlabel('Laser wavelength (nm)')
    plt.ylabel('SC photons / PD electrons')
    #plt.ylim([1500, 19000])
    plt.savefig(data_dir + save_str + 'SC_photons_over_PD_photons.pdf')
    plt.show()
    #can.saveListsToColumns([[wavelengths[i] for i in range(len(wavelengths)) if not(i in bad_data_indeces)], SC_photons_over_PD_electrons], save_str + 'SC_photons_over_PD_electrons.txt', data_dir, sep = ', ', header = 'Wavelength(nm), SC photons / PD electrons')
    can.saveListsToColumns([shared_good_wavelengths, SC_photons_over_PD_electrons], save_str + 'SC_photons_over_PD_electrons.txt', data_dir, sep = ', ', header = 'Wavelength(nm), SC photons / PD electrons')

    #plt.plot([wavelengths[i] for i in range(len(wavelengths)) if not(i in bad_data_indeces)], SC_phot_over_PD_phot, c ='k')
    plt.plot(shared_good_wavelengths, SC_phot_over_PD_phot, color ='k', marker = 'x')
    plt.xlabel('Laser wavelength (nm)')
    plt.ylabel('SC photons / PD photons')
    #plt.ylim([1900, 4100])
    plt.savefig(data_dir + save_str + 'SC_photons_over_PD_photons.pdf')
    plt.show()
    #can.saveListsToColumns([[wavelengths[i] for i in range(len(wavelengths)) if not(i in bad_data_indeces)], SC_phot_over_PD_phot], save_str + 'SC_photons_over_PD_photons.txt', data_dir, sep = ', ', header = 'Wavelength(nm), SC photons / PD photons')
    can.saveListsToColumns([shared_good_wavelengths, SC_phot_over_PD_phot], save_str + 'SC_photons_over_PD_photons.txt', data_dir, sep = ', ', header = 'Wavelength(nm), SC photons / PD photons')
