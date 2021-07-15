import cantrips as can
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy
import AstronomicalParameterArchive as apa

"""
authors: Sasha Brownsberger (SB, sashab@alumni.stanford.edu)

Modifications:
2021/07/04 - 2021/07/14: SB: First written and quickly developed.
2021/07/14: SB: Added documentation, removed unneeded functions, removed debugging/extraneous code.


Description:
This function takes in a scan of the CBP throughput and extracts measurements of the solar cell and photodiode
    photocharges.  Each monochromatic data point in the scan should consist of three files: solar cell current
    readout, photodiode current readout, and start and end times of the bursts.
"""

def groupBurstStartGuesses(ungrouped_guess_indeces, min_sep = 20, verbose = 0):
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
    while not(all_guesses_grouped):
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
    if verbose: print ('guess_groups = ' + str(guess_groups ))
    return guess_groups

def findTransitions(arr, n_target_transitions, max_n_sig_to_identify = 14, min_n_sig_to_identify = 1, pos_or_neg = 1, verbose = 0, max_iterations = 20, min_data_sep = 10, show_process = 0):
    """
    authors: Sasha Brownsberger (sashab@alumni.stanford.edu)

    Modifications:
    2021/07/04 - 2021/07/14: First written.

    Description:
    Takes an array of indeces of an array (integers) and sorts them into "groups" of nearby indeces.  For an
        index to be placed in a group, it must be within min_sep distance from any other component of that group.

    This function is used to separate distinct peaks above some detection threshold.
    """
    if verbose: print ('Finding transitions...')
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
            transitions = groupBurstStartGuesses(transition_indeces, verbose = verbose, min_sep = min_data_sep)
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

def pieceFitCharge(xs, ys, piecewise_parts_indeces, verbose = 0, poly_fit_order = 2):
    """
    authors: Sasha Brownsberger (sashab@alumni.stanford.edu)

    Modifications:
    2021/07/04 - 2021/07/14: First written.

    Description:
    Fit the charge of two adjacent sections of a photodetector (solar cell or photodiode) readout (bursts on, charging with
        photocurrent and dark current, or bursts off, charging with dark current).
    The fit performs a fit of the specified polynomial order (default 2) to each section of the piecewise sections.  The
        pivot between the piecewise data sections is also fitted and returned.

    Issues:
    For short bursts, the fits can be underconstrained or just outright fail because there are insufficient data points
       for a meaningful fit result.
    """
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
                centered_poly_fit = np.poly1d(np.polyfit(centered_xs, piecewise_ys, poly_fit_order))
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
    """
    authors: Sasha Brownsberger (sashab@alumni.stanford.edu)

    Modifications:
    2021/07/04 - 2021/07/14: First written.

    Description:
    Given the piecewise fits of the accumullated charge, this function does the final arithmetic.  For each illuminated charging period,
        the function calculates the differences between the beginning and end.  The dark current during this charging period is
        calculated by measuring the slope of the prior and subsequent dark charging times and averaging those slopes.  Multiplying the
        charging time by the dark current, we calculate the total accumulated dark charge.  We subtract that value from the total charge
        accumulated during the illuminated charging period to acquire the total accumulated charge.
    The function returns a list of length equal to the number of observed data points (typically number of observed wavelengths).  Each
        component of the larger returned list is itself a list of length equal to the number of bursts.

    Issues:
    We do not presently report uncertainties in the inferred photocharges.  Having estimates of those, ideally from
        uncertainties in the underlying fit, would be nice.
    """
    charges = [[transition_charges[j][2 * i + 2] - transition_charges[j][2 * i + 1] for i in range(n_bursts)] for j in range(len(orig_data)) if not (j in bad_data_indeces)]
    charge_times = [[time_seps[j][2 * i + 2] - time_seps[j][2 * i + 1] for i in range(n_bursts)] for j in range(len(orig_data)) if not (j in bad_data_indeces)]
    dark_charges = [[transition_charges[j][2 * i + 1] - transition_charges[j][2 * i] for i in range(n_bursts + 1)] for j in range(len(orig_data)) if not (j in bad_data_indeces)]
    dark_charge_times = [ [time_seps[j][2 * i + 1] - time_seps[j][2 * i] for i in range(n_bursts + 1)] for j in range(len(orig_data)) if not (j in bad_data_indeces) ]

    dark_currents = [[dark_charges[j][i] / dark_charge_times[j][i] for i in range(n_bursts + 1)] for j in range(len(dark_charges))]
    dark_charges_during_illumination = [[(dark_currents[j][i + 1] + dark_currents[j][i]) / 2 * charge_times[j][i] for i in range(n_bursts)] for j in range(len(charges)) ]

    photo_charges = [[(charges[j][i] - dark_charges_during_illumination[j][i]) * (-1 if current_neg else 1) for i in range(n_bursts)] for j in range(len(charges)) ]
    return photo_charges

def readInSCQEData(SC_QE_data_file):
    """
    authors: Sasha Brownsberger (sashab@alumni.stanford.edu)

    Modifications:
    2021/07/04 - 2021/07/14: First written.

    Description:
    Read in the laboratory measured QE of the monitoring solar cell and turn those discretely sampled data points into a single
        queryable curve of QE vs wavelength.
    The measured SC QE has many data points, with some underlying uncertainty.  So we must (and here do) average over the
        distribution of measured QE points and THEN create a smooth interpolator of QE in wavelength.
    """
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
    """
    authors: Sasha Brownsberger (sashab@alumni.stanford.edu)

    Modifications:
    2021/07/04 - 2021/07/14: First written.

    Description:
    Read in the laboratory measured QE of the monitoring photodiode and turn sampled data points into a single queryable
        curve of QE vs wavelength.
    The measured photodiode QE is based on Thorlabs reported measurements.  Thus, unlike the solar cell QE function, no
        smoothing over scattered points is necessary.  However, we do need to convert the reported values of Amps/Watt
        into our desired data set of e-/photon.
    """
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
                           show_burst_fits = 1, progress_plot_extra_title = '', save_transitions_plot = 1, save_plot_prefix = None,
                           show_burst_id_process = 0):
    """
    authors: Sasha Brownsberger (sashab@alumni.stanford.edu)

    Modifications:
    2021/07/04 - 2021/07/14: First written.

    Description:
    This function does most of the work.  For a list of data files, associated illumination wavelengths, and the files
       that contained the timestamps of the burst starts and ends, this function returns measurements of the photocharge
       that each of the bursts generated in the detector.
    This function does most of the work in this analysis.  It is the function that loops all of the above functions
       together.
    """
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

    #data_smoothed = [smoothArray(arr, smoothing) for arr in raw_charges]
    data_smoothed = [can.smoothList(arr, smooth_type = 'boxcar', averaging = 'mean', params = [smoothing]) for arr in raw_charges]
    data_to_find_transitions = [np.gradient(np.gradient(arr)) for arr in data_smoothed]

    ref_times = [can.readInColumnsToList(file, data_dir, delimiter = ', ', n_ignore = 1, convert_to_float = 1) for file in ref_burst_timestamps_files]
    n_bursts = len(ref_times[0][0])

    burst_on_indeces = [[] for arr in data_smoothed]
    burst_off_indeces = [[] for arr in data_smoothed]
    for i in range(len(data_smoothed)):
        arr = data_to_find_transitions[i]
        burst_on_indeces[i] = findTransitions(arr[smoothing:-smoothing], n_bursts, max_n_sig_to_identify = max_std_to_identify_bursts, min_n_sig_to_identify = min_std_to_identify_bursts, pos_or_neg = (-1 if photocharge_neg else 1), min_data_sep = min_data_sep_to_identify_bursts, show_process = show_burst_id_process, verbose = show_burst_id_process)
        burst_off_indeces[i] = findTransitions(arr[smoothing:-smoothing], n_bursts, max_n_sig_to_identify = max_std_to_identify_bursts, min_n_sig_to_identify = min_std_to_identify_bursts, pos_or_neg = (1 if photocharge_neg else -1), min_data_sep = min_data_sep_to_identify_bursts, show_process = show_burst_id_process, verbose = show_burst_id_process)
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

    bad_data_indeces = can.union([bad_data_indeces, failed_burst_id_indeces])

    burst_off_before_burst_on = [[burst_off_indeces[i][1][j-1] > burst_on_indeces[i][1][j] for j in range(1, len(burst_off_indeces[i][1]))] for i in range(len(burst_on_indeces))]
    burst_on_before_next_burst_off = [[burst_on_indeces[i][1][j] > burst_off_indeces[i][1][j] for j in range(1, len(burst_off_indeces[i][1]))] for i in range(len(burst_on_indeces))]


    bursts_out_of_order_indeces = [i for i in range(len(burst_on_indeces)) if (np.any(burst_off_before_burst_on[i]) or np.any(burst_on_before_next_burst_off[i])) ]
    if len(bursts_out_of_order_indeces ) > 0:
        print ('The following files identified burst starts and ends out of order. We will mark it as a bad observation: ')
        print ([data_files[i] for i in bursts_out_of_order_indeces ])
    bad_data_indeces = can.union([bad_data_indeces, bursts_out_of_order_indeces])

    corrected_times = [[] for i in range(len(uncorrected_times))]
    time_sep_fits = [[] for i in range(len(uncorrected_times))]
    for i in range(0, len(uncorrected_times)):
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

    #print ('bad_data_indeces = ' + str(bad_data_indeces))
    if show_burst_fits:
        for j in range(len(data_sets)):
            if not (j in bad_data_indeces):
                print ('[len(piece_fits[j][0]), len(piece_fits[j][1])] = ' + str([len(piece_fits[j][0]), len(piece_fits[j][1])]))
                f, axarr = plt.subplots(3,1, figsize = [12, 9])
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
                [axarr[2].plot(transition_times[j][piece_fits[j][0][i]:piece_fits[j][0][i+1]], piece_fits[j][1][i]( transition_times[j][piece_fits[j][0][i]:piece_fits[j][0][i+1]] ) * 10.0 ** 6.0, c = 'red') for i in range(len(piece_fits[j][1]))]
                plt.gcf().subplots_adjust(left=0.12)
                plt.draw()
                plt.pause(0.25)

                if save_transitions_plot and save_plot_prefix != None:
                    plt.savefig(data_dir + save_plot_prefix + str(wavelengths[j]) + '.pdf')
                plt.close('all')

    return charges_by_wavelength_dict


if __name__=="__main__":
    """
    This function is designed to be run from the command line (NOT directly in a Python environment), as follows:
    (bash) $ python AnalyzeFrenchWavelengthScans.py

    All of the variables that you might want to change, scan to scan, are directly below.  These include:
    wavelengths    => The scanned wavelengths.
    data_root      => The super directory where all of the data generally accumulated (probably don't
                          want to change often).
    QSWs           => The laser QSW (intensity) settings.  Must be a list, even if only one value.
    Pinholes       => The pinhole or iris setting feeding the CBP used in the observations.  Must be a
                          list, even if only one value.
    other_suffixes => The other suffixes added to the file names.  Must be a list, even if only one value.

    """

    ##### HERE IS THE STUFF THAT YOU MIGHT WANT TO MODIFY FOR EACH RUN #####

    wavelengths = np.arange(350, 1050, 5)
    wavelengths = np.arange(1000, 1050, 5)
    wavelengths = np.arange(400, 450, 5)
    data_root = '../data/CBP_throughput_calib_data/'
    #To add additional layers of file suffixes, append additional layers of list in the flattenListOfLists function below.
    #   Even if your suffix categories are only one element long, they must be in a list.
    QSWs = ['Max']
    pinholes = ['_5mmPin3']
    other_suffixes = ['_LinTest']
    data_dir = data_root + 'ut20210713/LinearityScans/QSWMax/'
    save_str = 'QSWMax_5mmPin'

    #### [END STUFF YOU FREQUENTLY WANT TO MODIFY] #####

    #### HERE ARE SOME CODED PARAMETERS THAT YOU COULD CHANGE, BUT ONLY IF THE DATA STREAM SETUP CHANGES IN A WAY THAT DOES NOT HAPPEN REGULARLY
    file_suffixes = can.flattenListOfLists([[[['_Wave' + str(wave) + '_QSW' + str(QSW) + str(pinhole) + str(other) + '.csv' for wave in wavelengths  ] for QSW in QSWs] for pinhole in pinholes ] for other in other_suffixes], fully_flatten = 1)
    print ('file_suffixes = '  + str(file_suffixes))
    #If the base names of the saved solar cell, photodiode, or burst times are changed, these files should be changed.
    SC_data_files = [ 'SolarCell_fromB2987A' + suffix for suffix in file_suffixes ]
    PD_data_files = ['Photodiode_fromKeithley' + suffix for suffix in file_suffixes ]
    ref_burst_timestamps_files = ['LaserBurstTimes' + suffix for suffix in file_suffixes ]

    #The data files from which the reference QEs of the solar cell and photodiode are pulled.
    #   If you want a new QE curve, here is where you put the new QE file.
    ref_data_root = '../refCalData/'
    SC_QE_data_file = ref_data_root + 'SC_QE_from_mono_SC_ED_20210618_MultiDay.txt'
    PD_QE_data_file = ref_data_root + 'SM05PD1B_QE.csv'
    SC_current_neg = 0
    PD_current_neg = 1
    max_SC_val = 2e-6
    max_PD_val = 1e-3
    SC_smoothing = 10 #How many adjacent solar cell samples should we sum when looking for the bursts?
    PD_smoothing = 1 #How many adjacent photodiode samples should we sum when looking for the bursts?

    #### [END STUFF YOU MIGHT WANT TO MODIFY] #####

    wavelength_range = [min(wavelengths),  max(wavelengths)]
    astro_arch = apa.AstronomicalParameterArchive()
    eletron_charge_in_C = astro_arch.getElectronCharge()
    SC_QE_interp = readInSCQEData(SC_QE_data_file)
    PD_QE_interp = readInPDQEData(PD_QE_data_file)
    #Make plots of the background data that we used in our measurements
    SC_QE_plot = plt.plot(wavelengths, SC_QE_interp(wavelengths), c = 'k')[0]
    PD_QE_plot = plt.plot(wavelengths, PD_QE_interp(wavelengths), c = 'r')[0]
    plt.xlabel('Wavelength (nm)')
    plt.ylabel(r'SC QE (e-/$\gamma$)')
    plt.legend([SC_QE_plot, PD_QE_plot], ['SC QE (measured)', 'PD QE (from ThorLabs)'])
    plt.savefig('../data/CBP_throughput_calib_data/' + save_str + 'PD_and_SC_QEs.pdf')
    plt.close('all' )

    double_Al_wavelength, Double_Al_responsivity = can.readInColumnsToList('../data/CBP_throughput_calib_data/' + 'TwoBounceAl.dat', delimiter = ', ', n_ignore = 1, convert_to_float = 1)
    al_responsivity = plt.plot(double_Al_wavelength, Double_Al_responsivity, c = 'k')[0]
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Responsivity')
    plt.legend([al_responsivity], ['2 Reflections off Aluminum'])
    plt.savefig('../data/CBP_throughput_calib_data/' + save_str + 'TwoBounceAl.pdf')
    plt.close('all' )

    #Start the analysis, going through each solar cell data file and looking for the correct number of bursts. Then do the same of the photodiode.
    SC_photo_charges_by_wavelength_dict = measureDetectorCharges(wavelengths, SC_data_files, ref_burst_timestamps_files, SC_smoothing, max_SC_val, progress_plot_extra_title = 'SC', photocharge_neg = SC_current_neg, data_dir = data_dir, save_transitions_plot = 1, save_plot_prefix = save_str + '_SC_Readout_with_burst_ids_', show_burst_id_process = 0, min_data_sep_to_identify_bursts = 10)
    SC_good_wavelengths = list(SC_photo_charges_by_wavelength_dict.keys())
    PD_photo_charges_by_wavelength_dict = measureDetectorCharges(wavelengths, PD_data_files, ref_burst_timestamps_files, PD_smoothing, max_PD_val, progress_plot_extra_title = 'PD', photocharge_neg = PD_current_neg, data_dir = data_dir, save_transitions_plot = 1, save_plot_prefix = save_str + '_PD_Readout_with_burst_ids_', show_burst_id_process = 0, min_data_sep_to_identify_bursts = 3)
    PD_good_wavelengths = list(PD_photo_charges_by_wavelength_dict.keys())
    shared_good_wavelengths = can.intersection(SC_good_wavelengths, PD_good_wavelengths)
    print ('SC_good_wavelengths = ' + str(SC_good_wavelengths))
    print ('PD_good_wavelengths = ' + str(PD_good_wavelengths))
    print ('shared_good_wavelengths = ' + str(shared_good_wavelengths))

    #Compute the quantities we want by averaging over the bursts at each wavelength, and applying the appropriate conversions.
    SC_over_PD_charge_ratio = [ np.mean(np.array(SC_photo_charges_by_wavelength_dict[wave]) / np.array(PD_photo_charges_by_wavelength_dict[wave])) for wave in shared_good_wavelengths ]
    SC_photons_over_PD_electrons = SC_over_PD_charge_ratio / SC_QE_interp([wave for wave in shared_good_wavelengths])
    SC_phot_over_PD_phot = SC_over_PD_charge_ratio / SC_QE_interp([wave for wave in shared_good_wavelengths]) * PD_QE_interp([wave for wave in shared_good_wavelengths])
    min_SC_charge, max_SC_charge = [np.min([np.median(SC_photo_charges_by_wavelength_dict[wave]) for wave in SC_good_wavelengths]), np.max([np.median(SC_photo_charges_by_wavelength_dict[wave]) for wave in SC_good_wavelengths])]
    min_PD_charge, max_PD_charge = [np.min([np.median(PD_photo_charges_by_wavelength_dict[wave]) for wave in PD_good_wavelengths]), np.max([np.median(PD_photo_charges_by_wavelength_dict[wave]) for wave in PD_good_wavelengths])]

    #Plot the results and save them to plain text, .csv files.
    f, axarr = plt.subplots(2,1, sharex = True, figsize = [10, 6] )
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
    can.saveListsToColumns([SC_good_wavelengths, [np.median(SC_photo_charges_by_wavelength_dict[wave]) / eletron_charge_in_C * 10.0 ** -6.0  for wave in SC_good_wavelengths]], save_str + 'SC_charges.csv', data_dir, sep = ', ', header = 'Wavelength(nm), SC charge (10^6 e-), PD charge (10^6 e-)')
    can.saveListsToColumns([PD_good_wavelengths, [np.median(PD_photo_charges_by_wavelength_dict[wave]) / eletron_charge_in_C * 10.0 ** -6.0  for wave in PD_good_wavelengths]], save_str + 'PD_charges.csv', data_dir, sep = ', ', header = 'Wavelength(nm), SC charge (10^6 e-), PD charge (10^6 e-)')

    plt.plot(shared_good_wavelengths, SC_phot_over_PD_phot, color ='k', marker = 'x')
    plt.xlabel('Laser wavelength (nm)')
    plt.ylabel('SC Int. Charge / PD Int. Charge')
    plt.ylim([0, 20000])
    plt.savefig(data_dir + save_str + 'SC_charge_over_PD_charge.pdf')
    plt.close('all')

    plt.plot(shared_good_wavelengths, SC_photons_over_PD_electrons, color ='k', marker = 'x')
    plt.xlabel('Laser wavelength (nm)')
    plt.ylabel('SC photons / PD electrons')
    plt.savefig(data_dir + save_str + 'SC_photons_over_PD_photons.pdf')
    plt.show()
    can.saveListsToColumns([shared_good_wavelengths, SC_photons_over_PD_electrons], save_str + 'SC_photons_over_PD_electrons.txt', data_dir, sep = ', ', header = 'Wavelength(nm), SC photons / PD electrons')

    plt.plot(shared_good_wavelengths, SC_phot_over_PD_phot, color ='k', marker = 'x')
    plt.xlabel('Laser wavelength (nm)')
    plt.ylabel('SC photons / PD photons')
    plt.savefig(data_dir + save_str + 'SC_photons_over_PD_photons.pdf')
    plt.show()
    can.saveListsToColumns([shared_good_wavelengths, SC_phot_over_PD_phot], save_str + 'SC_photons_over_PD_photons.txt', data_dir, sep = ', ', header = 'Wavelength(nm), SC photons / PD photons')
