"""
authors: Sasha Brownsberger (sashab@alumni.stanford.edu)

Modifications:
2021/07/13: First written
2021/07/14: Added CompareTransmissionToAluminum function to compare current best measurement of CBP transmission to
                two bounces off of aluminum.

Description:
This file contains a series of stand alone scrtips, designed to be run once to plot some component of the CBP data.
Each function has a sample of how it can be used to generate a plot.
These scripts should be kept as user friendsly as possible - they are designed to be run quickly and easily.  The
    hard part of the analysis should be done by the work-horse code, currently contained in AnalyzeFrenchWavelengthScans.py
"""
import matplotlib.pyplot as plt
import cantrips as can
import numpy as np


def showCurrentsAndCurrentRatios(data_files, figsize = [12, 8], ylabels = [r'Photocurrent (C)', r'Photocurrent ratios'], xlabel = 'Wavelength (nm)', colors = ['r', 'k'], legend_strs = [],
                                 save_data_dir = '', save_file_name = None):
    """
    Authors: Sasha Brownsberger (sashab@alumni.stanford.edu)
    Modifications:
    2021/07/13: First written

    Make a plot of the photocurrents (in electrons) for the photodiode for various obsrvation settings.  Also plot the ratio in the following plot frame.
    With the example below, make these plots as we switch between a 1mm and a 5mm pinhole.

    import FrenchDataSetupExtraScripts as fdata
    dir_root = '../data/CBP_throughput_calib_data/ut20210712/'
    legend_strs = [['Laser @ Max, 1mm hole', 'Laser @ Max, 5mm hole'], ['Laser @ Max, 1mm hole', 'Laser @ Max, 5mm hole']]
    fdata.showCBPTransmissionCurve([dir_root + '1mmPin_sequenceB/' + 'QSWMax_1mmPinSC_photons_over_PD_photons.txt', dir_root + '5mmPin_sequenceB/' + 'QSWMax_5mmPinSC_photons_over_PD_photons.txt'], legend_strs = legend_strs, save_data_dir = dir_root, save_file_name = 'CBP_transmissivity_1mm_vs_5mm_holes.pdf')
    """
    f, axarr = plt.subplots(2, 1, sharex = True, figsize = figsize)
    unscaled_plots = []
    data_by_wavelengths = []
    for i in range(len(data_files)):
        data = can.readInColumnsToList(data_files[i], n_ignore = 1, delimiter = ', ', convert_to_float = 1)
        unscaled_plots = unscaled_plots + [axarr[0].plot(data[0], data[1], color = colors[i], )[0]]
        axarr[0].set_xlabel(xlabel)
        axarr[0].set_ylabel(ylabels[0])
        data_by_wavelengths = data_by_wavelengths + [{data[0][i]:data[1][i] for i in range(len(data[0]))}]
    shared_wavelengths = can.intersection(list(data_by_wavelengths[0].keys()), list(data_by_wavelengths[1].keys()))
    max_val = np.max(data[1])
    scaled_plot = axarr[1].plot(shared_wavelengths, [data_by_wavelengths[0][wave] / data_by_wavelengths[1][wave] for wave in shared_wavelengths], c = 'r', marker = 'x')[0]
    axarr[1].axhline(1.0, c = 'k', linestyle = '--')
    axarr[1].set_xlabel(xlabel)
    axarr[1].set_ylabel(ylabels[1])

    if len(legend_strs) == len(data_files):
        axarr[0].legend(unscaled_plots, legend_strs[0])
        axarr[1].legend([scaled_plot], legend_strs[1])
    if save_file_name != None:
        plt.savefig(save_data_dir + save_file_name)
    plt.show()


def showCBPTransmissionCurve(data_files, figsize = [12, 8], ylabels = ['SC photons / PD photons', 'Median-normalized SC photons / PD photons'], xlabel = 'Wavelength (nm)', colors = ['r', 'g', 'k'], legend_strs = [],
                                 save_data_dir = '', save_file_name = None):
    """
    Authors: Sasha Brownsberger (sashab@alumni.stanford.edu)
    Modifications:
    2021/07/13: First written

    Make plots of the ratio of solar cell and photodiode photon signals for various settings.  In the top plot, show the plots in an absolute
        sense.  In the bottom plot, show the plots when both functions are normalized.
    In this example, we make these plots for a 1mmPinhole and a 5mmPinhole illuminating the CBP, with both cases normalized to have median value 1.

    import FrenchDataSetupExtraScripts as fdata
    dir_root = '../data/CBP_throughput_calib_data/'
    legend_strs = [['Laser @ Max, 1mm hole', 'Laser @ Max, 5mm hole', 'CBP Off Target'], ['Laser @ Max, 1mm hole', 'Laser @ Max, 5mm hole', 'CBP Off Target']]
    fdata.showCBPTransmissionCurve([dir_root + 'ut20210712/1mmPin_sequenceB/' + 'QSWMax_1mmPinSC_photons_over_PD_photons.txt', dir_root + 'ut20210712/5mmPin_sequenceB/' + 'QSWMax_5mmPinSC_photons_over_PD_photons.txt', dir_root + 'ut20210713/CBPOffTarget/' + 'QSWMax_5mmPin_OffTargetSC_photons_over_PD_photons.txt'], legend_strs = legend_strs, save_data_dir = dir_root, save_file_name = 'CBP_transmissivity_1mm_vs_5mm_vs_CBPOffTarget.pdf')
    """
    f, axarr = plt.subplots(2, 1, sharex = True, figsize = figsize)
    data_by_wavelengths = []
    unscaled_plots = []
    for i in range(len(data_files)):
        data = can.readInColumnsToList(data_files[i], n_ignore = 1, delimiter = ', ', convert_to_float = 1)
        unscaled_plots = unscaled_plots + [axarr[0].plot(data[0], data[1], color = colors[i], marker = 'x')[0]]
        axarr[0].set_xlabel(xlabel)
        axarr[0].set_ylabel(ylabels[0])
        data_by_wavelengths = data_by_wavelengths + [{data[0][i]:data[1][i] for i in range(len(data[0]))}]
    axarr[0].legend(unscaled_plots, legend_strs[0])

    scaled_plots = []
    for i in range(len(data_files)):
        data = can.readInColumnsToList(data_files[i], n_ignore = 1, delimiter = ', ', convert_to_float = 1)
        scaled_plots = scaled_plots + [axarr[1].plot(data[0], data[1] / np.median(data[1]), color = colors[i], marker = 'x')[0]]
        axarr[1].set_xlabel(xlabel)
        axarr[1].set_ylabel(ylabels[1])
        data_by_wavelengths = data_by_wavelengths + [{data[0][i]:data[1][i] for i in range(len(data[0]))}]
    axarr[1].legend(scaled_plots, legend_strs[1])
    if save_file_name != None:
        plt.savefig(save_data_dir + save_file_name)
    plt.show()


def checkLinearityResults(data_files, figsize = [12, 9], ylabels = ['PD Charge / QE', 'SC charge / QE', 'R=SC/PD, Scaled by R @ Max'], xlabel = 'Wavelength (nm)', colors = ['r', 'blue', 'green', 'orange','magenta','k', 'cyan'], legend_strs = [],
                                 save_data_dir = '', save_file_name = None):
    """
    Authors: Sasha Brownsberger (sashab@alumni.stanford.edu)
    Modifications:
    2021/07/13: First written

    Make a plot of the photocurrent ratio between the monitoring solar cell and photodiode over a range of observation settings.  In the
       top plot, these ratios are plotted withot modification.  In the bottom plot, they are normalized to a specified reference curve.
       The question answered here is whether the system is linear (same solar cell response per photodiode response) as the observing
       conditions are changed.
    In this example, we show the linearity over a range of laser intensity ("QSW") settings.

    import FrenchDataSetupExtraScripts as fdata
    dir_root = '../data/CBP_throughput_calib_data/ut20210713/'
    QSWs = ['Max', '300', '299', '296','293','289', '288']
    data_files = [[dir_root + 'LinearityScans/QSW' + str(QSW) + '/' + 'QSW' + str(QSW) + '_5mmPinPD_charges.csv' for QSW in QSWs], [dir_root + 'LinearityScans/QSW' + str(QSW) + '/' + 'QSW' + str(QSW) + '_5mmPinSC_charges.csv' for QSW in QSWs], [dir_root + 'LinearityScans/QSW' + str(QSW) + '/' + 'QSW' + str(QSW) + '_5mmPinSC_photons_over_PD_photons.txt' for QSW in QSWs]]
    legend_strs = [['QSW' + '-' + QSW for QSW in QSWs], ['QSW' + '-' + QSW for QSW in QSWs], ['QSW' + '-' + QSW for QSW in QSWs]]
    fdata.checkLinearityResults(data_files, legend_strs = legend_strs, save_data_dir = dir_root, save_file_name = 'CBP_linearity_varyingQSWs.pdf')
    """
    f, axarr = plt.subplots(3, 1, sharex = True, figsize = figsize)
    data_by_wavelengths = []
    PD_plots = []
    SC_plots = []
    for i in range(len(data_files[0])):
        data = can.readInColumnsToList(data_files[0][i], n_ignore = 1, delimiter = ', ', convert_to_float = 1)
        print ('data = ' + str(data))
        print ('colors = ' + str(colors))
        PD_plots = PD_plots + [axarr[0].plot(data[0], data[1], color = colors[i], marker = 'x')[0]]
        axarr[0].set_xlabel(xlabel)
        axarr[0].set_ylabel(ylabels[0])
        data_by_wavelengths = data_by_wavelengths + [{data[0][i]:data[1][i] for i in range(len(data[0]))}]
    axarr[0].legend(PD_plots, legend_strs[0])
    scaled_plots = []

    for i in range(len(data_files[1])):
        data = can.readInColumnsToList(data_files[1][i], n_ignore = 1, delimiter = ', ', convert_to_float = 1)
        print ('data = ' + str(data))
        print ('colors = ' + str(colors))
        SC_plots = SC_plots + [axarr[1].plot(data[0], data[1], color = colors[i], marker = 'x')[0]]
        axarr[1].set_xlabel(xlabel)
        axarr[1].set_ylabel(ylabels[1])
        data_by_wavelengths = data_by_wavelengths + [{data[0][i]:data[1][i] for i in range(len(data[0]))}]
    axarr[1].legend(SC_plots, legend_strs[1])

    ref_data = can.readInColumnsToList(data_files[2][0], n_ignore = 1, delimiter = ', ', convert_to_float = 1)
    ref_data_by_wavelength = {ref_data[0][i]:ref_data[1][i] for i in range(len(ref_data[0]))}
    ref_wavelengths = list(ref_data_by_wavelength.keys())
    for i in range(len(data_files[2])):
        data = can.readInColumnsToList(data_files[2][i], n_ignore = 1, delimiter = ', ', convert_to_float = 1)
        print ('data = ' + str(data))
        print ('colors = ' + str(colors))
        SC_plots = SC_plots + [axarr[2].plot([wave for wave in data[0] if wave in ref_wavelengths ], [data[1][i] / ref_data_by_wavelength[data[0][i]] for i in range(len(data[1])) if data[0][i] in ref_wavelengths ], color = colors[i], marker = 'x')[0]]
        axarr[2].set_xlabel(xlabel)
        axarr[2].set_ylabel(ylabels[2])
        data_by_wavelengths = data_by_wavelengths + [{data[0][i]:data[1][i] for i in range(len(data[0]))}]
        ylims = [0.6, 1.4]
        axarr[2].set_ylim(ylims)
    axarr[2].legend(SC_plots, legend_strs[2])

    if save_file_name != None:
        plt.savefig(save_data_dir + save_file_name)
    plt.show()


def CompareTransmissionToAluminum(data_files, figsize = [12, 6], ylabels = ['Transmission, normalized to 1 at 700nm'], xlabel = 'Wavelength (nm)', colors = ['r', 'blue', 'green', 'orange','magenta','k', 'cyan'], legend_strs = [],
                                 save_data_dir = '', save_file_name = None, al_ref_file = '../data/CBP_throughput_calib_data/TwoBounceAl.dat', pivot_wavelength = 700.0):
    """
    Authors: Sasha Brownsberger (sashab@alumni.stanford.edu)
    Modifications:
    2021/07/14: First written

    Make a plot of the CBP transmission, measured as the ratio of solar cell photons to photodiode photons, to the transmission expected
        from two bounces off of aluminum surfaces.

    import FrenchDataSetupExtraScripts as fdata
    dir_root = '../data/CBP_throughput_calib_data/ut20210713/'
    QSWs = ['Max']
    data_files = [dir_root + 'LinearityScans/QSW' + str(QSW) + '/' + 'QSW' + str(QSW) + '_5mmPinSC_photons_over_PD_photons.txt' for QSW in QSWs]
    legend_strs = ['QSW' + '-' + QSW + '; 5mmhole' for QSW in QSWs] + ['Theoretical 2 bounces off Al']
    save_file_name = 'CBP_transmission_vs_2XAluminum.pdf'
    fdata.CompareTransmissionToAluminum(data_files, legend_strs = legend_strs, save_data_dir = dir_root, save_file_name = save_file_name )
    """
    f, axarr = plt.subplots(1, 1, sharex = True, figsize = figsize)
    data_by_wavelengths = []
    transm_plots = []
    for i in range(len(data_files)):
        data = can.readInColumnsToList(data_files[i], n_ignore = 1, delimiter = ', ', convert_to_float = 1)
        print ('data = ' + str(data))
        print ('colors = ' + str(colors))
        data_by_wavelengths = data_by_wavelengths + [{data[0][i]:data[1][i] for i in range(len(data[0]))}]
        transm_plots = transm_plots + [axarr.plot(data[0], np.array(data[1]) / data_by_wavelengths[0][pivot_wavelength], color = colors[i], marker = 'x')[0]]
        axarr.set_xlabel(xlabel)
        axarr.set_ylabel(ylabels[0])
    al_data = can.readInColumnsToList(al_ref_file, n_ignore = 1, delimiter = ', ', convert_to_float = 1)
    al_data_by_wavelengths = {al_data[0][i]:al_data[1][i] for i in range(len(al_data[0]))}
    al_plot = axarr.plot(al_data[0], np.array(al_data[1]) / al_data_by_wavelengths[pivot_wavelength], color = 'k', linestyle = '--')[0]
    axarr.legend(transm_plots + [al_plot], legend_strs)

    if save_file_name != None:
        plt.savefig(save_data_dir + save_file_name)
    plt.show()
