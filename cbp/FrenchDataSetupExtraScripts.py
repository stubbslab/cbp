import matplotlib.pyplot as plt
import cantrips as can
import numpy as np

"""
import FrenchDataSetupExtraScripts as fdata
dir_root = '../data/CBP_throughput_calib_data/ut20210712/'
legend_strs = [['PD: Max Power, 1mm hole', 'PD: Max Power, 5mm hole'], ['RATIO: PD @ Max, 1mm hole / PD @ Max, 5mm hole ']]
fdata.showCurrentsAndCurrentRatios([dir_root + '1mmPin_sequenceB/' + 'QSWMax_1mmPinPD_charges.csv', dir_root + '5mmPin_sequenceB/' + 'QSWMax_5mmPinPD_charges.csv'], legend_strs = legend_strs, save_data_dir = dir_root, save_file_name = 'PD_Photocurrents_1mm_5mm_ratios.pdf')
"""
def showCurrentsAndCurrentRatios(data_files, figsize = [12, 8], ylabels = [r'Photocurrent (C)', r'Photocurrent ratios'], xlabel = 'Wavelength (nm)', colors = ['r', 'k'], legend_strs = [],
                                 save_data_dir = '', save_file_name = None):
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


"""
import FrenchDataSetupExtraScripts as fdata
dir_root = '../data/CBP_throughput_calib_data/ut20210712/'
legend_strs = [['Laser @ Max, 1mm hole', 'Laser @ Max, 5mm hole'], ['Laser @ Max, 1mm hole', 'Laser @ Max, 5mm hole']]
fdata.showCBPTransmissionCurve([dir_root + '1mmPin_sequenceB/' + 'QSWMax_1mmPinSC_photons_over_PD_photons.txt', dir_root + '5mmPin_sequenceB/' + 'QSWMax_5mmPinSC_photons_over_PD_photons.txt'], legend_strs = legend_strs, save_data_dir = dir_root, save_file_name = 'CBP_transmissivity_1mm_vs_5mm_holes.pdf')
"""
def showCBPTransmissionCurve(data_files, figsize = [12, 8], ylabels = ['SC photons / PD photons', 'Median-normalized SC photons / PD photons'], xlabel = 'Wavelength (nm)', colors = ['r', 'k'], legend_strs = [],
                                 save_data_dir = '', save_file_name = None):
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

"""
import FrenchDataSetupExtraScripts as fdata
dir_root = '../data/CBP_throughput_calib_data/ut20210713/'
legend_strs = [['Laser @ Max, 1mm hole', 'Laser @ Max, 5mm hole'], ['Laser @ Max, 1mm hole', 'Laser @ Max, 5mm hole']]
QSWs = ['Max', '300', '299', '296','293','289', '288']
data_files = [[dir_root + 'LinearityScans/QSW' + str(QSW) + '/' + 'QSW' + str(QSW) + '_5mmPinPD_charges.csv' for QSW in QSWs], [dir_root + 'LinearityScans/QSW' + str(QSW) + '/' + 'QSW' + str(QSW) + '_5mmPinSC_charges.csv' for QSW in QSWs], [dir_root + 'LinearityScans/QSW' + str(QSW) + '/' + 'QSW' + str(QSW) + '_5mmPinSC_photons_over_PD_photons.txt' for QSW in QSWs]]
legend_strs = [['QSW' + '-' + QSW for QSW in QSWs], ['QSW' + '-' + QSW for QSW in QSWs], ['QSW' + '-' + QSW for QSW in QSWs]]
fdata.checkLinearityResults(data_files, legend_strs = legend_strs, save_data_dir = dir_root, save_file_name = 'CBP_linearity_varyingQSWs.pdf')
"""
def checkLinearityResults(data_files, figsize = [12, 9], ylabels = ['PD Charge / QE', 'SC charge / QE', 'SC photons / PD photons'], xlabel = 'Wavelength (nm)', colors = ['r', 'blue', 'green', 'orange','magenta','k', 'cyan'], legend_strs = [],
                                 save_data_dir = '', save_file_name = None):
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

    for i in range(len(data_files[2])):
        data = can.readInColumnsToList(data_files[2][i], n_ignore = 1, delimiter = ', ', convert_to_float = 1)
        print ('data = ' + str(data))
        print ('colors = ' + str(colors))
        SC_plots = SC_plots + [axarr[2].plot(data[0], data[1], color = colors[i], marker = 'x')[0]]
        axarr[2].set_xlabel(xlabel)
        axarr[2].set_ylabel(ylabels[2])
        data_by_wavelengths = data_by_wavelengths + [{data[0][i]:data[1][i] for i in range(len(data[0]))}]
        ylims = [1.9, 5.1]
        axarr[2].set_ylim(ylims)
    axarr[2].legend(SC_plots, legend_strs[2])

    if save_file_name != None:
        plt.savefig(save_data_dir + save_file_name)
    plt.show()
