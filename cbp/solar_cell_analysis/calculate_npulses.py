import numpy as np
import scipy as sp
from scipy import interpolate

def calculate_npulses(wl) :
    
    '''Calculate the number of pulses to maximize and flatten the signal for each wavelength in nanometer
    '''
    
    #Load a previous file to find the turnout point between the two regimes of the laser
    
    charges_path = './SC_photocurrents_over_PD_charges.csv'
    data_charges = np.loadtxt(charges_path, skiprows=1, delimiter=",")
    l_wl, SC_charges, PD_charges = data_charges.T
    
    #Interpolate to fill with potentials empty values
    l_wl_full = np.arange(l_wl[0], l_wl[-1]+1)
    f_PD_charges = sp.interpolate.interp1d(l_wl, PD_charges)
    PD_charges = f_PD_charges(l_wl_full)
    
    f_SC_charges = sp.interpolate.interp1d(l_wl, SC_charges)
    SC_charges = f_SC_charges(l_wl_full)
    
    PD_charges_per_pulse = PD_charges/1000

    diff = []
    for i in range(len(PD_charges)) :
        diff.append(PD_charges[i] - PD_charges[i-1])

    turnout = np.array(np.where(diff == np.max(diff))).squeeze()
    #Calculate the mean before the turnout point to flatten the signal with this mean
    charges_mean = np.mean(PD_charges[:turnout])
    
    #Calculate the number of pulses for the wavelength wanted
    npulses = charges_mean/PD_charges_per_pulse[l_wl_full== wl]
    
    
    if npulses <= 1:
        npulses = 1
    elif npulses >= 1000:
        npulses = 1000
        
    return int(npulses)