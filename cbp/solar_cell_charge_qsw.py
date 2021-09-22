import numpy as np
import matplotlib.pyplot as plt
import time
import pyvisa as visa
import threading
import xmlrpc
from xmlrpc.client import ServerProxy
import cantrips as can
import sys
import datetime
import os
from calculate_npulses import calculate_npulses

def integration_loop(kei=None, nreads=1):
    kei.data = []
    while(kei._integrate):
        kei.data.append(kei.getread(nreads))

def non_blocking_integration_loop(kei, nreads=1):
    kei.p = threading.Thread(target=integration_loop, kwargs={'kei':kei, 'nreads':nreads})
    kei._integrate=True
    kei.p.start()

def join(kei):
    kei._integrate=False
    kei.p.join()
    data = np.array(kei.data).T
    return np.rec.fromarrays(data[[0,1]], names=['charges', 'time'])

def before_exposure(kei):
    keithley_pwc = 1
    keithley_discharge_time = 2
    kei.set_charge_mode(keithley_pwc)
    kei.zero_check(True)
    time.sleep(keithley_discharge_time)
    kei.zero_check(False)

delta_filtre = -45
phase_filtre = 100
number_filtre = 4
POS_0 = 'EMPTY'
POS_1 = 'red532'
POS_2 = 'blue680'
POS_3 = '1064'
l_POS = ['EMPTY', 'red532', 'blue680', '1064']


class LaserWheel():

     def __init__(self):
         self.band_names = {}
         self.number_filtre = number_filtre
         self.phase_filtre = phase_filtre
         self.delta_filtre = delta_filtre
         for i in range(0,self.number_filtre):
             self.band_names[i] = l_POS[i]
         self.band_pos = dict(list(zip(list(self.band_names.values()),
list(self.band_names.keys()))))
         self.get_filter()
         
     def angle_to_slot(self, angle):
         return round((angle - self.phase_filtre)/self.delta_filtre)

     def slot_to_angle(self, slot):
         return self.phase_filtre + slot * self.delta_filtre

     def get_filter(self):
         self._filter = self.band_names[self.angle_to_slot(laserwheel_proxy.get_position())]
         return self._filter

     def set_filter(self, filtername):
         if self._filter != filtername:
             filterpos = self.slot_to_angle(self.band_pos[filtername])
             laserwheel_proxy.set_position(filterpos)
             self._filter = filtername

laser = ServerProxy('http://127.0.0.1:8014')
laserwheel_proxy = ServerProxy('http://127.0.0.1:8015')
laserwheel = LaserWheel()
keithley = ServerProxy('http://127.0.0.1:8012')
before_exposure(keithley)
draw = True #define if you want to plot for every wavelength

laser.set_power_on()
laserwheel.set_filter('EMPTY')
time.sleep(10) 

laser.set_mode('Burst')
##- Alternative
# laser.set_mode('Continuous')

#- SZF
#laser.set_energy_level('MAX')
##- Alternative:
laser.set_energy_level('Adjust')
#laser.set_qsw(int(value))
npulses = 1000
laser.set_npulses(npulses)

##- Alternative check
#print(laser.get_mode())
#print(laser.get_energy_level())
#print(laser.get_interlock_state())

##- Set the keithley range:
keithley.set_charge_range_upper(2) # 2 or 20 micro C
# keithley.set_charge_range_lower(value) # 20 or 200 nano C

rm = visa.ResourceManager()
A_keysight_id = 'USB0::2391::37912::MY54321261::0::INSTR'
A_key = rm.open_resource(A_keysight_id) 
A_key.write(':SENSe:FUNCtion:ON "CHAR"')

A_key.timeout = 10000 #Keysight timeout in milliseconds - needs to be high if n_pl_cycles is high

#Set current range 
charge_range_str = '2e-6' #charge range in coloumbs - max value is 2e-6 
A_key.write(":SENS:CHAR:RANG:AUTO OFF")
A_key.write(":SENS:CHAR:RANG " + charge_range_str)

#Set measurement speed
n_pl_cycles_str = '100'
pl_freq = 1/50.0 
A_key.write(":SENS:CHAR:NPLC:AUTO OFF")
A_key.write(":SENS:CHAR:NPLC " + n_pl_cycles_str)

#Adjust trigger timing parameters
trigger_delay_time_str = '0'
A_key.write(':TRIG:ACQ:DEL ' + trigger_delay_time_str) 
A_key.write(':TRIG:SOUR TIM')
trigger_time_interval_str = '2e-3'
A_key.write(':TRIG:TIM ' + trigger_time_interval_str)
n_samples_str = '5500'
A_key.write(':TRIG:COUN ' + n_samples_str)

l_wavelength = np.arange(515, 550, 1)
#l_wavelength = [700, 750, 800, 850, 900, 950]
n_bursts = 5
start = time.time()
l_qsw = np.arange(285, 301, 1)
l_qsw = l_qsw.tolist()
l_qsw.reverse()
l_qsw = ['Max'] +  [str(elem) for elem in l_qsw]
l_qsw = ['Max']
today = datetime.date.today() 
date_str = str(today.year) + ('0' if today.month < 10 else '') + str(today.month) + ('0' if today.day < 10 else '') + str(today.day)
exp_dir = 'Filterwheel_2'
extra_end_str = '_5mm'
dir_root = './'
if not(os.path.exists(dir_root + 'ut' + date_str + '/' + exp_dir + '/')):
    if not(os.path.exists(dir_root + 'ut' + date_str + '/')):
        os.mkdir(dir_root + 'ut' + date_str + '/')
    os.mkdir(dir_root + 'ut' + date_str + '/' + exp_dir + '/') 

for filt in ['EMPTY', 'red532', 'blue680', '1064'] :

    laserwheel.set_filter(filt)
    time.sleep(1)
    print('Filter : ', laserwheel.get_filter())
    
    for qsw in l_qsw :
        
        if qsw == 'Max':
            laser.set_energy_level('MAX')
            time.sleep(10)
            print(f'QSW : {laser.get_energy_level()}')
        else:  
            laser.set_qsw(int(qsw))
            time.sleep(10)
            print(f'QSW : {laser.get_qsw()}')
        

        for wl in l_wavelength:
            iteration_start = time.time()
            npulses = calculate_npulses(wl)
            print('npulses = ' + str(npulses))
            print('wavelength = ' + str(wl))
            laser.set_npulses(npulses)
            burst_starts = [-1 for i in range(n_bursts)]
            burst_ends = [-1 for i in range(n_bursts)]
            laser.set_wavelength(f'{int(wl)}')
            before_exposure(keithley)
            A_key.write("SENS:CHAR:DISCharge") 
            #Tell Keysight to start taking data 
            non_blocking_integration_loop(keithley, nreads=1)

            print('starting Keithley acquisition ')
            A_key.write(':INP ON')
            A_key.write(':INIT:ACQ')
            print('starting Keysight acquisition ')
            for i in range(n_bursts) :
                start_burst = time.time()  
                burst_starts[i] = start_burst 
                laser.trigger_burst()
                end_burst = time.time()
                burst_ends[i] = end_burst 
                print ('Burst ' + str(i+1) + ' of ' + str(n_bursts) + ' took ' + str(end_burst - start_burst) + 's.')
                #A_key.write("SENS:CHAR:DISCharge")
                #print ('Discharge?')
                #time.sleep(0.2) 
                
            #print ('Burst starts ='  +str(burst_starts)) 
            results_str = A_key.query(':FETC:ARR:CHAR?')
            #print('end query arr', time.time()-start)
            kei_data = join(keithley)[0]
            #kei_data = kei_data.T
            kei_data= [[elem[1] for elem in kei_data], [elem[0] for elem in kei_data]]

            print(f'Charge in keithley :{kei_data[1][-1]}')
            
            #print ('results_str = ' + str(results_str))
            results_data = results_str.split(',')
            results_data[-1] = results_data[-1][:-1] 
            #print ('results_data = ' + str(results_data)) 
            results_data = [float(elem) for elem in results_data]

            data_point_time_sep = float(trigger_time_interval_str) + float(n_pl_cycles_str) * pl_freq
            delta_ts = [data_point_time_sep * i for i in range(int(n_samples_str))]
            #saveListsToColumns(lists_to_save, save_file, save_dir, sep = ' ', append = False, header = None, type_casts = None)
            can.saveListsToColumns([[can.round_to_n(elem, 5) for elem in delta_ts], [can.round_to_n(elem, 5) for elem in results_data]], dir_root + f'ut' + date_str + '/' + exp_dir + f'/SolarCell_fromB2987A_Wave{int(wl):04d}_QSW{qsw}_Filter{filt}' + extra_end_str + '.csv', '', sep = ', ', header = 'Time sep (ms?), Charge (C)')
            can.saveListsToColumns([kei_data[0], kei_data[1]], dir_root + f'ut' + date_str + '/' + exp_dir + f'/Photodiode_fromKeithley_Wave{int(wl):04d}_QSW{qsw}_Filter{filt}' + extra_end_str + '.csv', '', sep = ', ', header = 'Time sep (s?), Charge (C)')
            can.saveListsToColumns([burst_starts, burst_ends], dir_root + f'ut' + date_str + '/' + exp_dir + f'/LaserBurstTimes_Wave{int(wl):04d}_QSW{qsw}_Filter{filt}' + extra_end_str + '.csv', '', sep = ', ', header = 'Burst starts (s), Burst ends(A)')
            iteration_end = time.time()
            print ('One round through loop took ' + str(iteration_end - iteration_start) + 's')

            if draw == True :
                fig, ax = plt.subplots(2, 1)
                ax[1].plot(delta_ts, results_data)
                ax[1].scatter(delta_ts, results_data, marker = '+')
                ax[1].set_xlabel(r'$\Delta t$ (ms)')
                ax[1].set_ylabel(r'SC Charge (C)')
                ax[0].plot(kei_data[0], kei_data[1], '+') 
                ax[0].set_xlabel(r'PD time (s)')
                ax[0].set_ylabel(r'PD Charge (C)')
                plt.draw()
                plt.pause(0.05)
                plt.close()

        if qsw == 'Max':
            laser.set_energy_level('Adjust')

A_key.write(':INP OFF')
#print ('kei_data = ' + str(kei_data)) 
fig, ax = plt.subplots(2, 1)
ax[1].plot(delta_ts, results_data)
ax[1].scatter(delta_ts, results_data, marker = '+')
ax[1].set_xlabel(r'$\Delta t$ (ms)')
ax[1].set_ylabel(r'SC Charge (C)')
ax[0].plot(kei_data[0], kei_data[1], '+') 
ax[0].set_xlabel(r'PD time (s)')
ax[0].set_ylabel(r'PD Charge (C)')
plt.show()

##- Remember to turn off the laser
laser.set_power_off()

