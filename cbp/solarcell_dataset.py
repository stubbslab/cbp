import numpy as np
import matplotlib.pyplot as plt
import os
import scipy
from tqdm import tqdm
from scipy.optimize import curve_fit
import scipy as sp
from scipy import interpolate


def calculate_npulses(wl):
    """
    Calculate the number of pulses to maximize and flatten the signal for each wavelength in nanometer
    """

    # Load a previous file to find the turnout point between the two regimes of the laser

    charges_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'SC_photocurrents_over_PD_charges.csv')
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
    while endpoint < len(x) - 1 and counter <= len(x):
        chi2 = 0
        # start a line with at least three points
        for index in range(startpoint + min_len, len(x)):
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


def get_total_charge_and_leakage_currents(time, charge, delta_chi2=10, err=None, plot=False, min_len=2, deg=1):
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
        ax[0].set_ylim(1.1 * np.min(charge), 1.1 * np.max(charge))
        for ind in indices:
            ax[0].axvline(time[ind[-1]], color="k", linestyle="-.")
            ax[0].axvline(time[ind[0]], color="k", linestyle="-.")
        ax[0].set_title(f"Total charge: {total_charge:.4g} +/- {total_charge_err:.4g} [C]")
        ax[0].axhline(q1, color="k", linestyle="--")
        ax[0].axhline(q2, color="k", linestyle="--")
        ax[0].axvline(t1, color="k", linestyle="--")
        ax[0].axvline(t2, color="k", linestyle="--")
        ax[0].set_xlabel("Time [ms]")
        ax[0].set_ylabel(f"Charge [C]")
        ax[0].grid()
        ax[1].plot(time, res, "r+")
        ax[1].set_xlabel("Time [ms]")
        ax[1].set_ylabel(f"Difference [C]")
        ax[1].grid()
        fig.tight_layout()
        plt.show()
    return total_charge, total_charge_err, i_k1, i_k2, pvals, indices


def get_info_from_filename(filename):
    words = filename.split("_")
    wavelength = -1
    qsw = "undef"
    for w in words:
        if "Wave" in w:
            wavelength = float(w[4:])
        if "QSW" in w:
            qsw = w[3:]
    npulses = calculate_npulses(wavelength)
    return wavelength, qsw, npulses


class SolarCellPDDataSet:

    def __init__(self, hdu=None, filename="", npulses=-1, wavelength=-1., nbursts=1, qsw="undef"):
        self.filename = filename
        self.npulses = npulses
        self.wavelength = wavelength
        self.nbursts = nbursts
        self.keithley_err = 5e-12
        self.units = "C"
        self.qsw = qsw
        if filename != "":
            self.data = self.load(filename)

    def __str__(self):
        txt = f"{self.nbursts} bursts containing {self.npulses} laser pulses at wavelength {self.wavelength} nm."
        return txt

    def load(self, filename):
        if ".csv" in filename:
            a = np.loadtxt(filename, skiprows=1, delimiter=",")
            self.data = {"time": a.T[0], "charge": a.T[1]}
        else:
            raise ValueError(f"Unknown file extension for {filename}. Must be .csv.")
        return self.data

    def plot_data_set(self, ax):
        ax.plot(self.data["time"], self.data["charge"], "r+")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"Charge [{self.units}]")
        ax.grid()


class SolarCellSCDataSet:

    def __init__(self, hdu=None, filename="", npulses=-1, wavelength=-1., nbursts=1, qsw="undef"):
        self.npulses = npulses
        self.wavelength = wavelength
        self.nbursts = nbursts
        self.filename = filename
        self.units = "C"
        self.qsw = qsw
        if filename != "":
            self.data = self.load(filename)

    def load(self, filename):
        if ".csv" in filename:
            a = np.loadtxt(filename, skiprows=1, delimiter=",")
            self.data = {"time": a.T[0], "charge": a.T[1]}
        else:
            raise ValueError(f"Unknown file extension for {filename}. Must be .csv.")
        return self.data

    def plot_data_set(self, ax):
        ax.plot(1e-3 * self.data["time"], self.data["charge"])
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"Charge [{self.units}]")
        ax.grid()


class SolarCellDataSet:

    def __init__(self, pd_filename="", sc_filename="", npulses=-1, wavelength=-1., nbursts=1, qsw="undef"):
        self.pd = SolarCellPDDataSet(filename=pd_filename, npulses=npulses, wavelength=wavelength, nbursts=nbursts, qsw=qsw)
        self.sc = SolarCellSCDataSet(filename=sc_filename, npulses=npulses, wavelength=wavelength, nbursts=nbursts, qsw=qsw)
        self.npulses = npulses
        self.wavelength = wavelength
        self.nbursts = nbursts
        self.qsw = qsw
        self.filename = pd_filename

    def plot_data_set(self):
        fig, ax = plt.subplots(2, 1)
        self.pd.plot_data_set(ax[0])
        self.sc.plot_data_set(ax[1])


class SolarCellRun:

    def __init__(self, directory_path, tag="", nbursts=5):
        self.directory_path = directory_path
        self.nbursts = nbursts
        self.data_sets = []
        self.filenames = os.listdir(directory_path)
        self.filenames.sort()
        if tag != "":
            self.filenames = [f for f in self.filenames if tag in f]
        self.filenames = [f for f in self.filenames if ("Photodiode_fromKeithley" in f and ".csv" in f)]
        self.names = ["qsw", "set_nbursts", "set_npulses", "set_wl", "pd_charge_total", "pd_charge_total_err",
                      "sc_charge_total", "sc_charge_total_err",
                      "pd_ik1", "pd_ik2", "sc_ik1", "sc_ik2"]

    def load(self):
        for filename in tqdm(self.filenames):
            if "~" in filename:
                continue
            tmp = filename.split(".")[0]
            wavelength, qsw, npulses = get_info_from_filename(tmp)
            d = SolarCellDataSet(pd_filename=os.path.join(self.directory_path, filename),
                           sc_filename=os.path.join(self.directory_path, filename.replace("Photodiode_fromKeithley_",
                                                                                          "SolarCell_fromB2987A_")),
                           npulses=npulses, wavelength=wavelength, nbursts=self.nbursts, qsw=qsw)
            self.data_sets.append(d)

        self.laser_set_wavelengths = np.unique([d.wavelength for d in self.data_sets])
        self.laser_set_npulses = np.unique([d.npulses for d in self.data_sets])
        self.laser_set_nbursts = np.unique([d.nbursts for d in self.data_sets])
        self.ndata = len(self.data_sets)
        formats = [str] + [int, int] + [float] * int(len(self.names) - 2)
        self.data = np.recarray((self.ndata,), names=self.names, formats=formats)

    def __str__(self):
        txt = f"SolarCell run with {len(self.data_sets)} data sets.\n"
        txt += f"Laser set wavelengths: {self.laser_set_wavelengths}\n"
        txt += f"Laser set nbursts: {self.laser_set_nbursts}\n"
        return txt

    def get_data_set(self, laser_set_wavelength, nbursts):
        try:
            for d in self.data_sets:
                if d.wavelength == laser_set_wavelength and nbursts == d.nbursts:
                    return d
        except:
            raise FileNotFoundError(f"SolarCellDataSet with laser_set_wavelength={laser_set_wavelength} and nbursts={nbursts}"
                                    f" not in {self.directory_path}.")

    def solarcell_characterization(self):
        for id, d in enumerate(tqdm(self.data_sets)):
            for name in self.names:
                self.data[name][id] = 0
            self.data["set_nbursts"][id] = d.nbursts
            self.data["set_npulses"][id] = d.npulses
            self.data["set_wl"][id] = d.wavelength
            self.data["qsw"][id] = d.qsw

            noise = estimate_noise(d.sc.data["time"], d.sc.data["charge"])
            err = 10 * noise * np.ones_like(d.sc.data["time"])
            charge_sc, charge_sc_err, i_k1, i_k2, pvals_sc, indices_sc = get_total_charge_and_leakage_currents(
                d.sc.data["time"],
                d.sc.data["charge"], delta_chi2=20,
                err=err, plot=False, min_len=5, deg=1)
            self.data["sc_charge_total"][id] = charge_sc
            self.data["sc_charge_total_err"][id] = charge_sc_err
            self.data["sc_ik1"][id] = i_k1
            self.data["sc_ik2"][id] = i_k2
            #try:
            charge_pd, charge_pd_err, i_k1, i_k2, pvals_pd, indices_pd = get_total_charge_and_leakage_currents(
                d.pd.data["time"],
                d.pd.data["charge"], delta_chi2=10,
                err=1.1e-11 * np.ones_like(d.pd.data["time"]), plot=False)
            self.data["pd_charge_total"][id] = charge_pd
            self.data["pd_charge_total_err"][id] = charge_pd_err
            self.data["pd_ik1"][id] = i_k1
            self.data["pd_ik2"][id] = i_k2
            #except:
            #    print("Skip ?", filename)
            #    pass

    def save(self, filename):
        np.save(filename, self.data)

    def load_from_file(self, filename):
        self.data = np.load(filename)
        self.names = ["set_nbursts", "set_npulses", "set_wl", "qsw", "pd_charge_total", "pd_charge_total_err",
                      "sc_charge_total", "sc_charge_total_err",
                      "pd_ik1", "pd_ik2", "sc_ik1", "sc_ik2"]
        self.laser_set_wavelengths = np.unique(self.data["set_wl"])
        self.laser_set_npulses = np.unique(self.data["set_npulses"])
        self.laser_set_nbursts = np.unique(self.data["set_nbursts"])
        self.ndata = len(self.data)

    def plot_summary(self):
        fig, ax = plt.subplots(3, 1, sharex="all")
        ax[0].errorbar(self.data["set_wl"], self.data["pd_charge_total"], yerr=self.data["pd_charge_total_err"], linestyle="none",
                       label="total charge PD", markersize=3, marker="o")
        ax[1].errorbar(self.data["set_wl"], self.data["sc_charge_total"], yerr=self.data["sc_charge_total_err"], linestyle="none",
                       label="total charge SC", markersize=3, marker="o")
        ratio = self.data["sc_charge_total"] / self.data["pd_charge_total"]
        ratio_err = ratio * np.sqrt((self.data["sc_charge_total_err"] / self.data["sc_charge_total"]) ** 2
                                                    + (self.data["pd_charge_total_err"] / self.data["pd_charge_total"]) ** 2)
        ax[2].errorbar(self.data["set_wl"], self.data["sc_charge_total"] / self.data["pd_charge_total"], yerr=ratio_err, linestyle="none",
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
