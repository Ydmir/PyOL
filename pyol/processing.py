from .dataload import get_astronomical_arguments, load_potentialdata, load_model_data
import astropy
import astropy.constants as aconst
import datetime
import numpy as np
from typing import List

import sys

this = sys.modules[__name__]

this.potential_data = None
this.tide_names = None


def calc_tamura(calc_datetime: datetime.datetime, full_out=False) -> (np.array, np.array, np.array, list):
    """Calculates the tamura phases and amplitudes.

    Based on the astros.f file from H.G Schernecks fortran ocean loading implementation as well as the 1987 paper
    from Tamura.

    Args:
        calc_datetime: Time and date for calculation
        full_out (optional): if set to true, the degree and order of the tides are output

    Returns:
        (tuple):
            phases,
            frequencies, in degrees/h
            amplitudes,
            tide_names,
            n (optional),
            m (optional)

    References:
        Tamura, Yoshiaki. "A harmonic development of the tide-generating potential."
        Bull. Inf. Marées Terrestres 99 (1987): 6813-6855.
    """
    # time in julian centurieas from j2000
    t_j2000 = (calc_datetime - datetime.datetime(2000, 1, 1, 12, 0, 0)).total_seconds() / (86400 * 36525)

    if not np.any(this.potential_data):
        this.potential_data, this.tide_names = load_potentialdata()
    astr_args, rates = get_astronomical_arguments(calc_datetime.date().isoformat())

    geocentric_gravitational_constant = 398600.4418 * 1000**3
    rmmoon = 0.01230002
    earth_radius_equator = aconst.R_earth.si.value
    earth_radius_mean = 6.371000e6
    prlx = 3422.448  # in arc seconds

    wnm = np.zeros((6, 6))
    # Degree n<2 is not used, so these are left 0.
    wnm[2, 0] = -1.0
    wnm[2, 1] = 1.5
    wnm[2, 2] = 3.0
    wnm[3, 0] = -1.0/np.sqrt(5.0)
    wnm[3, 1] = -8.0/np.sqrt(15.0)
    wnm[3, 2] = 10.0/np.sqrt(3.0)
    wnm[3, 3] = 15.0
    wnm[4, 0] = 1.0
    wnm[4, 1] = -5.0*(3.0+np.sqrt(393.0))*np.sqrt(390.0+2*np.sqrt(393.0))/896.0
    wnm[4, 2] = -135.0/14.0
    wnm[4, 3] = 315.0*np.sqrt(3.0)/16.0
    wnm[4, 4] = 105.0
    wnm[5, 0] = 1.0
    wnm[5, 1] = -2.0588572  # X_max = sqrt((3-2*sqrt(11/21))/5)
    wnm[5, 2] = -10.47079983  # X_max = sqrt((2-sqrt(7/3))/5)
    wnm[5, 3] = 140.0/3.0/np.sqrt(1.5)
    wnm[5, 4] = 3024.0/5.0/np.sqrt(5.0)
    wnm[5, 5] = 945.0

    doodsn = (0.75 * geocentric_gravitational_constant * rmmoon / earth_radius_equator *
              (earth_radius_mean/earth_radius_equator)**2 * (np.deg2rad(prlx/3600))**3
              )

    argument = this.potential_data[:, 3:11]
    n = this.potential_data[:, 2].astype(int)
    m = this.potential_data[:, 3].astype(int)
    k = this.potential_data[:, 11].astype(int)

    phases = (np.sum(astr_args*argument, axis=1)+k*90) % 360        # Degrees
    frequencies = np.sum(rates*argument, axis=1)                    # In degrees/h

    tamura_amp_j2000 = this.potential_data[:, 16]
    tamura_amp_derivative = this.potential_data[:, 17]

    tamura_amp_present = tamura_amp_j2000 + tamura_amp_derivative*t_j2000
    rawamp = tamura_amp_present * (earth_radius_equator/earth_radius_mean)**n
    amplitudes = doodsn * rawamp / wnm[n, m]

    if full_out:
        return phases, frequencies, amplitudes, this.tide_names, n, m

    return phases, frequencies, amplitudes, this.tide_names


def calc_displacement(calc_datetimes: List[datetime.datetime], site: str, model: str) -> np.array:
    """Calculates displacement using only the frequencies with a corresponding ocean loading model parameter value.

    Args:
        calc_datetimes: List of date and times at which to calculate the displacement in the datetime format.
        site: Name used for the site. Must correspond to the name used in the file name of the ocean loading file.
        model: Model used to generate the time series. Must correspond to an existing ocean loading file.
    Returns:
        Numpy array with up, north and east component of the ocean loading displacement.
    """
    if isinstance(calc_datetimes, datetime.datetime):
        # if only a single datetime is input
        calc_datetimes = [calc_datetimes]

    model_data, tide_names_model = load_model_data(site, model)

    curr_date = None
    displacement = np.zeros((len(calc_datetimes), 3))

    for i, calc_datetime in enumerate(calc_datetimes):
        if calc_datetime.date() != curr_date:
            phases, frequencies, amp, tide_names_tamura = calc_tamura(calc_datetime)
            curr_date = calc_datetime.date()

        displacement_amplitude = np.zeros((3, len(tide_names_model)))
        displacement_phase = np.zeros((3, len(tide_names_model)))

        hours_since_midnight = calc_datetime.hour + calc_datetime.minute/60 + calc_datetime.second/3600

        for idx_model_data, tide in enumerate(tide_names_model):
            idx_tamura = tide_names_tamura.index(tide)

            displacement_amplitude[:, idx_model_data] = model_data[:3, idx_model_data]
            displacement_phase[:, idx_model_data] = (phases[idx_tamura] - model_data[3:, idx_model_data] +
                                                    hours_since_midnight * frequencies[idx_tamura])

        displacement[i, :] = np.sum(displacement_amplitude*np.cos(np.deg2rad(displacement_phase)), axis=1)

    return displacement


def calc_displacement_interpolated(calc_datetimes: List[datetime.datetime], site: str, model: str) -> np.array:
    """Calculates displacement with interpolation of amplitudes to unmodeled frequencies.

    Calculates displacement using all the frequencies in the tamura potential that lies within the range of the
    frequencies which have defined ocean loading model parameter values.
    (J.Strandberg)

    Args:
        calc_datetimes: List of date and times at which to calculate the displacement in the datetime format.
        site: Name used for the site. Must correspond to the name used in the file name of the ocean loading file.
        model: Model used to generate the time series. Must correspond to an existing ocean loading file.

    Returns:
        Numpy array with up, north and east component of the ocean loading displacement.
    """
    if isinstance(calc_datetimes, datetime.datetime):
        # if only a single datetime is input
        calc_datetimes = [calc_datetimes]

    model_data, tide_names_model = load_model_data(site, model)

    curr_date = None
    interpolated_displacement = np.zeros((len(calc_datetimes), 3))

    for i, calc_datetime in enumerate(calc_datetimes):
        if calc_datetime.date() != curr_date:
            curr_date = calc_datetime.date()

            phases, frequencies, amp, tide_names_tamura, n, m = calc_tamura(calc_datetime, full_out=True)

            # Load parameters and interpolate. Check oltides.py

            # Z=S*OTAMP(II,JCOMP)*EXP(DCMPLX(0.0D0,-OTPHA(II,JCOMP)*RAD)) /AU/GU
            # FMT(L,JCOMP)=FRE(K)

            modeled_components = np.array([tide_names_tamura.index(tide) for tide in tide_names_model])


            modeled_phases = phases[modeled_components]
            modeled_frequencies = frequencies[modeled_components]
            modeled_amplitudes = amp[modeled_components]

            modeled_admittance = (
                model_data[:3, :] * np.exp(-np.deg2rad(model_data[3:, :])*1j) /
                np.repeat((modeled_amplitudes * __tilt(modeled_frequencies, m[modeled_components]))[np.newaxis, :], 3, axis=0)
            )
            # This array is in the order that the tides appear in the model files, which is not in strictly increasing freq-order
            # Therefor we sort it:
            sort_order = np.argsort(modeled_frequencies)

            modeled_components = modeled_components[sort_order]
            # modeled_phases = modeled_phases[sort_order]
            modeled_frequencies = modeled_frequencies[sort_order]
            # equivalent: modeled_frequencies =  frequencies[modeled_components]
            # modeled_amplitudes = modeled_amplitudes[sort_order]
            modeled_admittance = modeled_admittance[:, sort_order]
            modeled_m = m[modeled_components]

            bins = np.digitize(frequencies, modeled_frequencies)  # the interval/band in which an unmodeled component lies

            #  # Ignore components whose frequencies are outside those of the modeled components:
            #  mask = (bins >= len(modeled_components)) | (bins == 0)

            # If the value is outside the modeled frequencies, set it to the closest value for now. This will create
            # extrapolation. Will be overwritten when bands are considered anyway:
            bins[bins == 0] = 1
            bins[bins >= len(modeled_components)] = len(modeled_components)-1

            # Remove all that does not have degree n=2
            mask = (n > 2)
            # Remove already modeled components:
            mask[modeled_components] = True
            # Ignore anything not in a band where we have modeled components
            mask |= np.logical_not(np.in1d(m, np.unique(modeled_m)))
            # Removing the constant period (unless it is explicitly modeled!)
            mask |= frequencies == 0
            # TODO: mask out small tides and jupiter/mars?

            bins = bins[~mask]
            phases = phases[~mask]
            frequencies = frequencies[~mask]
            amp = amp[~mask]
            n = n[~mask]
            m = m[~mask]

            interpolated_admittance = (
                modeled_admittance[:, bins-1] + (modeled_admittance[:, bins]-modeled_admittance[:, bins-1]) *
                (frequencies - modeled_frequencies[bins-1]) / (modeled_frequencies[bins] - modeled_frequencies[bins-1])
            )

            # Limits for interpolation. Anything within a band but outside of the modeled frequencies in the same band
            # will be set to the value of the nearest modeled frequency, overwriting the previous wrongly interpolated
            # value.
            for m_band in np.unique(modeled_m):
                inband = (m == m_band)

                index_min_freq_band = np.argmin(modeled_frequencies[modeled_m == m_band])
                index_max_freq_band = np.argmax(modeled_frequencies[modeled_m == m_band])

                min_modeled_freq_band = modeled_frequencies[modeled_m == m_band][index_min_freq_band]
                max_modeled_freq_band = modeled_frequencies[modeled_m == m_band][index_max_freq_band]
                # Corresponding admittances:
                min_modeled_admitance_band = modeled_admittance[:, modeled_m == m_band][:, index_min_freq_band]
                max_modeled_admitance_band = modeled_admittance[:, modeled_m == m_band][:, index_max_freq_band]

                interpolated_admittance[:, inband][:, frequencies[inband] < min_modeled_freq_band] = min_modeled_admitance_band[:, np.newaxis]
                interpolated_admittance[:, inband][:, frequencies[inband] > max_modeled_freq_band] = max_modeled_admitance_band[:, np.newaxis]

            interpolated_displacement_at_midnight = interpolated_admittance * np.exp(np.deg2rad(phases) * 1j) * amp * __tilt(frequencies, m)

        hours_since_midnight = calc_datetime.hour + calc_datetime.minute/60 + calc_datetime.second/3600

        interpolated_displacement[i, :] = np.sum(
            np.real(interpolated_displacement_at_midnight * np.exp(np.deg2rad(hours_since_midnight * frequencies)*1j)),
            axis=1
        )

    return interpolated_displacement


def __tilt(f: np.array, m: np.array) -> np.array:
    """
    Help-function to calculate gamma factor. Important especially for frequencies near the Near Diurnal Free Wobble.
    Assumes that the degree of the potential is n=2.
    (J.Strandberg)

    args:
        f: frequency in degree per hour
        m: order of the corresponding potential

    returns:
        gamma: the gamma factor
    """
    k2 = 0.298
    h2 = 0.6030
    k2_strength = 4.13e-3
    h2_strength = 4.08e-3

    f_rad = np.deg2rad(f)

    f0 = 0.243351886
    erot = 0.26251614
    ndr_beat_period = 432
    f1 = (1 + 1/ndr_beat_period)*erot

    gamma = 1 + k2-h2+((k2_strength*k2-h2_strength*h2)*(f_rad-f0)/(f_rad-f1)) * (m == 1)

    return gamma
