import numpy as np
import astropy.time
import pkg_resources
from typing import List


def load_potentialdata() -> (np.array, List[str]):
    """Loads the potential data stored in the file data/etcpot.dat and returns it in numpy format. Each row contain
    information for one tide constituent.

    Returns:
        potentialdata: Contents of each column of the output potential data table:
            [0]:  WAVE NUMBER OF TAMURA 1987 TIDAL POTENTIAL DEVELOPMENT.
            [1]:  WAVE NUMBER OF CARTWRIGHT-TAYLER-EDDEN 1973 TIDAL POTENTIAL DEVELOPMENT.
            [2]:  DEGREE OF THE POTENTIAL.
            [3]:  ORDER  OF THE POTENTIAL (= ARGUMENT NO. 1).
            [4]:  ARGUMENT NO. 2.
            [5]:  ARGUMENT NO. 3.
            [6]:  ARGUMENT NO. 4.
            [7]:  ARGUMENT NO. 5.
            [8]:  ARGUMENT NO. 6.
            [9]:  ARGUMENT NO. 7 (ONLY VALID FOR TAMURA 1987 POTENTIAL)
            [10]: ARGUMENT NO. 8 (ONLY VALID FOR TAMURA 1987 POTENTIAL)
            [11]: PHASE NUMBER NP, NP*PI/2 IS BE ADDED TO PHASE.
            [12]: CARTWRIGHT-TAYLER-EDDEN AMPLITUDE FOR EPOCH 1870.
            [13]: CARTWRIGHT-TAYLER-EDDEN AMPLITUDE FOR EPOCH 1960.
            [14]: CARTWRIGHT-TAYLER-EDDEN AMPLITUDE FOR EPOCH 1960.
            [15]: DOODSON AMPLITUDE REFERRING TO EPOCH 1900.
            [16]: TAMURA  AMPLITUDE REFERRING TO EPOCH 2000.
            [17]: TAMURA  TIME DERIVATIVE OF AMPLITUDE PER JULIAN CENTURY.

        tide_names: List of Darwin names for the tides.
    """
    potentialfilename = pkg_resources.resource_filename(__name__, 'data/etcpot.dat')

    with open(potentialfilename) as f:
        for i, line in enumerate(f):
            if line[:4] == "C***":
                nheaderrows = i+2
                break

    columnwidths = [4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 7, 7, 7, 5, 7, 8, 9]
    potentialdata = np.genfromtxt(potentialfilename, skip_header=nheaderrows, delimiter=columnwidths)
    tide_names = np.genfromtxt(potentialfilename, skip_header=nheaderrows, delimiter=columnwidths, usecols=15, dtype=str).tolist()
    tide_names = [s.strip() for s in tide_names]
    potentialdata = np.hstack((potentialdata[:, :15], potentialdata[:, 16:]))
    potentialdata[np.isnan(potentialdata)] = 0
    return potentialdata, tide_names


def get_astronomical_arguments(datestring: str) -> (np.array, np.array):
    """The routine computes the astronomical elements for the given epoch in UTC

    ( G. Klopotek, J.Strandberg)
    Args:
        datestring: -- time in UTC in iso format: 2016-01-16T23:45:00

    Returns
        Tuple containing
            astr_args:
                astr_args[0]:  --  MEAN MOONTIME IN DEGREE.
                astr_args[1]:  --  MEAN LONGITUDE OF THE MOON IN DEGREE.
                astr_args[2]:  --  MEAN LONGITUDE OF THE SUN  IN DEGREE.
                astr_args[3]:  --  MEAN LONGITUDE OF THE PERIGEE OF THE MONN'S ORBIT  IN DEGREE.
                astr_args[4]:  --  NEGATIVE MEAN LONGITUDE OF THE ASCENDING NODE OF THE MOON'S ORBIT IN DEGREE.
                astr_args[5]:  --  MEAN LONGITUDE OF THE PERIGEE OF THE SUN'S ORBIT IN DEGREE.
                astr_args[6]:  --  PERIOD OF JUPITER'S OPPOSITION IN DEGREE (FOR TAMURA 1987 TIDAL POTENTIAL DEVELOPMENT).
                astr_args[7]:  --  PERIOD OF VENUS'S CONJUNCTION IN DEGREE (FOR TAMURA 1987 TIDAL POTENTIAL DEVELOPMENT).

            rates: TIME DERIVATIVES OF THE CORRESPONDING VARIABLES IN astr_args IN DEGREE PER HOUR.
    """

    # 1) Define the J2000 epoch and given epoch in UTC,TT,UT1
    j2000_utc = astropy.time.Time('2000-01-01T12:00:00', format='isot', scale='utc')
    date_utc = astropy.time.Time(datestring, format='isot', scale='utc')

    j2000_tt = j2000_utc.tt
    date_tt = date_utc.tt

    # TODO: Fix the implementation of UT1. Astropy is currently broken...
    # As a hotfix we will use utc.
    j2000_ut1 = j2000_utc  #.ut1
    date_ut1 = date_utc  #.ut1

    # anyDate_TT- j2000_tt difference in 36525 units
    dt_tt = (date_tt.jd - j2000_tt.jd)/36525
    # date_ut1- j2000_ut1 difference
    dt_ut1 = (date_ut1.jd - j2000_ut1.jd)/36525

    # 2) Get t_u and t_d times
    # t_u - universal time measured from 1/01/2000 12:00:00 UT1 in 36525 days unit
    # t_d - dynamical time measured from 1/01/2000 12:00:00 TD  in 36525 days unit
    t_universal = dt_ut1
    t_dynamic = dt_tt

    # 2) Compute astronomical elements from Tamura,1987 formulas
    hoursperjuliancentury = 876600  # hours per julian century

    am = 280.4606184 + 36000.7700536*t_universal + 0.00038793*t_universal**2-0.0000000258*t_universal**3
    alp = (36000.7700536 + 2.0*0.00038793*t_universal - 3.0*0.0000000258*t_universal**2)/hoursperjuliancentury

    s = 218.316656+481267.881342*t_dynamic-0.001330*t_dynamic**2
    sp = (481267.881342 - 2.0 * 0.001330 * t_dynamic) / hoursperjuliancentury

    h = 280.466449+36000.769822*t_dynamic+0.0003036*t_dynamic**2
    hp = (36000.769822+2.0*0.0003036*t_dynamic)/hoursperjuliancentury

    ds = 0.0040*np.cos(np.deg2rad(29+133*t_dynamic))
    dsp = -0.0040*133*np.pi/180*np.sin(np.deg2rad(29+133.0*t_dynamic))/hoursperjuliancentury

    dh = 0.0018*np.cos(np.deg2rad(159+19*t_dynamic))
    dhp = (-0.0018*19*np.pi/180*np.sin(np.deg2rad(159+19*t_dynamic)))/hoursperjuliancentury

    f_4 = 83.353243 + 4069.013711*t_dynamic - 0.010324*t_dynamic**2
    f_5 = 234.955444 + 1934.136185*t_dynamic - 0.002076*t_dynamic**2
    f_6 = 282.937348 + 1.719533*t_dynamic + 0.0004597*t_dynamic**2
    f_7 = 248.1 + 32964.47*t_dynamic
    f_8 = 81.5 + 22518.44*t_dynamic

    astr_args = np.zeros(8)
    rates = np.zeros(8)

    astr_args[0] = am-s   # 15*t_universal + site_eastlongitude ?!?!?
    astr_args[1] = s+ds
    astr_args[2] = h+dh
    astr_args[3] = f_4
    astr_args[4] = f_5
    astr_args[5] = f_6
    astr_args[6] = f_7
    astr_args[7] = f_8

    # 3) Compute speeds in Degrees per hour
    rates[0] = alp-sp+15.0
    rates[1] = sp+dsp
    rates[2] = hp+dhp
    rates[3] = (4069.013711-2*0.010324*t_dynamic)/hoursperjuliancentury
    rates[4] = (1934.136185-2*0.002076*t_dynamic)/hoursperjuliancentury
    rates[5] = (1.719533+2*0.0004597*t_dynamic)/hoursperjuliancentury
    rates[6] = 32964.47/hoursperjuliancentury
    rates[7] = 22518.44/hoursperjuliancentury

    astr_args %= 360.0

    return astr_args, rates


def load_model_data(site: str = "", model: str = "GOT00.2", potentialfilepath: str = "") -> (np.array, List[str]):
    """Loads displacement amplitudes for the major tide constituents.

    The module search for a blq data file corresponding to site and model in the package data directory unless
    potentialfilepath is specified, in which case the file at the specified location is used to load displacement
    amplitudes and phase lags.

    Args:
        site (optional): Name of site for which to retrieve amplitudes, not used if potentialfilepath
            is specified.
        model (optional): Name of model used to calculate amplitudes (default: GOT00.2), not used if potentialfilepath
            is specified.
        potentialfilepath (optional): path to blq-file containg displacement amplitudes and phases for the
            desired site.

    Returns:
        tuple containing:
            displacement_data: numpy array where the rows contains up, west, south component of the displacement
                amplitudes, followed by their respective phase lag.
            tide_names: list of the tide names corresponding to each column of displacement_data
    """
    try:
        if potentialfilepath:
            path = potentialfilepath
        else:
            path = pkg_resources.resource_filename(__name__, 'data/' + site.upper() + '_' + model.upper() + '.blq')
        displacement_data = np.loadtxt(path)
    except FileNotFoundError:
        if potentialfilepath:
            print('There exists no displacement file at %s.' % potentialfilepath)
        else:
            print('There exists no displacement file for site %s and model %s.' % (site, model))
        print('Calculate the data using http://holt.oso.chalmers.se/loading/ and put the results in %s.' %
              pkg_resources.resource_filename(__name__, 'data/'))
        raise

    tide_names = ['M2', 'S2', 'N2', 'K2', 'K1', 'O1', 'P1', 'Q1', 'MF', 'MM', 'SSA']

    return displacement_data, tide_names


def load_brest_data() -> (np.array, np.array):
    """Load brest data

    Returns:
        tuple containing:
            displacement_data: numpy array where the rows contains up, west, south component of the displacement
                amplitudes, followed by their respective phase lag. (Warning: west and south will only be 0s.)
            tide_arguments: numpy array where each row is the Tamura degree of potential and tide arguments for the
                corresponding column in displacement data, i.e. row one is:
                [2, 1, 1, 0, 0, 0, 0, 0]
                if the first column in displacement data correponds exactly to the K1 tide.
                Note that argument number 8 is omitted (assumed 0).
    """
    path = pkg_resources.resource_filename(__name__, 'data/brst_tg_10m-prl.txt')

    displacement_data = None
    tide_arguments = None

    return displacement_data, tide_arguments
