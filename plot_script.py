import numpy as np
import oceanloading, datetime
import matplotlib as mpl
from matplotlib import style
from matplotlib import pyplot as plt
import time
from scipy import signal

style.use(['classic', 'seaborn-whitegrid', 'seaborn-talk'])

mpl.rcParams['figure.figsize'] = [9, 9]
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['xtick.labelsize'] = 11.5
mpl.rcParams['ytick.labelsize'] = 11.5
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['savefig.dpi'] = 200

hours_per_year = 366*24  # 2016 is a leap year

date_list = [datetime.datetime(2016,1,1) + datetime.timedelta(hours=i) for i in range(hours_per_year)]
data = np.genfromtxt('truthdata/ONSALA_REN_GOT00.2.tsf', skip_header=1, skip_footer=1)
site = 'ONSALA'; model = 'GOT00.2'

t0 = time.time()
py_data = oceanloading.calc_displacement(date_list, site, model)
t1 = time.time()
py_data_interp = oceanloading.calc_displacement_interpolated(date_list, site, model)
t2 = time.time()

plt.plot(py_data[:, 0]*1000, label='Only modeled')
#plt.plot(py_data_interp[:, 0]*1000)
plt.plot(py_data[:, 0]*1000 + py_data_interp[:, 0]*1000, label='Interpolated')
plt.plot(data[:len(py_data), 7], label='Truth')
diff = data[:len(py_data), 7] - (py_data[:, 0]*1000 + py_data_interp[:, 0]*1000)
plt.plot(diff, label='Diff (truth - interp)')
plt.xlabel('Hours from 2016-01-01 00:00')
plt.ylabel('Ofset [mm]')
plt.legend()

f, Pxx_den = signal.periodogram(diff, 24)

plt.figure()
plt.semilogy(f, Pxx_den)
plt.xlabel('frequency [cyc/day]')
plt.ylabel('PSD [V**2/day]')

print('Computing modeled tides: %.3f s' % (t1-t0))
print('Computing interp  tides: %.3f s' % (t2-t1))
print('Total time: %.3f s' % (t2-t0))
plt.show()
