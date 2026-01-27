import torch 
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


# path_dataset = "/home/jaramy/thesis_ws/IsaacLabExtensionTemplate/dataset_numpy/trial_1_15-1-26/episode_data_20.npy"
path_dataset = "/home/jaramy/thesis_ws/IsaacLabExtensionTemplate/dataset_numpy/test_trial_/episode_data_00000.npy"
# Load the .npy file
data = np.load(path_dataset)
print(data.shape)

vx = data[:, 4]
vy = data[:, 5]
vz = data[:, 6]

sigma = 8 # Standard deviation for the Gaussian kernel
smoothed_vx = gaussian_filter1d(vx, sigma=sigma)
smoothed_vy = gaussian_filter1d(vy, sigma=sigma)
# plt.figure()
# plt.plot(vx)
# plt.xlabel("Timestep")
# plt.ylabel("Vx")
# plt.title("Velocity X over Time")
# plt.show()

# plt.figure()
# plt.plot(vy)
# plt.xlabel("Timestep")
# plt.ylabel("Vy")
# plt.title("Velocity Y over Time")
# plt.show()

# plt.figure()
# plt.plot(vz)
# plt.xlabel("Timestep")
# plt.ylabel("Vz")
# plt.title("Velocity Z over Time")
# plt.show()

sigma = 8 
gas_left = data[:, 0]
gas_right = data[:, 1]
smoothed_gas_left = gaussian_filter1d(gas_left, sigma=sigma)
smoothed_gas_right = gaussian_filter1d(gas_right, sigma=sigma)

# plt.figure()
# plt.plot(gas_left)
# plt.xlabel("Timestep")
# plt.ylabel("Gas Left")
# plt.title("Odor (Left Sensor) over Time")
# plt.show()

# plt.figure()
# plt.plot(gas_right)
# plt.xlabel("Timestep")
# plt.ylabel("Gas Right")
# plt.title("Odor (Right Sensor) over Time")
# plt.show()

wind_dir = data[:, 2]
wind_spd = data[:, 3]

# plt.figure()
# plt.plot(wind_dir)
# plt.xlabel("Timestep")
# plt.ylabel("Wind Direction (rad or deg)")
# plt.title("Wind Direction over Time")
# plt.show()

# plt.figure()
# plt.plot(wind_spd)
# plt.xlabel("Timestep")
# plt.ylabel("Wind Speed")
# plt.title("Wind Speed over Time")
# plt.show()

# plt.figure()
# plt.plot(wind_spd, gas_left)
# plt.xlabel("Wind Speed")
# plt.ylabel("Gas Left")
# plt.title("Gas Left vs Wind Speed")
# plt.show()

# plt.figure()
# plt.plot(wind_spd, gas_right)
# plt.xlabel("Wind Speed")
# plt.ylabel("Gas Right")
# plt.title("Gas Right vs Wind Speed")
# plt.show()


norm_vxy = np.sqrt(smoothed_vx**2 + smoothed_vy**2)

plt.figure()
plt.subplot(5, 1, 1)
plt.plot(smoothed_gas_left)
plt.ylabel("Gas Left")

plt.subplot(5, 1, 2)
plt.plot(smoothed_gas_right)
plt.xlabel("Timestep")
plt.ylabel("Gas Right")

plt.subplot(5, 1, 3)
plt.plot(smoothed_vx)
plt.xlabel("Timestep")
plt.ylabel("vx")

plt.subplot(5, 1, 4)
plt.plot(smoothed_vy)
plt.xlabel("Timestep")
plt.ylabel("vy")

plt.subplot(5, 1, 5)
plt.plot(norm_vxy)
plt.xlabel("Timestep")
plt.ylabel("norm vel")


plt.show()