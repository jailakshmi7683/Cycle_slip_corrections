import georinex as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

obs = gr.load("iisc0200.24o")
print(obs)

gps_sats = [sv for sv in obs.sv.values if sv.startswith("G")]
print(gps_sats[:10])

sat = gps_sats[0]

L1 = obs["L1"].sel(sv=sat).values
L2 = obs["L2"].sel(sv=sat).values
P1 = obs["P1"].sel(sv=sat).values
P2 = obs["P2"].sel(sv=sat).values

time = obs.time.values

print("Satellite:", sat)
print("Number of epochs:", len(time))

# ----------------------------
# GNSS CONSTANTS
# ----------------------------

c = 299792458

f1 = 1575.42e6
f2 = 1227.60e6

lambda1 = c / f1
lambda2 = c / f2
lambda_wl = c / (f1 - f2)

# ----------------------------
# REMOVE NaN VALUES
# ----------------------------

valid = (
    ~np.isnan(L1) &
    ~np.isnan(L2) &
    ~np.isnan(P1) &
    ~np.isnan(P2)
)

L1 = L1[valid]
L2 = L2[valid]
P1 = P1[valid]
P2 = P2[valid]
time_valid = time[valid]

print("Valid epochs:", len(time_valid))

# ----------------------------
# MELBOURNE-WUBBENA
# ----------------------------

MW = (L1*lambda1 - L2*lambda2 - (P1 - P2)) / lambda_wl

print("MW sample values:")
print(MW[:10])

# ----------------------------
# CYCLE SLIP DETECTION
# ----------------------------

window = 30

slips = []

for i in range(window, len(MW)):

    mean_MW = np.mean(MW[i-window:i])
    std_MW = np.std(MW[i-window:i])

    if abs(MW[i] - mean_MW) > 4 * std_MW:
        slips.append(i)

print("Number of cycle slips detected:", len(slips))

plt.figure(figsize=(10,4))

plt.plot(time_valid, MW, label="MW")

plt.scatter(time_valid[slips], MW[slips],
            color='red', label="Detected Slip")

plt.title(f"Cycle Slip Detection - {sat}")
plt.xlabel("Time")
plt.ylabel("MW cycles")

plt.legend()
plt.grid()

plt.show()