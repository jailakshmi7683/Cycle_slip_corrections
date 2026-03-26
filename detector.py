import georinex as gr
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ----------------------------
# CONSTANTS
# ----------------------------
c = 299792458
f1 = 1575.42e6
f2 = 1227.60e6

lambda1 = c / f1
lambda2 = c / f2
lambda_wl = c / (f1 - f2)

# ----------------------------
# PARAMETERS
# ----------------------------
k = 3.5
N = 20
min_std = 0.3

# ----------------------------
# LOAD DATA
# ----------------------------
obs = gr.load("iisc0200.24o")

gps_sats = [sv for sv in obs.sv.values if sv.startswith("G")]
sat = gps_sats[0]

# ----------------------------
# OBSERVABLE SELECTION
# ----------------------------
def get_obs(obs, names, sat):
    for name in names:
        if name in obs:
            return obs[name].sel(sv=sat).values
    raise ValueError(f"Missing {names}")

L1 = get_obs(obs, ["L1"], sat)
L2 = get_obs(obs, ["L2"], sat)

P1 = get_obs(obs, ["P1", "C1"], sat)
P2 = get_obs(obs, ["P2", "C2"], sat)

time = obs.time.values

# ----------------------------
# REMOVE NaNs
# ----------------------------
mask = (~np.isnan(L1)) & (~np.isnan(L2)) & (~np.isnan(P1)) & (~np.isnan(P2))
L1, L2, P1, P2 = L1[mask], L2[mask], P1[mask], P2[mask]
time = time[mask]

# ----------------------------
# MELBOURNE-WUBBENA (CYCLES)
# ----------------------------
MW = (L1*lambda1 - L2*lambda2 - (P1 - P2)) / lambda_wl

# ----------------------------
# REMOVE DRIFT (CRITICAL STEP)
# ----------------------------
window_drift = 100

MW_smooth = np.convolve(MW, np.ones(window_drift)/window_drift, mode='same')
MW_hp = MW - MW_smooth   # high-pass filtered MW

# ----------------------------
# DETECTION
# ----------------------------
slips = np.zeros(len(MW_hp))

k = 3.5              # slightly relaxed but stable
abs_threshold = 1.5  # IMPORTANT: increase to avoid noise

for i in range(N, len(MW_hp)):

    window_data = MW_hp[i-N:i]

    mean = np.mean(window_data)
    std = np.std(window_data)

    diff = abs(MW_hp[i] - mean)

    cond_stat = diff > k * std
    cond_abs  = diff > abs_threshold
    cond_std  = std >= min_std

    # KEY: BOTH conditions required (not OR)
    if cond_stat and cond_abs and cond_std:

        slips[i] = 1

        t = time[i].astype('datetime64[s]').astype(datetime)

        year = t.year
        doy = t.timetuple().tm_yday
        sod = t.hour*3600 + t.minute*60 + t.second

        time_str = t.strftime("%H:%M:%S.00")

        print(f"CS {year} {doy:03d} {sod:.2f} {time_str} GPS {sat[1:]} "
              f"C1P-C2P-L1P-L2P MW = {MW[i]:.4f} THRESHOLD = {(k*std):.4f}")

# ----------------------------
# CORRECTION
# ----------------------------
MW_corr = MW_hp.copy()
bias = 0

for i in range(1, len(MW_hp)):
    if slips[i] == 1:
        jump = MW_hp[i] - MW_hp[i-1]
        bias += jump
    MW_corr[i] -= bias

# ----------------------------
# PLOT
# ----------------------------
plt.figure(figsize=(14,6))

plt.plot(time, MW, label="Raw MW", alpha=0.5)
plt.plot(time, MW_hp, label="Filtered MW (Used)", linewidth=1)
plt.plot(time, MW_corr, label="Corrected MW", linewidth=1)

slip_idx = np.where(slips == 1)[0]

plt.scatter(time[slip_idx], MW_hp[slip_idx],
            color='red', s=70, label='Cycle Slip')

plt.xlabel("Time")
plt.ylabel("MW (cycles)")
plt.title(f"MW Cycle Slip Detection (DEBUG MODE) - {sat}")
plt.legend()
plt.grid()

plt.show()