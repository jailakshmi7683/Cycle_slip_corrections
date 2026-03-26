import georinex as gr
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------
# LOAD DATA
# ----------------------------
obs = gr.load("iisc0200.24o")

gps_sats = [sv for sv in obs.sv.values if sv.startswith("G")]
sat = gps_sats[4]

L1 = obs["L1"].sel(sv=sat).values
L2 = obs["L2"].sel(sv=sat).values
P1 = obs["P1"].sel(sv=sat).values
P2 = obs["P2"].sel(sv=sat).values
time = obs.time.values

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
# CLEAN DATA
# ----------------------------
valid = np.isfinite(L1) & np.isfinite(L2) & np.isfinite(P1) & np.isfinite(P2)

L1, L2, P1, P2 = L1[valid], L2[valid], P1[valid], P2[valid]
time_valid = time[valid]

# ----------------------------
# COMPUTATIONS
# ----------------------------

# MW (meters)
MW = (L1*lambda1 - L2*lambda2 - (P1 - P2))

# GF (meters)
GF = (L1*lambda1 - L2*lambda2)

# time
time_sec = time_valid.astype("datetime64[s]").astype(int)

# ----------------------------
# PARAMETERS
# ----------------------------
gap_threshold = 40
window_size = 10   # 300 sec / 30 sec

k = 5
min_sigma = 0.8
GF_threshold = 0.05

# ----------------------------
# DETECTION
# ----------------------------
gap_epochs = []
cycle_slip_epochs = []

for i in range(1, len(MW)):

    dt = time_sec[i] - time_sec[i-1]

    # ----------------------------
    # GAP
    # ----------------------------
    if dt > gap_threshold:
        gap_epochs.append(i)
        continue

    # ----------------------------
    # MW DETECTOR
    # ----------------------------
    mw_flag = False

    if i >= window_size:
        window = MW[i-window_size:i]

        if not np.any(np.isnan(window)):
            mean = np.mean(window)
            sigma = max(np.std(window), min_sigma)

            if abs(MW[i] - mean) > k * sigma:
                mw_flag = True

    # ----------------------------
    # GF DETECTOR
    # ----------------------------
    gf_flag = False

    dGF = GF[i] - GF[i-1]

    if abs(dGF) > GF_threshold:
        gf_flag = True

    # ----------------------------
    # FINAL DECISION
    # ----------------------------
    if mw_flag or gf_flag:
        cycle_slip_epochs.append(i)

# ----------------------------
# PRINT OUTPUT
# ----------------------------
def dt_to_glab(dt_ns):
    dt_str = np.datetime_as_string(dt_ns, unit='s')
    dt_obj = datetime.strptime(dt_str, '%Y-%m-%dT%H:%M:%S')
    sod = (dt_obj - dt_obj.replace(hour=0, minute=0, second=0)).total_seconds()
    return f"{sod:07.2f}", dt_obj.strftime("%H:%M:%S.00")

prn = int(sat[1:])

print("\nCycle Slips (TurboEdit):")
for cs in cycle_slip_epochs:
    sod, hhmmss = dt_to_glab(time_valid[cs])
    print(f"CS 2024 020 {sod} {hhmmss} GPS {prn} - SLIP DETECTED")

print("\nData Gaps:")
for g in gap_epochs:
    gap_sec = time_sec[g] - time_sec[g-1]
    sod, hhmmss = dt_to_glab(time_valid[g])
    print(f"CS 2024 020 {sod} {hhmmss} GPS {prn} - DATA_GAP = {gap_sec:.2f}")

# ----------------------------
# PLOTTING
# ----------------------------
MW_cycles = MW / lambda_wl

MW_solid = MW_cycles.copy()

for g in gap_epochs:
    MW_solid[g] = np.nan
    MW_solid[g-1] = np.nan

plt.figure(figsize=(12,4))

plt.plot(time_valid, MW_solid, label="MW")

# gaps
plt.scatter(time_valid[gap_epochs], MW_cycles[gap_epochs],
            color='black', label='Data Gap')

# slips
plt.scatter(time_valid[cycle_slip_epochs], MW_cycles[cycle_slip_epochs],
            color='red', marker='x', label='Cycle Slip')

plt.title(f"TurboEdit Cycle Slip Detection - {sat}")
plt.xlabel("Time")
plt.ylabel("MW (cycles)")
plt.grid()
plt.legend()
plt.show()