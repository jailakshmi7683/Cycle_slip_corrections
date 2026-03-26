import georinex as gr
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------
# LOAD RINEX
# ----------------------------
obs = gr.load("iisc0200.24o")

gps_sats = [sv for sv in obs.sv.values if sv.startswith("G")]
sat = gps_sats[0]

L1 = obs["L1"].sel(sv=sat).values
L2 = obs["L2"].sel(sv=sat).values
P1 = obs["P1"].sel(sv=sat).values
P2 = obs["P2"].sel(sv=sat).values
time = obs.time.values

print("Satellite:", sat)

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
# MW COMPUTATION (cycles → meters)
# ----------------------------
MW_cycles = (L1*lambda1 - L2*lambda2 - (P1 - P2)) / lambda_wl
MW = MW_cycles * lambda_wl   # meters

# ----------------------------
# TIME
# ----------------------------
time_sec = time_valid.astype("datetime64[s]").astype(int)

# ----------------------------
# PARAMETERS (gLAB)
# ----------------------------
gap_threshold = 40        # seconds
window_sec = 300
sampling = 30             # assumed
window_size = int(window_sec / sampling)

k = 5
min_sigma = 0.8
N_required = 2

# ----------------------------
# DETECTION
# ----------------------------
gap_epochs = []
cycle_slip_epochs = []

candidate_count = 0

for i in range(1, len(MW)):

    dt = time_sec[i] - time_sec[i-1]

    # ----------------------------
    # DATA GAP
    # ----------------------------
    if dt > gap_threshold:
        gap_epochs.append(i)
        candidate_count = 0
        continue

    # ----------------------------
    # WINDOW CHECK
    # ----------------------------
    if i < window_size:
        continue

    window = MW[i-window_size:i]

    if np.any(np.isnan(window)):
        continue

    mean = np.mean(window)
    sigma = np.std(window)

    sigma = max(sigma, min_sigma)

    # ----------------------------
    # TEST
    # ----------------------------
    if abs(MW[i] - mean) > k * sigma:
        candidate_count += 1
    else:
        candidate_count = 0

    # ----------------------------
    # CONFIRM SLIP
    # ----------------------------
    if candidate_count >= N_required:
        cycle_slip_epochs.append(i)
        candidate_count = 0

# ----------------------------
# TIME FORMAT
# ----------------------------
def dt_to_glab(dt_ns):

    dt_str = np.datetime_as_string(dt_ns, unit='s')
    dt_obj = datetime.strptime(dt_str, '%Y-%m-%dT%H:%M:%S')

    midnight = dt_obj.replace(hour=0, minute=0, second=0)
    sod = (dt_obj - midnight).total_seconds()

    sod_str = f"{sod:07.2f}"
    hhmmss = f"{dt_obj.hour:02d}:{dt_obj.minute:02d}:{dt_obj.second:02d}.00"

    return sod_str, hhmmss

prn = int(sat[1:])

# ----------------------------
# OUTPUT
# ----------------------------
print("\nCycle Slips Detected (gLAB-style):")

for cs in cycle_slip_epochs:
    sod, hhmmss = dt_to_glab(time_valid[cs])
    print(f"CS 2024 020 {sod} {hhmmss} GPS {prn} - MW_DEVIATION DETECTED")

print("\nData Gaps Detected:")

for g in gap_epochs:
    gap_sec = time_sec[g] - time_sec[g-1]
    sod, hhmmss = dt_to_glab(time_valid[g])
    print(f"CS 2024 020 {sod} {hhmmss} GPS {prn} - DATA_GAP = {gap_sec:.2f}")

# ----------------------------
# PLOTTING
# ----------------------------
MW_full = MW_cycles.copy()
MW_solid = MW_cycles.copy()

for g in gap_epochs:
    MW_solid[g] = np.nan
    MW_solid[g-1] = np.nan

plt.figure(figsize=(12,4))

# Valid MW
plt.plot(time_valid, MW_solid, label="MW (valid)")

# Gap dotted lines
for g in gap_epochs:
    plt.plot([time_valid[g-1], time_valid[g]],
             [MW_full[g-1], MW_full[g]],
             linestyle='dotted',
             color='red',
             linewidth=2,
             label="Fake MW (gap)" if g == gap_epochs[0] else "")

# Gap markers
plt.scatter(time_valid[gap_epochs], MW_full[gap_epochs],
            color='black', label='Data Gap', zorder=5)

# Circle gaps
for g in gap_epochs:
    plt.scatter(time_valid[g], MW_full[g],
                s=200, facecolors='none',
                edgecolors='red', linewidths=2)

# Cycle slips
if len(cycle_slip_epochs) > 0:
    plt.scatter(time_valid[cycle_slip_epochs],
                MW_full[cycle_slip_epochs],
                color='red', marker='x',
                s=80, label='Cycle Slip', zorder=6)

plt.title(f"gLAB-style MW Cycle Slip Detection - {sat}")
plt.xlabel("Time")
plt.ylabel("MW (cycles)")
plt.grid()
plt.legend()
plt.show()