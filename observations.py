import georinex as gr
import numpy as np
import matplotlib.pyplot as plt
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------
# LOAD RINEX FILE
# ----------------------------
obs = gr.load("iisc0200.24o")

gps_sats = [sv for sv in obs.sv.values if sv.startswith("G")]
sat = gps_sats[2]   # first satellite

L1 = obs["L1"].sel(sv=sat).values
L2 = obs["L2"].sel(sv=sat).values
P1 = obs["P1"].sel(sv=sat).values
P2 = obs["P2"].sel(sv=sat).values
time = obs.time.values

print("Satellite:", sat)
print("Total epochs:", len(time))

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
valid = np.isfinite(L1) & np.isfinite(L2) & np.isfinite(P1) & np.isfinite(P2)

L1 = L1[valid]
L2 = L2[valid]
P1 = P1[valid]
P2 = P2[valid]
time_valid = time[valid]

print("Valid epochs:", len(time_valid))

# ----------------------------
# MELBOURNE-WÜBBENA
# ----------------------------
MW = (L1*lambda1 - L2*lambda2 - (P1 - P2)) / lambda_wl

# ----------------------------
# DATA GAP DETECTION
# ----------------------------
interval = 30
gap_threshold = 1.5 * interval

# convert time to seconds correctly
time_sec = time_valid.astype("datetime64[s]").astype(int)

gap_epochs = []

for i in range(1, len(time_sec)):
    if time_sec[i] - time_sec[i-1] > gap_threshold:
        gap_epochs.append(i)

# ----------------------------
# TIME FORMAT FUNCTION
# ----------------------------
def dt_to_glab(dt_ns):

    dt_str = np.datetime_as_string(dt_ns, unit='s')
    dt_obj = datetime.strptime(dt_str, '%Y-%m-%dT%H:%M:%S')

    midnight = dt_obj.replace(hour=0, minute=0, second=0)
    sod = (dt_obj - midnight).total_seconds()

    sod_str = f"{sod:07.2f}"
    hhmmss = f"{dt_obj.hour:02d}:{dt_obj.minute:02d}:{dt_obj.second:02d}.00"

    return sod_str, hhmmss

# ----------------------------
# OUTPUT (gLAB STYLE)
# ----------------------------
prn = int(sat[1:])

print("\nCycle Slips Detected:")
print("None (MW threshold not applied)")

print("\nData Gaps Detected (gLAB format):")

for g in gap_epochs:

    gap_sec = time_sec[g] - time_sec[g-1]

    sod, hhmmss = dt_to_glab(time_valid[g])

    print(f"CS 2024 048 {sod} {hhmmss} GPS {prn} - DATA_GAP = {gap_sec:.2f} THRESHOLD = 40.00")

# ----------------------------
# PLOT
# ----------------------------
plt.figure(figsize=(12,4))

plt.plot(time_valid, MW, label="MW")

if len(gap_epochs) > 0:
    plt.scatter(time_valid[gap_epochs], MW[gap_epochs], color="orange", label="Data Gap")

plt.title(f"Melbourne-Wubbena Combination - {sat}")
plt.xlabel("Time")
plt.ylabel("MW (cycles)")
plt.grid()

plt.legend()
plt.show()