import georinex as gr
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# LOAD RINEX OBSERVATION FILE
# ----------------------------

obs = gr.load("iisc0200.24o")

print(obs)

gps_sats = [sv for sv in obs.sv.values if sv.startswith("G")]
print(gps_sats[:10])

# Select first satellite for testing
sat = gps_sats[3]

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
# MELBOURNE-WUBBENA COMBINATION
# ----------------------------

MW = (L1*lambda1 - L2*lambda2 - (P1 - P2)) / lambda_wl

print("MW sample values:")
print(MW[:10])

# ----------------------------
# DATA GAP DETECTION
# ----------------------------

interval = 30  # expected sampling interval (seconds)
gap_threshold = 1.5 * interval

time_sec = time_valid.astype('datetime64[s]').astype(int)

gap_epochs = []

for i in range(1, len(time_sec)):
    if time_sec[i] - time_sec[i-1] > gap_threshold:
        gap_epochs.append(i)

# ----------------------------
# MW CYCLE SLIP DETECTION
# ----------------------------

window = 20
slips = []

i = window

while i < len(MW):

    mean_MW = np.nanmean(MW[i-window:i])
    std_MW = np.nanstd(MW[i-window:i])

    if std_MW == 0:
        i += 1
        continue

    if abs(MW[i] - mean_MW) > 4 * std_MW:
        slips.append(i)

        # reset window after slip
        i += window
    else:
        i += 1

# ----------------------------
# PRINT DETECTED EVENTS
# ----------------------------

print("\nCycle Slips Detected:")

for s in slips:

    jump = MW[s] - MW[s-1]

    print(
        f"CS | Time: {time_valid[s]} | Satellite: {sat} | MW Jump: {jump:.2f} cycles"
    )


print("\nData Gaps Detected:")

for g in gap_epochs:

    gap = time_sec[g] - time_sec[g-1]

    print(
        f"GAP | Time: {time_valid[g]} | Satellite: {sat} | Gap: {gap} seconds"
    )

# ----------------------------
# PLOT RESULTS
# ----------------------------

plt.figure(figsize=(10,4))

plt.plot(time_valid, MW, label="MW")

plt.scatter(time_valid[slips], MW[slips],
            color='red', label="Detected Slip")

plt.scatter(time_valid[gap_epochs], MW[gap_epochs],
            color='orange', label="Data Gap")

plt.title(f"Cycle Slip Detection - {sat}")
plt.xlabel("Time")
plt.ylabel("MW cycles")

plt.legend()
plt.grid()

plt.show()
