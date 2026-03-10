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