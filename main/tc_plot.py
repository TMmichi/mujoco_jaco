import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter


fig_name = "logger_csv/picking_HPC.csv"
df = pd.read_csv(fig_name)
plt.figure(figsize=(10,6))
fig_name = "logger_csv/picking_HIRO.csv"
df_HIRO = pd.read_csv(fig_name)

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def step_mod(scalars, max_step):
    num_data = len(scalars)
    step = int(max_step / num_data)+1
    return np.linspace(0,max_step, max_step+1)[::step]

th = 250000
min_index = np.where(df['Step'] == min(df['Step'][df['Step']>th]))[0][0]
print(min_index)

plt.plot(df['Step'][min_index:]-th, smooth(df['Value'],0.95)[min_index:], color='darkorange', linewidth=2, label='HPC(Ours)')
# plt.plot(df['Step'][min_index:]-th, smooth(df['Value'],0.95)[min_index:], color='darkorange', linewidth=2, label='Flat')
plt.plot(step_mod(df_HIRO['Step'],max(df['Step'])), smooth(df_HIRO['Value'],0.95), color='dodgerblue', linewidth=2, label='Hiro')
plt.plot(df['Step'][min_index:]-th, df['Value'][min_index:], color='darkorange', linewidth=2, alpha=0.3)
plt.plot(step_mod(df_HIRO['Step'],max(df['Step'])), df_HIRO['Value'], color='dodgerblue', linewidth=2, alpha=0.3)

plt.ylabel('Return', fontsize=18)
plt.xlabel('Training Steps', fontsize=18)
plt.xticks(np.arange(0,max(df['Step'])+1, 5e5))
plt.grid(axis='y', alpha=0.5, linestyle='--')
plt.legend(loc='lower right', fontsize=18)
plt.tight_layout()
plt.show()


