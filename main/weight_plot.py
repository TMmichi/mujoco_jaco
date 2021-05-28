import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter


fig_name = "logger_csv/log_pickAndplace_0.csv"
df = pd.read_csv(fig_name)


fig, axes = plt.subplots(2,1,figsize=(10,6), sharex=True)
plt.suptitle("Task Weights of Pick and Place", size=16)
axes[0].plot(df['Step'], df['Picking'], color='red', linewidth=1, label='Picking')
axes[0].plot(df['Step'], df['Placing'], color='mediumblue', linewidth=1, label='Placing')

axes[1].plot(df['Step'], df['Pick-Reaching'], color='dodgerblue', linestyle='-',linewidth=2, label='Pick-Reaching')
axes[1].plot(df['Step'], df['Pick-Grasping'], color='yellowgreen', linestyle='-', linewidth=2, label='Pick-Grasping')
axes[1].plot(df['Step'], df['Place-Reaching'], color='aqua', linestyle='-', linewidth=2, label='Place-Reaching')
axes[1].plot(df['Step'], df['Place-Releasing'], color='yellow', linestyle='-', linewidth=1, label='Place-Releasing')


axes[0].set_title('Weights of level 2 tasks')
axes[1].set_title('Weights of level 1 tasks')
axes[0].set_ylabel('Weights', fontsize=13)
axes[1].set_xlabel('Step', fontsize=13)
axes[1].set_ylabel('Weights', fontsize=13)
axes[1].set_xticks(np.arange(0,len(df['Step'])+1, 10))
axes[0].grid(axis='y', alpha=0.5, linestyle='--')
axes[1].grid(axis='y', alpha=0.5, linestyle='--')
axes[0].legend(loc='right', fontsize=13)
axes[1].legend(loc='right', fontsize=13)
plt.tight_layout()
plt.show()






# fig_name = "logger_csv/log_picking_0.csv"
# df = pd.read_csv(fig_name)
# plt.figure(figsize=(10,6))
# plt.title("Task Weights of Picking", size=16)
# plt.plot(df['Step'], df['Reaching'], color='dodgerblue', linewidth=2, label='Reaching')
# plt.plot(df['Step'], df['Grasping'], color='yellowgreen', linewidth=2, label='Grasping')
# plt.ylabel('Weights', fontsize=13)
# plt.xlabel('Step', fontsize=13)
# plt.xticks(np.arange(0,len(df['Step'])+1, 10))
# plt.grid(axis='y', alpha=0.5, linestyle='--')
# plt.legend(loc='right', fontsize=13)
# plt.tight_layout()
# plt.show()


# fig_name = "logger_csv/log_placing_0.csv"
# df = pd.read_csv(fig_name)
# plt.figure(figsize=(10,6))
# plt.title("Task Weights of Placing", size=16)
# plt.plot(df['Step'], df['Reaching'], color='goldenrod', linewidth=1, label='Reaching')
# plt.plot(df['Step'], df['Releasing'], color='dodgerblue', linewidth=1, label='Releasing')
# plt.ylabel('Weights', fontsize=13)
# plt.xlabel('Steps', fontsize=13)
# plt.xticks(np.arange(0,len(df['Step'])+1, 10))
# plt.grid(axis='y', alpha=0.5, linestyle='--')
# plt.legend(loc='right', fontsize=13)
# plt.tight_layout()
# plt.show()
