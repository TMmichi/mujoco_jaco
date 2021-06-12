import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


plt.ion()
fig, axes = plt.subplots(2,1, figsize=(10,6), sharex=True)

pick, = axes[0].plot([], [], color='red', linewidth=1, label='Picking')
place, = axes[0].plot([], [], color='mediumblue', linewidth=1, label='Placing')
pick_reach, = axes[1].plot([], [], color='dodgerblue', linestyle='-',linewidth=2, label='Pick-Reaching')
pick_grasp, = axes[1].plot([], [], color='yellowgreen', linestyle='-', linewidth=2, label='Pick-Grasping')
place_reach, = axes[1].plot([], [], color='aqua', linestyle='-', linewidth=2, label='Place-Reaching')
place_release, = axes[1].plot([], [], color='yellow', linestyle='-', linewidth=1, label='Place-Releasing')

axes[0].set_title('Intents of level 2 tasks')
axes[1].set_title('Intents of level 1 tasks')
axes[0].set_ylabel('Intents', fontsize=13)
axes[1].set_xlabel('Step', fontsize=13)
axes[1].set_ylabel('Intents', fontsize=13)
axes[0].grid(axis='y', alpha=0.5, linestyle='--')
axes[1].grid(axis='y', alpha=0.5, linestyle='--')
axes[0].legend(loc='right', fontsize=13)
axes[1].legend(loc='right', fontsize=13)
plt.xlim(0, 200)
axes[0].set_ylim(-0.05, 1.05)
axes[1].set_ylim(-0.05, 1.05)

x_val, y1_val, y2_val, y3_val, y4_val, y5_val, y6_val = [], [], [], [], [], [], []
def animate(data):
    x, y1, y2, y3, y4, y5, y6 = data
    x_val.append(x)
    y1_val.append(y1)
    y2_val.append(y2)
    y3_val.append(y3)
    y4_val.append(y4)
    y5_val.append(y5)
    y6_val.append(y6)
    print(data)
    pick.set_data(x_val, y1_val)
    place.set_data(x_val, y2_val)
    pick_reach.set_data(x_val, y3_val)
    pick_grasp.set_data(x_val, y4_val)
    place_reach.set_data(x_val, y5_val)
    place_release.set_data(x_val, y6_val)
    fig.canvas.draw()
    fig.canvas.flush_events()

for i in range(200):
    ret1 = random.random()
    ret2 = random.random()
    ret3 = random.random()
    ret4 = random.random()
    ret5 = random.random()
    ret6 = random.random()
    data = (i, ret1, ret2, ret3, ret4, ret5, ret6)
    animate(data)

# ani = FuncAnimation(fig, animate, data_gen, blit=True, interval=1)
# plt.tight_layout()
# plt.show()