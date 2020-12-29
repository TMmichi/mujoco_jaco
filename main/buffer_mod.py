import numpy as np
from pathlib import Path


# traj_dict = np.load("../models_baseline/trajectories/grasping_trajectory_expert5.npz", allow_pickle=True)
traj_dict = np.load("../models_baseline/trajectories/grasping_trajectory_expert5_mod.npz", allow_pickle=True)
obs = np.copy(traj_dict['obs'])
print("episodes: ", len(traj_dict['actions']))
for i in range(len(traj_dict['actions'])):
    before = obs[i][7]
    # after = np.around(((before*0.35+0.65)-0.8)/0.2,5)
    # obs[i][7] = after
    print(before)

# new_dict = {}
# new_dict.update(traj_dict)
# new_dict['obs'] = obs
# for i in range(100):
#     print(new_dict['obs'][i][7])
# np.savez("../models_baseline/trajectories/grasping_trajectory_expert5_mod.npz", **new_dict)