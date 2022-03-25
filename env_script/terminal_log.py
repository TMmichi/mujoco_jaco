import numpy as np

def logging(obs, prev_obs, action, wb, reward):
    write_str = "Act:"
    for i in range(len(action)):
        write_str += " {0: 2.3f}".format(action[i])
    write_str += "\t| Obs:" 
    write_log = write_str
    write_str += str(int(obs[0]))
    for i in range(1,1+len(action)):
        write_str += colored_string(obs[i],prev_obs[i],action[i-1])
        write_log += ", {0: 2.3f}".format(obs[i])
    write_str += "\t| wb = {0: 2.3f} | \033[92mReward:\t{1:1.5f}\033[0m".format(wb,reward)
    write_log += "\t| wb = {0: 2.3f} | Reward:\t{1:1.5f}".format(wb,reward)
    print(write_str, end='\r')

def colored_string(self, obs_val, prev_obs_val, action):
    if int(np.sign(obs_val-prev_obs_val)) == int(np.sign(action)):
        return "\t\033[92m{0:2.3f}\033[0m".format(obs_val)
    else:
        return "\t\033[91m{0:2.3f}\033[0m".format(obs_val)