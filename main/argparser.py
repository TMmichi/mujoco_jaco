import argparse

class ArgParser():
    def __init__(self,isbaseline=False):
        self.parser = argparse.ArgumentParser(description=None)
        if not isbaseline:
            self._add_arguments()
        else:
            self._add_arguments_baseline()

    def parse_args(self):
        return self.parser.parse_args()

    def _add_arguments_baseline(self):
        self.parser.add_argument('--task', help='task name', type=str, default='pickAndplace')
        self.parser.add_argument('--robot-file', help='environment xml', type=str, default='jaco2_curtain_torque')
        self.parser.add_argument('--n-robot', help='number of robots', type=int, default=1)
        self.parser.add_argument('--seed', help='RNG seed', type=int, default=0)
        self.parser.add_argument('--visualize', help='render environment', type=bool, default=True)
        self.parser.add_argument('--auxiliary', help='auxiliary network', type=bool, default=False)
        self.parser.add_argument('--save-interval', help='save interval', type=int, default=10000)
        self.parser.add_argument('--ent-coef', help='entropy coefficient for SAC', type=float, default=1e-7)
        self.parser.add_argument('--num-timesteps', type=int, default=1e6)