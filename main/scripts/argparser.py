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
        #self.parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
        self.parser.add_argument('--seed', help='RNG seed', type=int, default=0)
        #self.parser.add_argument('--num-timesteps', type=int, default=int(10500))
        self.parser.add_argument('--play', default=False, action='store_true')