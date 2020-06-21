# mujoco_jaco

Please clone this repo with the command
```bash
git clone --recursive https://github.com/TMmichi/mujoco_jaco.git
```
## Installation
Mujoco
------
If you would like to use the Mujoco API you will need to install a
forked version of `mujoco-py <https://github.com/studywolf/mujoco-py/>`_ with hooks for
exitting out of simulations with the ESC key. To use the mujoco API, make sure you are
in your anaconda environment and run::
```
git clone https://github.com/studywolf/mujoco-py.git
cd mujoco-py
pip install -e .
pip install requests
```
After installing Mujoco-py, install additional requirements via:
```
pip install -r requirements.txt
```
