# mujoco_jaco

Please clone this repo with the command

```bash
git clone --recursive https://github.com/TMmichi/mujoco_jaco.git
```

## Installation

### Mujoco

------
From abr_control:
If you would like to use the Mujoco API you will need to install a
forked version of `mujoco-py <https://github.com/studywolf/mujoco-py/>`_ with hooks for
exitting out of simulations with the ESC key. To use the mujoco API, make sure you are
in your anaconda environment and run::

```bash
git clone https://github.com/studywolf/mujoco-py.git
cd mujoco-py
pip install -e .
pip install requests
```

------
After installing mujoco-py, install additional requirements via:

```bash
pip install -r requirements.txt
```

#### Note

If you get `No matching distribution found` error for tensorflow==1.15.0, you should upgrade your pip version to the latest one.

### Pyspacenav

Python API for 3D Spacenav can be found [here](https://github.com/mastersign/pyspacenav)

## Usage

```bash
cd main
python main.py
```
