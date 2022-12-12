# Learning Heuristic Search via Imitation

Official repository containing OpenAI Gym environments, agents and ML models for the CoRL paper [Learning Heuristic Search via Imitation](https://arxiv.org/pdf/1707.03034.pdf)

# Getting Started

## Create a virtual environment

```
python3 -m venv myenv
source myenv/bin/activate
```

## Cloning the required repositories

```
mkdir heuristics_learning
cd heuristics_learning
git clone https://github.com/naveentvelu/SaIL.git
git clone https://github.com/naveentvelu/planning_python.git
git clone https://github.com/naveentvelu/motion_planning_datasets.git
```
## Installing the required dependencies:

```
pip install gym
pip install numpy
pip install matplotlib
pip install tensorflow
pip install tflearn
pip install scipy
pip install torch
```

## Install dubins

```
clone - git clone https://github.com/AndrewWalker/pydubins
cd pydubins
rm dubins/dubins.c
python setup.py build_ext --inplace
```
Refer https://github.com/AndrewWalker/pydubins/issues/16 (if facing issues with installing dubins). If install is unsuccessful, reinstall dubins after `pip install Cython` 

## Necessary imports
These can be added to the bash profile of the virtual environment. After adding activate the environment in the heuristics_learning folder.

```
export CURR_DIR=$(cd "$(dirname "$1")" && pwd -P)/$(basename "$1")
export PYTHONPATH=$PYTHONPATH:$CURR_DIR/SaIL/examples/
export PYTHONPATH=$PYTHONPATH:$CURR_DIR/SaIL/SaIL/oracle/
export PYTHONPATH=$PYTHONPATH:$CURR_DIR/SaIL/SaIL/learners/
export PYTHONPATH=$PYTHONPATH:$CURR_DIR/SaIL/SaIL/envs/
export PYTHONPATH=$PYTHONPATH:$CURR_DIR/SaIL/SaIL/planners/
export PYTHONPATH=$PYTHONPATH:$CURR_DIR/SaIL/SaIL/agents/
export PYTHONPATH=$PYTHONPATH:$CURR_DIR/SaIL/SaIL/
export PYTHONPATH=$PYTHONPATH:$CURR_DIR/planning_python/planning_python/data_structures/
export PYTHONPATH=$PYTHONPATH:$CURR_DIR/planning_python/planning_python/state_lattices/common_lattice/
export PYTHONPATH=$PYTHONPATH:$CURR_DIR/planning_python/planning_python/state_lattices/
export PYTHONPATH=$PYTHONPATH:$CURR_DIR/planning_python/planning_python/environment_interface/
export PYTHONPATH=$PYTHONPATH:$CURR_DIR/planning_python/planning_python/
export PYTHONPATH=$PYTHONPATH:$CURR_DIR/planning_python/planning_python/heuristic_functions/
export PYTHONPATH=$PYTHONPATH:$CURR_DIR/planning_python/planning_python/utils/
export PYTHONPATH=$PYTHONPATH:$CURR_DIR/planning_python/planning_python/cost_functions/
export PYTHONPATH=$PYTHONPATH:$CURR_DIR/planning_python/planning_python/planners
export PYTHONPATH=$PYTHONPATH:$CURR_DIR/pydubins/
```

Installtion is now complete.

## Generating oracles
 - Get the SaIL repository 
 - Go to the ``examples/`` folder: ``cd ~/heuristics_learning/SaIL/examples``
 - Run ``./run_generate_oracles_xy.sh`` which will generate oracles for all the train, validation and test datasets in the ``motion_planning_datasets`` folder
 - Run ``./run_sail_xy_train.sh`` to train a heuristic for one of the datasets (you can specify the dataset you want inside the script). This runs SaIL for 10 iterations by default. For more information on the rest of the parameters used see the file ``sail_xy_train.py`` 

# Contact
For more information contact
Mohak Bhardwaj : [mohak.bhardwaj@gmail.com](mohak.bhardwaj@gmail.com)
Naveen Thangavelu: [tnaveen1998@gmail.com](tnaveen1998@gmail.com)
Mahesh Jayasankar
Adeem Jassani
## Main Changes from the Original Repo
- Python 2 to Python 3
- Added Pytorch networks (Linear Network and CNN)
- Added Admissibility Loss function