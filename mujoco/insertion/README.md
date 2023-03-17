# insertion
Peg-in-Hole Insertion Simulation in Mujoco using RL.

## Instructions
### General Usage
- ```source insertion_env/bin/activate```
- ```cd mer_lab/mujoco/insertion```
- ```python3 insert_test.py```

### Creating New Environments
- Create a new file in robosuite/environments/manipulation using an existing environment as a template.
- Add an associated import line to robosuite/__init__.py.

### Adding New Models
- New STL and XML files go in the appropriate subfolders of robosuite/models/assets.
- The associated environment file (i.e., insert.py) will need to be updated to actually spawn the new objects.

## Setup
### Assumptions:
- Assumptions in Mujoco README are met.
- LLVM is installed (```sudo apt install llvm```).
- python3-tk is installed (```sudo apt-get install python3-tk```)
- Note this setup can work on Windows, but it's necessary to modify robosuite to use absolute paths in the XML that is generated for Mujoco.

### First Time Setup
1. ```python3 -m venv insertion_env```
2. ```source insertion_env/bin/activate```
3. ```pip install llvmlite```
	- Verify that a .so file is built: ```sudo find / -name libllvmlite.so*```
	- TODO: Test this part. There is something strange with the numba/llvmlite installation step, but it is possible. Might need to use ```easy_install llvmlite```. The pip versions that ended up work are llvmlite (0.32.1) and numba (0.49.1). Using ```sudo``` might be required. LLVM must be installed first.
4. ```pip install gym h5py matplotlib mujoco-py pandas robosuite roboticstoolbox-python torch xlsxwriter```
5. Replace robosuite (/home/<YOUR_USERNAME>/insertion_env/lib/python3.6/site-packages/robosuite) with the custom version in this folder (mer_lab/mujoco/insertion/robosuite).

## Notes
- This was tested on Ubuntu 18.04 with Python 3.6.9.

## References
- [Peter Corke's Robotics Toolbox in Python](https://github.com/petercorke/robotics-toolbox-python)
	- Only used for calculating initial robot pose.
- [Robosuite Documentation](https://robosuite.ai/docs/index.html)
- [Vikash Kumar's Franka Panda Mujoco Model](https://github.com/vikashplus/franka_sim)
	- Note there is also one in the robosuite package that may be more complete in terms of control.