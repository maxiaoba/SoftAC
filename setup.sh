# conda activate garage
export PYTHONPATH=$PYTHONPATH:$PWD
export PYTHONPATH=$PYTHONPATH:$PWD/tests
export PYTHONPATH=$PYTHONPATH:$PWD/../rllab
# export MUJOCO_PY_MJKEY_PATH=$PWD/../rllab/vendor/mujoco
# this doesn't work, please put mjkey.txt at ~/.mujoco
export MUJOCO_PY_MJPRO_PATH=$PWD/../rllab/vendor/mujoco/mjpro131

