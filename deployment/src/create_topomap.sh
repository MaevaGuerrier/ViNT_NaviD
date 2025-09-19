#!/bin/bash

# Failsafe to make sure that pip install -e has been executed
# This is necessary to ensure that the package needed are present
# eval "$(conda shell.bash hook)"
# conda activate vint_deployment
# Navigate to the directory containing the package
# Install the package in editable mode

# Change back the directory to the working dir with the navigate.py script
cd /workspace/src/visualnav-transformer/deployment/src

# Create a new tmux session
session_name="vint_locobot_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into three panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves
tmux selectp -t 0    # go back to the first pane
tmux splitw -h -p 50 # split it into two halves

# Run roscore in the first pane
tmux select-pane -t 0
tmux send-keys "roscore" Enter

# Run the create_topoplan.py script with command line args in the second pane
tmux select-pane -t 1
# tmux send-keys "conda activate vint_deployment" Enter
tmux send-keys "python3 create_topomap.py --dt 1 --dir $1" Enter

# Change the directory to ../topomaps/bags and run the rosbag play command in the third pane
tmux select-pane -t 2
tmux send-keys "mkdir -p ../topomaps/bags" Enter
tmux send-keys "cd ../topomaps/bags" Enter
tmux send-keys "rosbag play -r 1.5 $2" # feel free to change the playback rate to change the edge length in the graph

# Attach to the tmux session
tmux -2 attach-session -t $session_name