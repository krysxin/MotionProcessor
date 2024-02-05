#!/bin/bash

# Set the flag 1: local test 2: realtime test 
flag="$1"
# Check if the flag is provided
if [ -z "$flag" ]; then
    echo "Please provide a number flag as input, 1 for local test, 2 for realtime test."
    exit 1
fi

# input_file="PoseData/3DDynamicPoseData_1207/misSwipeWSID8.txt"
input_file="Data/movementShuaiID34587.txt"

if [ "$flag" -eq 1 ]; then
    cat "$input_file" | python PoseRecognizor/recognizor_wrapper.py
elif [ "$flag" -eq 2 ]; then
    mosquitto_sub -h 192.168.0.76 -t sony/mciot/jsonfused | python PoseRecognizor/recognizor_wrapper.py
elif [ "$flag" -eq 3 ]; then
    mosquitto_sub -h 192.168.1.100 -t sony/jsonfused | python PoseRecognizor/recognizor_wrapper_gym2.py
fi