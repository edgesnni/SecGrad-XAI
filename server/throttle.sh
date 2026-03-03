#!/bin/bash

set -e
DEV= #Same as defined in common.sh

sudo tc qdisc del dev $DEV root || true
sudo tc qdisc add dev $DEV root handle 1: tbf rate 10gbit burst 100000 limit 1000m
sudo tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 1msec

if [ "$RANK" == "0" ]; then
    echo "Starting iperf3 validation server (port 5201)..."
    sleep 1
    iperf3 -s -p 5201 -1
else
    exit 1 
fi
