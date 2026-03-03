#!/bin/bash

set -e
DEV= #Same as in common.sh

sudo tc qdisc del dev $DEV root || true
sudo tc qdisc add dev $DEV root handle 1: tbf rate 10gbit burst 100000 limit 1000m
sudo tc qdisc add dev $DEV parent 1:1 handle 10: netem delay 1msec

if [ "$RANK" == "1" ]; then
    echo "Connecting to server $MASTER_ADDR..."
    # Attempt iperf3 with a retry logic
    until iperf3 -c $MASTER_ADDR -p 5201 -t 2; do
      echo "Server not ready yet, retrying in 2 seconds..."
      sleep 2
    done
    echo "Validation test finished."
else
    exit 1
fi
