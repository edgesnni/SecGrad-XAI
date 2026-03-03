#!/bin/bash

DATASET="Tiny Imagenet" #("MNIST" "Tiny Imagenet" "Imagenet")
MODEL="AlexNet" # ("AlexNet" "VGG11" "VGG19" "AliceNet")
X_METHOD="vanilla" # "vanilla" "xinput" "integrated"  #  "CAM" # "LRP"
MODE="secure" #  "plaintext" "secure"



if [ "$MODE" == "secure" ]; then
    sudo pkill -9 python3 iperf3 2>/dev/null || true
    sudo fuser -k 29500/tcp 2>/dev/null || true
    echo "Cleanup complete."
    source common.sh
    source throttle.sh 
else
    echo "Can only do plaintext on this machine"
    exit 1
fi

echo "====================================================="
echo ""
if [ -z "$X_METHOD" ]; then
    xmethod_display="No explanation method chosen."
else
    xmethod_display="Explain with $X_METHOD method."
fi
echo "$MODE inference for $DATASET with $MODEL ; $xmethod_display"
if [ "$MODEL" == "AliceNet" ] && [ "$DATASET" != "MNIST" ]; then
    echo "⚠️  Skip: AliceNet is only compatible with MNIST."
    echo "Current configuration: Model=$MODEL, Dataset=$DATASET"
    exit 0
fi
if [ "$MODEL" == "AliceNet" ] && [ "$X_METHOD" == "CAM" ]; then
    echo "⚠️  Skip: AliceNet does not have enough CONV layers for CAM."
    exit 0
fi


TF_ENABLE_ONEDNN_OPTS=0 TF_CPP_MIN_LOG_LEVEL=1 PYTHONWARNINGS="ignore::UserWarning" PYTHONWARNINGS="ignore::UserWarning" PYTHONUNBUFFERED=1 python3 main.py --dataset "$DATASET" --model "$MODEL" --mode "$MODE" --explain "$X_METHOD"


echo ""
echo "====================================================="
echo ""
echo "Cleaning up __pycache__ directories..."
find . -type d -name "__pycache__" -exec rm -rf {} +
echo "Cleanup complete."

