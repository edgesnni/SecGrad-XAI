#!/bin/bash

DATASET="" #"MNIST" "Tiny Imagenet" "Imagenet"
MODEL="" # "AlexNet" "VGG11" "VGG19" "AliceNet"
X_METHOD="" # "vanilla" "xinput" "integrated" "CAM" "LRP"
MODE="" #  "plaintext" "secure"
INFERENCE="" # "single" "multiple"
MAP="yes" # "yes"  "no"
INDEX=
BASELINE="" # "zero" "gray" "white" "random" "blur"
STEPS=2
EPSILON=0.00000001
NORMALIZED="no"

if [ "$MODE" == "secure" ]; then
    sudo pkill -9 python3 iperf3 2>/dev/null || true
    sudo fuser -k 29500/tcp 2>/dev/null || true
    echo "Cleanup complete."
    source common.sh
    source throttle.sh
fi

echo "====================================================="
echo ""
if [ -z "$X_METHOD" ]; then
    xmethod_display="No explanation method chosen."
else
    xmethod_display="Explain with $X_METHOD method."
fi

echo "$INFERENCE ($MODE) inference for $DATASET with $MODEL ; $xmethod_display"

if [ "$MODEL" == "AliceNet" ] && [ "$DATASET" != "MNIST" ]; then
    echo "⚠️  Skip: AliceNet is only compatible with MNIST."
    echo "Current configuration: Model=$MODEL, Dataset=$DATASET"
    exit 0
fi

if [ "$MODEL" == "AliceNet" ] && [ "$X_METHOD" == "CAM" ]; then
    echo "⚠️  Skip: AliceNet does not have enough CONV layers for CAM."
    exit 0
fi

if [ "$INFERENCE" == "multiple" ]; then
    X_METHOD=""
fi

TF_ENABLE_ONEDNN_OPTS=0 TF_CPP_MIN_LOG_LEVEL=1 PYTHONWARNINGS="ignore::UserWarning" PYTHONUNBUFFERED=1 python3 main.py --dataset "$DATASET" --model "$MODEL" --mode "$MODE" --explain "$X_METHOD" --inference "$INFERENCE" --idx $INDEX --steps $STEPS --baseline "$BASELINE" --epsilon $EPSILON --normalized "$NORMALIZED" --map "$MAP"

echo ""
echo "====================================================="
echo ""
echo "Cleaning up __pycache__ directories..."
find . -type d -name "__pycache__" -exec rm -rf {} +
echo "Cleanup complete."
