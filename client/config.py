
from typing import Callable, Union, Dict, Any
from secure.secure_explainables import *
from plaintext.models import *
from secure.secure_models import *
from torchvision import models
from plaintext.explainables import *

import warnings
warnings.filterwarnings("ignore")

PLAINTEXT_CONFIG: Dict[str, Dict[str, Any]] = {
    "CAM": {
        "title": "Gradient-weighted Class Activation Mapping",
        "short_title": "Grad-CAM",
        "class": GradCAM,
    },
    "vanilla": {
        "title": "Vanilla Gradients",
        "short_title": "Grad-Vanilla",
        "class": VanillaGradients,
    },
    "xinput": {
        "title": "Gradients x Input",
        "short_title": "Grad-xInput",
        "class": GradientxInput,
    },
    "integrated": {
        "title": "Integrated Gradients",
        "short_title": "Grad-Integrated",
        "class": IntegratedGradients,
    },
    "LRP": {
        "title": "Layer-wise Relevance Propagration",
        "short_title": "LRP",
        "class": LayerwiseRelevancePropagation,
    },
}

SECURE_CONFIG: Dict[str, Dict[str, Any]] = {
    "vanilla": {
        "title": "Secure Vanilla Gradients",
        "short_title": "SecureGrad-Vanilla",
        "class": SecureVanillaGradients,
    },

    "xinput": {
        "title": "Secure Gradients x Input",
        "short_title": "SecureGrad-xInput",
        "class": SecureGradientxInput,
    },
    "integrated": {
        "title": "Secure Integrated Gradients",
        "short_title": "SecureGrad-Integrated",
        "class": SecureIntegratedGradients,
    },
    "CAM": {
        "title": "Secure Gradient-weighted Class Activation Mapping",
        "short_title": "SecureGrad-CAM",
        "class": SecureGradCAM,
    },
    "LRP": {
        "title": "Secure Layer-wise Relevance Propagration",
        "short_title": "Secure LRP",
        "class": SecureLRP,
    },
}

MODEL_CONFIG = {
    'AlexNet': {
        'plaintext': AlexNet,
        'secure': AlexNet,
    },
    'AliceNet': {
        'plaintext': AliceNet,
        'secure': AliceNet,
    },
    'VGG11': {
        'plaintext': VGG11,
        'secure': VGG11,
    },
    'VGG19': {
        'plaintext': VGG19,
        'secure': VGG19,
    },
    'Google': {
        'plaintext': GoogLeNet,
        'secure': SecureGoogLeNet,
    },
    'ResNet18': {
        'plaintext': CustomResNet18,
        'secure': CustomResNet18,
    },
    'ResNet50': {
        'plaintext': CustomResNet50,
        'secure': CustomResNet50,
    },
    'ResNet152': {
        'plaintext': CustomResNet152,
        'secure': CustomResNet152,
    }
}

IMAGENET_LABELS = {
    "test_0_image.jpg": 207,  # Labrador retriever (Placeholder index)
    "test_1_image.jpg": 1,    # tench (Placeholder index)
    "test_2_image.jpg": 8,    # hen (Placeholder index)
    "test_3_image.jpg": 7,    # fire salamander (Placeholder index)
    "test_4_image.jpg": 36,   # leatherback turtle (Placeholder index)
    "test_5_image.jpg": 40,   # American alligator (Placeholder index)
    "test_6_image.jpg": 72,   # wolf spider (Placeholder index)
    "test_7_image.jpg": 105,  # sea slug (Placeholder index)
    "test_8_image.jpg": 150,  # Chihuahua (Placeholder index)
    "test_9_image.jpg": 333,  # golden hamster (Placeholder index)
}

ALL_CLASS_NAMES = [
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead", 
    "fire salamander", # Index 7
    "hen",             # Index 8
    "Labrador retriever", # Index 207
    "golden hamster",     # Index 333
]