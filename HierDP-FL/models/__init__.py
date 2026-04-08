"""models/ – Model factory for HierDP-FL."""

import tensorflow as tf
from models.cifar_cnn import CifarCNN3Conv, CifarResNet18
from models.gtsrb_model import GTSRBConvNet, GTSRBMobileNet


def build_model(args) -> tf.keras.Model:
    """
    Build and return a Keras model based on CLI args.

    Supported combinations:
        dataset=cifar10,   model=cnn_complex  → CifarCNN3Conv
        dataset=cifar10,   model=resnet18     → CifarResNet18
        dataset=cifar100,  model=resnet18     → CifarResNet18 (100 classes)
        dataset=gtsrb,     model=convnet      → GTSRBConvNet
        dataset=gtsrb,     model=mobilenet    → GTSRBMobileNet
    """
    dataset = args.dataset.lower()
    model_name = args.model.lower()

    if dataset == "cifar10":
        num_classes = 10
        input_shape = (32, 32, 3)
        if model_name in ("cnn_complex", "cnn"):
            return CifarCNN3Conv(input_shape=input_shape, num_classes=num_classes)
        elif model_name == "resnet18":
            return CifarResNet18(input_shape=input_shape, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model '{model_name}' for cifar10. "
                             f"Options: cnn_complex, resnet18")

    elif dataset == "cifar100":
        num_classes = 100
        input_shape = (32, 32, 3)
        if model_name in ("cnn_complex", "cnn"):
            return CifarCNN3Conv(input_shape=input_shape, num_classes=num_classes)
        elif model_name == "resnet18":
            return CifarResNet18(input_shape=input_shape, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model '{model_name}' for cifar100.")

    elif dataset == "gtsrb":
        num_classes = 43
        input_shape = (32, 32, 3)
        if model_name in ("convnet", "cnn"):
            return GTSRBConvNet(input_shape=input_shape, num_classes=num_classes)
        elif model_name in ("mobilenet", "mobilenetv2"):
            return GTSRBMobileNet(input_shape=input_shape, num_classes=num_classes)
        elif model_name == "resnet18":
            return CifarResNet18(input_shape=input_shape, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model '{model_name}' for gtsrb. "
                             f"Options: convnet, mobilenet, resnet18")

    else:
        raise ValueError(f"Unknown dataset '{dataset}'. "
                         f"Options: cifar10, cifar100, gtsrb")


__all__ = [
    "build_model",
    "CifarCNN3Conv",
    "CifarResNet18",
    "GTSRBConvNet",
    "GTSRBMobileNet",
]
