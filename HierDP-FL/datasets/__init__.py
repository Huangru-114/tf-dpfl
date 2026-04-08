"""datasets/ – Dataset factory for HierDP-FL."""
from datasets.cifar10 import get_cifar_dataloaders
from datasets.gtsrb import get_gtsrb_dataloaders


def get_dataloaders(args):
    """
    Unified data loader factory.

    Returns
    -------
    train_datasets    : List[tf.data.Dataset]  – one per client
    test_datasets     : List[tf.data.Dataset]  – one per client
    v_train_dataset   : tf.data.Dataset        – global train set
    v_test_dataset    : tf.data.Dataset        – global test set
    num_train_samples : List[int]              – per-client sample counts
    """
    dataset = args.dataset.lower()
    if dataset in ("cifar10", "cifar100"):
        return get_cifar_dataloaders(args)
    elif dataset == "gtsrb":
        return get_gtsrb_dataloaders(args)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Options: cifar10, cifar100, gtsrb")


__all__ = ["get_dataloaders", "get_cifar_dataloaders", "get_gtsrb_dataloaders"]
