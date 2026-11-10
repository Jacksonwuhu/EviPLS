"""Shared LLP dataset and DataLoader construction."""

from typing import Any, Callable, Tuple, Type

from torch.utils.data import DataLoader
from torchvision import transforms

from dataloader import LLP_dataset, ToTensor


def build_llp_dataloaders(
    args: Any,
    *,
    dataset_cls: Type[LLP_dataset] = LLP_dataset,
    transform_factory: Callable[[], Any] = lambda: transforms.Compose([ToTensor()]),
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build the standard train/validation/test LLP loaders from CLI arguments."""
    common = {
        "audio_dir": args.audio_dir,
        "video_dir": args.video_dir,
        "st_dir": args.st_dir,
        "v_pseudo_data_dir": args.v_pseudo_data_dir,
        "a_pseudo_data_dir": args.a_pseudo_data_dir,
    }
    datasets = [
        dataset_cls(label=label, transform=transform_factory(), **common)
        for label in (args.label_train, args.label_val, args.label_test)
    ]
    train_workers = getattr(args, "num_workers", 5)
    eval_workers = getattr(args, "eval_num_workers", 1)
    pin_memory = getattr(args, "pin_memory", True)
    return (
        DataLoader(datasets[0], batch_size=args.batch_size, shuffle=True,
                   num_workers=train_workers, pin_memory=pin_memory),
        DataLoader(datasets[1], batch_size=1, shuffle=False,
                   num_workers=eval_workers, pin_memory=pin_memory),
        DataLoader(datasets[2], batch_size=1, shuffle=False,
                   num_workers=eval_workers, pin_memory=pin_memory),
    )
