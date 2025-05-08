import os
import warnings
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, List, Callable, Any

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image


class DomainRole(Enum):
    SOURCE = "source"
    TARGET = "target"


class PreprocessingPipeline:
    def __init__(self, source_transform, target_transform):
        self.source_transform = source_transform
        self.target_transform = target_transform

    def __call__(self, sample, role: DomainRole):
        if role == DomainRole.SOURCE:
            return self.source_transform(sample)
        else:
            return self.target_transform(sample)


class SingleDomainDataset(Dataset):
    IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')

    def __init__(self, role: DomainRole, root: str,
                 transforms: Optional[PreprocessingPipeline] = None):
        self.role = role
        self.root = Path(root)
        self.transforms = transforms

        self.img_paths, self.labels, self.classes = self._discover_data(self.root)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[int, torch.Tensor, int]:
        img_path = self.img_paths[idx]
        label_name = self.labels[idx]
        label = self.class_to_idx[label_name]
        image = read_image(str(img_path))
        if self.transforms:
            image = self.transforms(image, self.role)
        return idx, image, label

    @staticmethod
    def _discover_data(root: Path) -> tuple[
        list[Path], list[str], list[Any]]:

        img_paths: list[Path] = []
        labels: list[str] = []
        class_set = set()

        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.lower().endswith(SingleDomainDataset.IMG_EXTENSIONS):
                    img_path = Path(dirpath) / filename
                    label = Path(dirpath).name  # class name is the immediate parent
                    class_set.add(label)
                    img_paths.append(img_path)
                    labels.append(label)
        classes = sorted(class_set)
        return img_paths, labels, classes

    def set_role(self, role: DomainRole):
        self.role = role


class CyclicDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        while True:
            yield from super().__iter__()


class MultiDomainDataset:
    def __init__(self, root: str, domains: List[str], roles: List[DomainRole],
                 transforms: List[Optional[PreprocessingPipeline]] = None,
                 seed: int = 0, split_ratio: float = 0.80,
                 ):
        # FIXME: forces the user to provide domains explicitly, but could be inferred
        #        (although the current approach ensures correct pairing of domain/role/transform)
        """
        Utility class to handle multiple SingleDomainDatasets.
        Each domain role (source, target) can be dynamically set.
        The preprocessing transformations are applied accordingly to the domain role.
        The dataloader semantics differ depending on the domain role.
        """
        self.root = root
        self.domains = sorted(domains)
        self.roles = roles
        self.transforms = transforms
        self.seed = seed
        self.split_ratio = split_ratio

        self._discover_domains()
        assert len(domains) == len(roles)
        if transforms:
            assert len(transforms) == len(domains)

        self.datasets = []
        self.splits = []

        self._build_datasets()
        self._make_splits()

        assert len(self.datasets) == len(self.domains)

    def size(self) -> List[int]:
        return [len(ds) for ds in self.datasets]

    def refresh_splits(self, new_seed: Optional[int] = None, new_split_ratio: Optional[float] = None):
        if new_seed is not None:
            self.seed = new_seed
        if new_split_ratio is not None and 0. < new_split_ratio < 1.:
            self.split_ratio = new_split_ratio
        self.splits.clear()
        self._make_splits()

    def build_source_dataloaders(
            self,
            use_splits: bool = False,
            batch_size: int = 1,
            shuffle: bool = True,
            num_workers: int = 0,
            pin_memory: bool = False,
            drop_last: bool = True,  # FIXME: prob. should always be True
            collate_fn: Optional[Callable] = None,
            persistent_workers: Optional[bool] = None,
    ) -> Tuple[int, List[CyclicDataLoader]] | Tuple[List[int], List[List[CyclicDataLoader]]]:
        """
        Creates one dataloader per source domain.
        They can be combined with zip(*dataloaders) to get the single (infinite) iterable over all domains.
        Each dataloader is cyclic, meaning that it will wrap around when it reaches the end of the dataset.
        This is required to handle domains with different numbers of samples.
        Within-batch domain balance is ensured, as the same number of samples is drawn from each domain.
        Since the loaders are infinite, we need to explicitly define how long an epoch lasts.
        We adopt max(domain_size) // batch_size as the epoch length.
        Due to cyclic nature and shuffling, some samples can appear more often, but the effect should be small.
        """
        if not use_splits:
            steps_per_epoch = max(self.size()) // batch_size
            dataloaders = [
                CyclicDataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    drop_last=drop_last,
                    collate_fn=collate_fn,
                    persistent_workers=persistent_workers,
                )
                for role, dataset in zip(self.roles, self.datasets)
                if role == DomainRole.SOURCE
            ]
            return steps_per_epoch, dataloaders

        else:
            steps_per_epoch = [
                max(len(split[i]) for split in self.splits) // batch_size
                for i in range(2)
            ]
            dataloaders = [
                [
                    CyclicDataLoader(
                        splits[i],
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        drop_last=drop_last,
                        collate_fn=collate_fn,
                        persistent_workers=persistent_workers,
                    )
                    for role, splits in zip(self.roles, self.splits)
                    if role == DomainRole.SOURCE
                ]
                for i in range(2)
            ]
            return steps_per_epoch, dataloaders

    def build_target_dataloaders(
            self,
            use_splits: bool = False,
            batch_size: int = 1,
            shuffle: bool = False,
            num_workers: int = 0,
            pin_memory: bool = False,
            drop_last: bool = False,
            collate_fn: Optional[Callable] = None,
            persistent_workers: Optional[bool] = None,
    ) -> list[DataLoader[Any]] | list[list[DataLoader[Any]]]:
        """
        Creates one dataloader per target domain.
        Assuming the target domains are used only for evaluation, we do not need to pair them.
        This means that a basic torch DataLoader is enough.
        """
        if not use_splits:
            dataloaders = [
                DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    drop_last=drop_last,
                    collate_fn=collate_fn,
                    persistent_workers=persistent_workers,
                )
                for role, dataset in zip(self.roles, self.datasets)
                if role == DomainRole.TARGET
            ]
            return dataloaders
        else:
            dataloaders = [
                [
                    DataLoader(
                        splits[i],
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        drop_last=drop_last,
                        collate_fn=collate_fn,
                        persistent_workers=persistent_workers,
                    )
                    for role, splits in zip(self.roles, self.splits)
                    if role == DomainRole.TARGET
                ]
                for i in range(2)
            ]
            return dataloaders

    def _discover_domains(self):
        root_path = Path(self.root)
        if not root_path.is_dir():
            raise NotADirectoryError(f"Dataset root '{self.root}' is not a valid directory.")
        available_domains = {d.name for d in root_path.iterdir() if d.is_dir()}
        expected_domains = set(self.domains)
        missing = expected_domains - available_domains
        if missing:
            raise FileNotFoundError(
                f"The following expected domains are missing under '{self.root}': {sorted(missing)}"
            )
        extra = available_domains - expected_domains
        if extra:
            warnings.warn(
                f"The following domains exist under '{self.root}' but are not listed in self.domains: {sorted(extra)}",
                UserWarning
            )

    def _build_datasets(self):
        if self.transforms:
            for domain, role, transform in zip(self.domains, self.roles, self.transforms):
                self.datasets.append(
                    self._load_domain(
                        role=role,
                        root=str(os.path.join(self.root, domain)),
                        transforms=transform,
                    )
                )
        else:
            for domain, role in zip(self.domains, self.roles):
                self.datasets.append(
                    self._load_domain(
                        role=role,
                        root=str(os.path.join(self.root, domain)),
                        transforms=None,
                    )
                )

    def _make_splits(self):
        generator = torch.Generator().manual_seed(self.seed)
        for ds in self.datasets:
            splits = torch.utils.data.random_split(ds, [self.split_ratio, 1.0 - self.split_ratio], generator=generator)
            self.splits.append(splits)

    @staticmethod
    def _load_domain(role: DomainRole, root: str, transforms: Optional[PreprocessingPipeline]) -> SingleDomainDataset:
        return SingleDomainDataset(
            role=role,
            root=root,
            transforms=transforms,
        )

    def set_role(self, idx: int, role: DomainRole):
        assert idx < len(self.datasets)
        self.roles[idx] = role
        self.datasets[idx].set_role(role)

    def set_all_targets(self):
        for idx in range(len(self.datasets)):
            self.set_role(idx, DomainRole.TARGET)
