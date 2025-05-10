import pytest

from src.datasets import DomainRole, SingleDomainDataset, MultiDomainDataset

@pytest.fixture
def dataset_pacs():
    return MultiDomainDataset(
        root="data/PACS/",
        domains=["art_painting","cartoon", "photo", "sketch"],
        roles=[DomainRole.SOURCE, DomainRole.SOURCE, DomainRole.SOURCE, DomainRole.TARGET],
        transforms=[None, None, None, None],
        seed=0,
        split_ratio=0.80,
    )

@pytest.fixture
def dataset_pacs_photo():
    return SingleDomainDataset(
        role=DomainRole.SOURCE,
        root="data/PACS/photo",
        transforms=None,
    )

def test_pacs(dataset_pacs_photo, dataset_pacs):
    assert len(dataset_pacs_photo) == 1670
    assert dataset_pacs_photo.class_to_idx == {
        'dog': 0,
        'elephant': 1,
        'giraffe': 2,
        'guitar': 3,
        'horse': 4,
        'house': 5,
        'person': 6
    }
    assert dataset_pacs.size() == [2048, 2344, 1670, 3929]
    assert dataset_pacs.domains == ['art_painting', 'cartoon', 'photo', 'sketch']


def test_pacs_dataloaders(dataset_pacs):
    steps_per_epoch, loaders = dataset_pacs.build_source_dataloaders(
        use_splits=False, batch_size=16, num_workers=0,
        shuffle=True, persistent_workers=None
    )
    assert steps_per_epoch == 245
    assert len(loaders) == 3

    zipped = zip(*loaders)
    _batch = next(zipped)

    for batch in _batch:
        assert tuple(batch[0].shape) == (16, )
        assert tuple(batch[1].shape) == (16, 3, 227, 227)
        assert tuple(batch[2].shape) == (16, )

def test_pacs_dataloaders_with_splits(dataset_pacs):
    steps_per_epoch, split_loaders = dataset_pacs.build_source_dataloaders(
        use_splits=True, batch_size=16, num_workers=0,
        shuffle=True, persistent_workers=None
    )
    assert steps_per_epoch == [196, 49]
    assert len(split_loaders) == 2

    for loaders in split_loaders:
        zipped = zip(*loaders)
        _batch = next(zipped)

        for batch in _batch:
            assert tuple(batch[0].shape) == (16, )
            assert tuple(batch[1].shape) == (16, 3, 227, 227)
            assert tuple(batch[2].shape) == (16, )

    dataset_pacs.refresh_splits(new_split_ratio=0.5)
    steps_per_epoch, _ = dataset_pacs.build_source_dataloaders(
        use_splits=True, batch_size=16, num_workers=0,
        shuffle=True, persistent_workers=None
    )
    assert steps_per_epoch == [122, 122]
