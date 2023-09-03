"""
Utility functions for dataset testing.
"""
from nodetracker.datasets.torch.core import TrajectoryDataset, TorchTrajectoryDataset
from nodetracker.datasets import augmentations, transforms
from nodetracker.datasets.utils import OdeDataloaderCollateFunctional
from torch.utils.data import DataLoader


def run_dataset_test(dataset: TrajectoryDataset) -> None:
    print(f'Dataset size: {len(dataset)}')
    print(f'Sample example: {dataset[5]}')

    torch_dataset = TorchTrajectoryDataset(
        dataset=dataset,
        transform=transforms.BBoxStandardizedFirstOrderDifferenceTransform(
            mean=-8.65566333861711e-05,
            std=0.0009227107879355021
        )
    )

    print(f'Torch Dataset size: {len(torch_dataset)}')
    data = torch_dataset[5]
    data_shapes = {k: v.shape for k, v in data.items() if k != 'metadata'}
    print(f'Torch sample example shapes: {data_shapes}')
    print(f'Torch sample example: {torch_dataset[5]}')

    torch_dataloader = DataLoader(torch_dataset, batch_size=4, collate_fn=OdeDataloaderCollateFunctional())
    for bboxes_obs, bboxes_unobs, ts_obs, ts_unobs, _, metadata in torch_dataloader:
        print(f'Torch batch sample example shapes: bboxes_obs={bboxes_obs.shape}, bboxes_unobs={bboxes_unobs.shape}, '
              f'ts_obs={ts_obs.shape}, ts_unobs={ts_unobs.shape}')
        print('Torch batch metadata', metadata)

        break

    torch_dataset_with_noise = TorchTrajectoryDataset(
        dataset=dataset,
        transform=transforms.BBoxStandardizedFirstOrderDifferenceTransform(
            mean=-8.65566333861711e-05,
            std=0.0009227107879355021
        ),
        augmentation_before_transform=augmentations.DetectorNoiseAugmentation(sigma=0.05, proba=1.0)
    )

    print(f'Torch sample with noise example: {torch_dataset_with_noise[5]}')
