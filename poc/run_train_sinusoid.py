import torch
from typing import List, Tuple, Optional
import plotly.express as px
import random


@torch.no_grad()
def create_periodic_synthetic_data(n_categories: int, n_samples_per_category: int) \
        -> Tuple[torch.TensorType, List[torch.TensorType], List[torch.TensorType], List[torch.TensorType]]:
    ts = torch.linspace(0, 10, steps=n_samples_per_category, dtype=torch.float32) * torch.pi
    ground_truth = []
    data = []
    metadata = []
    for _ in range(n_categories):

        amplitude = 0.5 + torch.rand(1)  # U[0.5, 1.5]
        phase = 2 * (torch.rand(1) - 0.5) * torch.pi  # U[-Pi, Pi]
        period_multiplier = 0.5 + 1.5 * torch.rand(1)  # U[0.5, 2.0]
        noise_multiplier = 0.05 + 0.25 * torch.rand(1)  # U[0.05, 0.30]

        gt = torch.sin((ts + phase) * period_multiplier) * amplitude
        measurements = gt + noise_multiplier * torch.randn_like(gt)

        ground_truth.append(gt)
        data.append(measurements)
        metadata.append([amplitude, phase, period_multiplier, noise_multiplier])

    return ts, ground_truth, data, metadata


def visualize_periodic_category_data(
    ts: torch.TensorType,
    gt: torch.TensorType,
    data: torch.TensorType,
    metadata: List[torch.TensorType],
    predictions: Optional[torch.TensorType] = None,
) -> None:
    amplitude, phase, period_multiplier, noise_multiplier = [float(m.item()) for m in metadata]
    plot_data = {
        'Time': ts,
        'GroundTruth': gt,
        'Measurements': data
    }
    if predictions is not None:
        plot_data['Prediction'] = predictions

    fig = px.line(
        plot_data,
        x='Time',
        y=['GroundTruth', 'Measurements'],
        title=f'Amplitude={amplitude:.2f}, Phase={phase:.2f}, PeriodMultiplier={period_multiplier:.2f}, NoiseMultiplier={noise_multiplier:.2f}'
    )
    fig.update_layout(legend={'title': 'TimeSeries'})
    fig.show()


example_ts, example_ground_truth, example_data, example_metadata = create_periodic_synthetic_data(5, 100)
visualize_periodic_category_data(example_ts, example_ground_truth[0], example_data[0], example_metadata[0])


from nodetracker.node.odernn import LightningODERNN
from nodetracker.node.utils.training import LightningTrainConfig
from torch.utils.data import Dataset, DataLoader
from nodetracker.datasets.utils import OdeDataloaderCollateFunctional, split_trajectory_observed_unobserved
import pytorch_lightning as pl


class SinusoidDataset(Dataset):
    def __init__(self, n_categories: int, n_samples_per_category: int, n_steps):
        self._n_categories = n_categories
        self._n_samples_per_category = n_samples_per_category
        self._n_steps = n_steps
        self._ts, self._ground_truth, self._data, self._metadata = \
            create_periodic_synthetic_data(n_categories, n_samples_per_category)

    def __len__(self):
        return self._n_steps

    def __getitem__(self, i: int) -> Tuple[torch.TensorType, ...]:
        category_index = random.randrange(0, self._n_categories)
        start = random.randrange(0, self._n_samples_per_category - 3)
        middle = random.randrange(start + 1, self._n_samples_per_category - 2)
        end = random.randrange(middle + 1, self._n_samples_per_category)

        t_obs = self._ts[start:middle].view(-1, 1)
        t_unobs = self._ts[middle:end].view(-1, 1)
        x_obs = self._data[category_index][start:middle].view(-1, 1)
        # x_unobs = self._data[category_index][middle+1:end]
        gt_obs = self._ground_truth[category_index][start:middle].view(-1, 1)
        gt_unobs = self._ground_truth[category_index][middle:end].view(-1, 1)

        assert t_obs.shape[0] > 0 and t_unobs.shape[0] > 0, \
            f'Invalid sequence sample {start}-{middle}-{end}'

        return x_obs, gt_unobs, t_obs, t_unobs, gt_obs, {}


train_steps: int = 100
train_dataset = SinusoidDataset(n_categories=10, n_samples_per_category=100, n_steps=train_steps)
val_dataset = SinusoidDataset(n_categories=10, n_samples_per_category=100, n_steps=100)
train_loader = DataLoader(train_dataset, batch_size=1, num_workers=8, collate_fn=OdeDataloaderCollateFunctional())
val_loader = DataLoader(val_dataset, batch_size=1, num_workers=8, collate_fn=OdeDataloaderCollateFunctional())


trainer = pl.Trainer(
    devices=[0],
    accelerator='cuda',
    max_epochs=10
)

model = LightningODERNN(
    observable_dim=1,
    hidden_dim=8,
    train_config=LightningTrainConfig(
        sched_lr_step=5
    ),
    log_epoch_metrics=False
)

trainer.fit(
    model=model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader
)

