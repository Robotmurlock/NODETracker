import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import plotly.graph_objects as go
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader

from nodetracker.node.utils.training import LightningTrainConfig
from nodetracker.node.odernn import LightningODERNN, LightningRNNODE
from nodetracker.standard.rnn.seq_to_seq import LightningRNNSeq2Seq

torch.manual_seed(42)


@torch.no_grad()
def create_periodic_synthetic_data(n_categories: int, n_samples_per_category: int) \
        -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    ts = torch.linspace(0, 10, steps=n_samples_per_category, dtype=torch.float32) * torch.pi
    ground_truth = []
    data = []
    metadata = []
    for _ in range(n_categories):

        amplitude = torch.tensor([1])  # 0.5 + torch.rand(1)  # U[0.5, 1.5]
        phase = torch.zeros(1)  # 2 * (torch.rand(1) - 0.5) * torch.pi  # U[-Pi, Pi]
        period_multiplier = torch.tensor([1])  # 0.5 + 1.5 * torch.rand(1)  # U[0.5, 2.0]
        noise_multiplier = 0.05 + 0.25 * torch.rand(1)  # U[0.05, 0.30]

        gt = torch.sin((ts + phase) * period_multiplier) * amplitude
        measurements = gt # + noise_multiplier * torch.randn_like(gt)
        positional_embedding = torch.sin(ts)
        features = torch.stack([measurements, positional_embedding], dim=-1)

        ground_truth.append(gt)
        data.append(features)
        metadata.append([amplitude, phase, period_multiplier, noise_multiplier])

    return ts, ground_truth, data, metadata


@torch.no_grad()
def create_constant_data(n_categories: int, n_samples_per_category: int) \
        -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    ts = torch.linspace(0, 10, steps=n_samples_per_category, dtype=torch.float32) * torch.pi
    ground_truth = []
    data = []
    metadata = []
    for _ in range(n_categories):

        gt = torch.ones_like(ts) * (torch.rand(1) - 0.5)
        measurements = gt # + noise_multiplier * torch.randn_like(gt)
        positional_embedding = torch.sin(ts)
        features = torch.stack([measurements, positional_embedding], dim=-1)

        ground_truth.append(gt)
        data.append(features)
        metadata.append([1])

    return ts, ground_truth, data, metadata


def visualize_periodic_category_data(
    ts: torch.Tensor,
    gt: torch.Tensor,
    data: torch.Tensor,
    predictions: Optional[torch.Tensor] = None,
    predictions_ts: Optional[torch.Tensor] = None
) -> None:

    fig = go.Figure(
        layout=go.Layout(
            title=go.layout.Title(text='Test')
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ts,
            y=gt,
            mode='lines',
            name='GroundTruth'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ts,
            y=data,
            mode='lines',
            name='Measurements'
        )
    )

    if predictions is not None and predictions_ts is not None:
        fig.add_trace(
            go.Scatter(
                x=predictions_ts,
                y=predictions,
                mode='lines',
                name='Prediction'
            )
        )

    fig.update_layout(legend={'title': 'TimeSeries'})
    return fig


images_path = 'sin_images'
Path(images_path).mkdir(parents=True, exist_ok=True)
example_ts, example_ground_truth, example_data, example_metadata = create_periodic_synthetic_data(5, 100)
fig = visualize_periodic_category_data(example_ts, example_ground_truth[0], example_data[0], example_metadata[0])
example_image_path = os.path.join(images_path, 'example.png')
fig.write_image(example_image_path)


class SinusoidDataset(Dataset):
    def __init__(self, n_categories: int, n_samples_per_category: int, n_steps, test: bool = False):
        self._n_categories = n_categories
        self._n_steps = n_steps
        self._ts, self._ground_truth, self._data, self._metadata = \
            create_periodic_synthetic_data(n_categories, n_samples_per_category)
        self._last_category_index = (self._n_categories - 1)
        self._test = test

    def __len__(self):
        return self._n_steps

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, ...]:
        if not self._test:
            category_index = random.randrange(0, self._n_categories)
        else:
            category_index = (self._last_category_index + 1) % self._n_categories
            self._last_category_index = category_index

        x = self._data[category_index]
        gt = self._ground_truth[category_index].view(-1, 1)
        ts = self._ts.view(-1, 1)
        return x, gt, ts


class OdeSinusoidDataloaderCollateFunctional:
    def __init__(self, n_samples_per_category: int, test: bool = False):
        self._n_samples_per_category = n_samples_per_category
        self._test = test

    def __call__(self, items: List[torch.Tensor]):
        x, gt, ts = zip(*items)
        x, gt, ts = [torch.stack(v, dim=1) for v in [x, gt, ts]]

        if not self._test:
            assert False
            start = random.randrange(0, self._n_samples_per_category - 3)
            middle = random.randrange(start + 1, self._n_samples_per_category - 2)
            end = random.randrange(middle + 1, self._n_samples_per_category)
        else:
            start = 0
            middle = x.shape[0] // 2
            end = x.shape[0]

        x_obs, x_unobs = x[start:middle], x[middle:end]
        t_obs, t_unobs = ts[start:middle], ts[middle:end]
        gt_obs, gt_unobs = gt[start:middle], gt[middle:end]

        return x_obs, gt_unobs, t_obs, t_unobs, gt_obs, gt_unobs


train_steps: int = 1280
val_steps: int = 320
test_steps: int = 10
n_samples_per_category: int = 100
n_categories: int = 10
n_test_categories: int = 10
epochs: int = 20
sched_step: int = 7

train_dataset = SinusoidDataset(n_categories=n_categories, n_samples_per_category=n_samples_per_category, n_steps=train_steps, test=True)
val_dataset = SinusoidDataset(n_categories=n_categories, n_samples_per_category=n_samples_per_category, n_steps=val_steps, test=True)
test_dataset = SinusoidDataset(n_categories=n_test_categories, n_samples_per_category=n_samples_per_category, n_steps=n_categories, test=True)
train_collate_func = OdeSinusoidDataloaderCollateFunctional(n_samples_per_category, test=True)
test_collate_func = OdeSinusoidDataloaderCollateFunctional(n_samples_per_category, test=True)

train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4, collate_fn=train_collate_func)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4, collate_fn=train_collate_func)
test_loader = DataLoader(test_dataset, batch_size=1, num_workers=1, collate_fn=test_collate_func)

USE_CUDA = True


pl_accelerator = 'cuda' if USE_CUDA else 'cpu'
pl_devices = [0] if USE_CUDA else 4
trainer = pl.Trainer(
    devices=pl_devices,
    accelerator=pl_accelerator,
    max_epochs=epochs,
    log_every_n_steps=1
)

# model = LightningODERNN(
#     observable_dim=2,
#     hidden_dim=32,
#     output_dim=1,
#     train_config=LightningTrainConfig(
#         sched_lr_step=sched_step,
#         sched_lr_gamma=0.1
#     ),
#     log_epoch_metrics=False,
#     decoder_global_state=False,
#     decoder_solver_name='euler',
#     decoder_solver_params={
#         'max_step_size': 0.25
#     }
# )

# model = LightningRNNSeq2Seq(
#     observable_dim=2,
#     hidden_dim=32,
#     latent_dim=32,
#     output_dim=1,
#     train_config=LightningTrainConfig(
#         sched_lr_step=sched_step,
#         sched_lr_gamma=0.1
#     ),
#     log_epoch_metrics=False
# )

model = LightningRNNODE(
    observable_dim=2,
    hidden_dim=32,
    output_dim=1,
    train_config=LightningTrainConfig(
        sched_lr_step=4,
        sched_lr_gamma=0.1
    ),
    log_epoch_metrics=False,
    decoder_global_state=False,
    decoder_solver_name='rk4',
    decoder_solver_params={
        'max_step_size': 0.01
    }
)

accelerator = 'cuda:0' if USE_CUDA else 'cpu'

model.train()
model.to(accelerator)
trainer.fit(
    model=model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader
)


model.to(accelerator)
model.eval()
with torch.no_grad():
    for i, (x_obs, x_unobs, t_obs, t_unobs, gt_obs, gt_unobs) in enumerate(test_loader):
        x_obs, x_unobs, t_obs, t_unobs, gt_obs, gt_unobs = [v.to('cuda:0') for v in [x_obs, x_unobs, t_obs, t_unobs, gt_obs, gt_unobs]]

        x_hat_unobs, *_ = model.inference(x_obs, t_obs, t_unobs)
        x = torch.cat([x_obs[:, 0, 0], x_unobs[:, 0, 0]], dim=-1).detach().cpu()
        gt = torch.cat([gt_obs[:, 0, 0], gt_unobs[:, 0, 0]], dim=-1).detach().cpu()
        ts = torch.cat([t_obs[:, 0, 0], t_unobs[:, 0, 0]], dim=-1).detach().cpu()
        x_hat_unobs, t_unobs = x_hat_unobs[:, 0, 0].detach().cpu(), t_unobs[:, 0, 0].detach().cpu()


        metadata = torch.tensor(4 * [.0])
        fig = visualize_periodic_category_data(ts, gt, x, predictions=x_hat_unobs, predictions_ts=t_unobs)

        image_path = os.path.join(images_path, f'inf_{i:04d}.png')
        fig.write_image(image_path)
