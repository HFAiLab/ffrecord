import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import unittest
from hfai.pl import HFAIEnvironment

import ffrecord
from ffrecord.torch import DataLoader, Dataset
print(ffrecord.__version__, ffrecord)


class ToyDataset(Dataset):
    def __init__(self):
        self.data = [(np.random.rand(128), np.random.rand(1)) for i in range(50)]

    def __getitem__(self, indices):
        samples = []
        for index in indices:
            data, label = self.data[index]
            data, label = torch.Tensor(data), torch.Tensor(label)
            samples.append((data, label))
        return samples

    def __len__(self):
        return len(self.data)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(128, 256)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, 1)

    def forward(self, input):
        output = self.linear1(input)
        output = self.linear2(output)
        output = self.linear3(output)
        return output


class ToyNetModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ToyModel()
        self.criterion = nn.MSELoss()

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        data, labels = batch
        output = self(data)
        loss = self.criterion(output, labels)
        return loss

    def configure_optimizers(self, lr=1e-3):
        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=lr)
        return optimizer

    def train_dataloader(self, batch_size=1, num_workers=2):
        traindataloader = DataLoader(ToyDataset(), batch_size=batch_size, num_workers=num_workers, skippable=False)
        return traindataloader


class TestDDPHF(unittest.TestCase):

    def test_ddp_hf(self):
        trainer = pl.Trainer(
            max_epochs=3,
            accelerator="gpu",
            devices=8,
            strategy="ddp_hf",
            plugins=[HFAIEnvironment()]
        )

        model_module = ToyNetModule()
        trainer.fit(model_module)


if __name__ == '__main__':
    unittest.main()
