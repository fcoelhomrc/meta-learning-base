import copy
import itertools
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchmetrics

def compute_metrics(x, y, num_classes):
    probs = F.softmax(x, dim=1)
    labels = torch.argmax(probs, dim=1)
    acc = torchmetrics.functional.accuracy(labels, y, task="multiclass", num_classes=num_classes)
    return acc

def prep_resnet50():
    from torchvision.models import resnet50, ResNet50_Weights
    encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
    encoder.fc = nn.Identity()
    return encoder

def prep_classifier(num_feats: int, num_classes: int, dropout: float = 0., nonlinear=False):
    if nonlinear:
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_feats, num_feats // 2),
            nn.ReLU(),
            nn.Linear(num_feats // 2, num_feats // 4),
            nn.ReLU(),
            nn.Linear(num_feats // 4, num_classes),
        )
    else:
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_feats, num_classes),
        )


class Encoder(nn.Module):

    def __init__(self, hparams):
        super(Encoder, self).__init__()
        self.encoder = prep_resnet50()

    def forward(self, x):
        return self.encoder(x)


class Classifier(nn.Module):

    def __init__(self, hparams):
        super(Classifier, self).__init__()
        self.classifier = prep_classifier(
            num_feats=hparams.get("num_feats"),
            num_classes=hparams.get("num_classes"),
            dropout=hparams.get("dropout")
        )

    def forward(self, x):
        return self.classifier(x)


class BaseLearner(nn.Module):

    def __init__(self, encoder: Encoder, classifier: Classifier):
        super(BaseLearner, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        x = self.encoder(x)
        return self.classifier(x)


class MLDG(nn.Module):
    """
    Model-Agnostic Meta-Learning
    Algorithm 1 / Equation (3) from: https://arxiv.org/pdf/1710.03463.pdf
    Related: https://arxiv.org/pdf/1703.03400.pdf
    Related: https://arxiv.org/pdf/1910.13580.pdf
    """

    def __init__(self, network: BaseLearner, device: str, num_classes: int, num_domains: int, hparams: dict):
        super(MLDG, self).__init__()
        self.network = network
        self.device = device
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.hparams = hparams

        self.num_meta_test = hparams.get("num_meta_test")
        self.optimizer = self._setup_optimizer()

        self.to(self.device)

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams.get("lr"),
            weight_decay=self.hparams.get("weight_decay"),
        )

    def _split_meta_train_test(self, batch):
        # x: tuple with size (domains, ) of batches (batch_size, ...)
        # num_test: number of meta-test domains
        assert len(batch) == self.num_domains
        meta_train_test = torch.randperm(self.num_domains)
        meta_test = meta_train_test[:self.num_meta_test]
        meta_train = meta_train_test[self.num_meta_test:]
        if len(meta_train) < len(meta_test):
            for i, j in zip(itertools.cycle(meta_train), meta_test):
                yield batch[i], batch[j]
        else:
            for i, j in zip(meta_train, itertools.cycle(meta_test)):
                yield batch[i], batch[j]

    def _update_clone_network(self, x, y) -> Tuple[nn.Module, float]:
        clone_network = copy.deepcopy(self.network)
        optimizer = torch.optim.Adam(
            clone_network.parameters(),
            lr=self.hparams.get("lr_clone"),
            weight_decay=self.hparams.get("weight_decay_clone"),
        )
        loss = F.cross_entropy(clone_network(x), y)
        optimizer.zero_grad(set_to_none=False)  # make sure all grads init to zero
        loss.backward()  # compute grads (accumulated to clone network)
        optimizer.step()  # update params
        return clone_network, loss.item()

    def _accumulate_grads(self, source_network, scaling_factor):
        for p_tgt, p_src in zip(self.network.parameters(), source_network.parameters()):
            if p_src.grad is not None:
                p_tgt.grad.data.add_(
                    scaling_factor * p_src.grad.data
                )

    @staticmethod
    def _reset_grads(network):
        for p in network.parameters():
            p.grad = torch.zeros_like(p)

    def update(self, batch) -> float:
        """
        First-order impl of MLDG algorithm.
        * Clone network with params P
        * Compute grads Gi with meta-train data (xi, yi) using P
        * Update clone network params P -> P' with grads Gi
        * Compute grads Gj with meta-test data (xj, yj) using P'
        * Update original network params P -> P'' with accumulated grads Gi + beta * Gj
        """
        total_loss = 0
        self.optimizer.zero_grad() # set grads to None
        self._reset_grads(self.network)  # init parameter grads to zero
        for (_, xi, yi), (_, xj, yj) in self._split_meta_train_test(batch):
            xi, yi, xj, yj = xi.to(self.device), yi.to(self.device), xj.to(self.device), yj.to(self.device)
            clone_network, clone_loss = self._update_clone_network(xi, yi)
            self._accumulate_grads(clone_network, scaling_factor=1 / self.num_domains)
            self._reset_grads(clone_network)
            loss = F.cross_entropy(clone_network(xj), yj)
            loss.backward()  # compute grads
            self._accumulate_grads(clone_network, scaling_factor=self.hparams.get("beta") / self.num_domains)
            total_loss += (clone_loss + self.hparams.get("beta") * loss.item())
        self.optimizer.step()  # update original network params with accumulated grads
        return total_loss / self.num_domains

    @torch.inference_mode()
    def log_metrics(self, batch):
        acc = 0
        count = 0
        for (_, x, y) in batch:
            x, y = x.to(self.device), y.to(self.device)
            acc += compute_metrics(self.network(x), y, num_classes=self.num_classes)
            count += 1
        acc /= count
        return acc
