# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function
from torch.optim import AdamW
import numpy as np
import pandas as pd
import copy
import math
from ...utils import get_or_create_path
from ...log import get_module_logger
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LinearLR

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from torch.optim.lr_scheduler import StepLR
from colors import print_green

class TransformerModel(Model):
    def __init__(
        self,
        d_feat: int = 20,
        d_model: int = 64,
        batch_size: int = 8192,
        nhead: int = 2,
        num_layers: int = 2,
        dropout: float = 0,
        n_epochs=100,
        lr=0.0001,
        metric="loss",
        early_stop=5,
        loss="mse",
        optimizer="adam",
        reg=1e-3,
        n_jobs=10,
        GPU=0,
		use_amp=False,
        seed=None,
        accum_steps=2,
        **kwargs,
    ):
        self.accum_steps = accum_steps
        # set hyper-parameters.
        self.fitted = False
        self.use_amp = use_amp
        self.d_feat = d_feat
        self.d_model = d_model
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss.lower()
        self.n_jobs = n_jobs
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed
        self.logger = get_module_logger("TransformerModel")
        self.logger.info("Naive Transformer:" "\nbatch_size : {}" "\ndevice : {}".format(self.batch_size, self.device))
        if self.use_amp:
            from torch.cuda.amp import GradScaler
            self.scaler = torch.amp.GradScaler('cuda')
            
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.model = Transformer(d_feat, d_model, nhead, num_layers, dropout, self.device)
        from qlib.contrib.model.pytorch_transformer_ts import TransformerModel


        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        elif optimizer.lower() == "adamw":
            self.train_optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))
        
        p = sum(p.numel() for p in self.model.parameters())
        print_green(f"Total params: {p / 1e6:.1f} M")

        self.fitted = False
        self.model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred.float() - label.float()) ** 2
        # mse = torch.mean(loss).item()
        # print("mse: %.6f" % mse)
        return torch.mean(loss)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)

        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])
        elif self.loss == "huber":
            huber = nn.SmoothL1Loss(reduction="mean")
            return huber(pred[mask], label[mask])

        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        mask = torch.isfinite(label)

        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])

        raise ValueError("unknown metric `%s`" % self.metric)

    def train_epoch(self, data_loader):
        self.model.train()

        for it, data in enumerate(data_loader):
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)
            self.train_optimizer.zero_grad()
            if self.use_amp:
                from torch.cuda.amp import autocast
                with torch.amp.autocast('cuda'):
                    pred = self.model(feature.float())
                    loss = self.loss_fn(pred, label) / self.accum_steps
                self.scaler.scale(loss).backward()
                if (it + 1) % self.accum_steps == 0:
                    self.scaler.unscale_(self.train_optimizer)
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
                    self.scaler.step(self.train_optimizer)
                    self.scaler.update()
                    self.train_optimizer.zero_grad()
            else:
                pred = self.model(feature.float())
                loss = self.loss_fn(pred, label)
                loss.backward()
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
                self.train_optimizer.step()


    def test_epoch(self, data_loader):
        self.model.eval()

        scores = []
        losses = []

        for data in data_loader:
            feature = data[:, :, :-1].to(self.device)
            label = data[:, -1, -1].to(self.device)
            with torch.no_grad():
                # keep pred as tensor on device to compute loss/metric
                if self.use_amp:
                    from torch.cuda.amp import autocast
                    with torch.amp.autocast('cuda'):
                        pred_t = self.model(feature.float())
                else:
                    pred_t = self.model(feature.float())
                loss_t = self.loss_fn(pred_t, label)
                score_t = self.metric_fn(pred_t, label)
            losses.append(loss_t.item())
            scores.append(score_t.item())

        if not losses:
            return 0.0, 0.0
        return float(np.mean(losses)), float(np.mean(scores))

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        save_path=None,
    ):
        # load train/valid as DataFrame and fill NA
        df_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        df_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        # fallback fillna if no .config method
        df_train = df_train.fillna(method="ffill").fillna(method="bfill")
        df_valid = df_valid.fillna(method="ffill").fillna(method="bfill")

        # reshape flat DataFrame → [N, T, F+1] where last dim includes features+label
        arr_train = df_train.values.astype(np.float32)
        arr_valid = df_valid.values.astype(np.float32)
        per_step = self.d_feat + 1
        seq_len = arr_train.shape[1] // per_step
        if seq_len * per_step != arr_train.shape[1]:
            raise ValueError(f"Cannot reshape train data of width {arr_train.shape[1]} "
                             f"into steps × (d_feat+1)={per_step}")

        train_tensor = torch.from_numpy(arr_train).view(-1, seq_len, per_step)
        valid_tensor = torch.from_numpy(arr_valid).view(-1, seq_len, per_step)

        train_loader = DataLoader(
            train_tensor,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_jobs,
            drop_last=True,
        )
        valid_loader = DataLoader(
            valid_tensor,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_jobs,
            drop_last=False,  # keep the last smaller batch so valid_loader isn't empty
        )

        save_path = get_or_create_path(save_path)

        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # train
        self.logger.info("training...")
        self.fitted = True
        from torch.optim.lr_scheduler import OneCycleLR

        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.reg)
        # scheduler = OneCycleLR(
        #     optimizer,
        #     max_lr=self.lr,
        #     total_steps=self.n_epochs * len(train_loader),
        #     pct_start=0.1,
        #     anneal_strategy="cos",
        # )
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=10)
        cosine = CosineAnnealingLR(optimizer, T_max=self.n_epochs - 10)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[10])
        print(f"n_epochs = {self.n_epochs}")
        for step in range(self.n_epochs):
            self.logger.info("Epoch%d:", step)
            self.logger.info("training...")
            self.train_epoch(train_loader)
            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(train_loader) # train_loss 沒用到是正常的
            val_loss, val_score = self.test_epoch(valid_loader) # valid_loss 沒用到是正常的
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)
            optimizer.step()
            scheduler.step()    # 每50轮 lr *= 0.8
            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        if self.use_gpu:
            torch.cuda.empty_cache()

    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        df_test = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        if df_test.empty:
            return pd.Series([], dtype=float)
        df_test = df_test.ffill().bfill()
        arr_test = df_test.values.astype(np.float32)
        # reshape to [N, seq_len, d_feat+1]
        per_step = self.d_feat + 1
        seq_len = arr_test.shape[1] // per_step
        test_tensor = torch.from_numpy(arr_test).view(-1, seq_len, per_step)
        test_loader = DataLoader(
            test_tensor,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_jobs,
            drop_last=False,
        )

        self.model.eval()
        preds = []

        for data in test_loader:
            feature = data[:, :, 0:-1].to(self.device)

            with torch.no_grad():
                pred = self.model(feature.float()).detach().cpu().numpy()

            preds.append(pred)

        if not preds:
            raise ValueError("No predictions were made, check your test data.")
        arr = np.concatenate(preds)
        return pd.Series(arr, index=df_test.index)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # [T, N, F]
        return x + self.pe[: x.size(0), :]


class Transformer(nn.Module):
    def __init__(self, d_feat=6, d_model=8, nhead=4, num_layers=2, dropout=0.5, device=None):
        super(Transformer, self).__init__()
        self.feature_layer = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=True,
        )

        self.decoder_layer = nn.Linear(d_model, 1)
        self.device = device
        self.d_feat = d_feat

    def forward(self, src):
        # src [N, T, F] 保持 batch_first
        x = self.feature_layer(src)      # [N, T, d_model]
        x = self.pos_encoder(x)          # [N, T, d_model]
        out = self.transformer_encoder(x)  # [N, T, d_model]
        last = out[:, -1, :]             # 取最后一个时间步 [N, d_model]
        return self.decoder_layer(last).squeeze(-1)
  