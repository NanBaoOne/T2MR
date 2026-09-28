"""
Copyright 2024 LY Corporation
LY Corporation licenses this file to you under the CC BY-NC 4.0
(the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at:
    https://creativecommons.org/licenses/by-nc/4.0/
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.
"""
import itertools
import logging
import os
import random
import sys
from os.path import join as pjoin

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.getcwd())
from datasets import TextMotionPatchDataset
from models.clip import ClipModel
from scripts.test import eval, prepare_test_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "true"

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_name="config", config_path="../conf")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    os.makedirs(cfg.checkpoints_dir, exist_ok=True)
    set_seed(cfg.train.seed)
    train_dataloader, test_dataloader = prepare_dataset(cfg)
    eval_dataloader = prepare_test_dataset(cfg)
    model, optimizer, scheduler, tokenizer = prepare_model(cfg, train_dataloader)
    train(
        cfg,
        train_dataloader,
        test_dataloader,
        eval_dataloader,
        model,
        tokenizer,
        optimizer,
        scheduler,
    )


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)


def prepare_dataset(cfg):
    mean = np.load(pjoin(cfg.dataset.data_root, "Mean_ntu.npy"))
    std = np.load(pjoin(cfg.dataset.data_root, "Std_ntu.npy"))

    train_split_file = pjoin(cfg.dataset.data_root, "train.txt")
    train_dataset = TextMotionPatchDataset(
        cfg, mean, std, train_split_file, fps=True,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.train.num_workers,
    )

    val_split_file = pjoin(cfg.dataset.data_root, "val.txt")
    val_dataset = TextMotionPatchDataset(
        cfg, mean, std, val_split_file, fps=True,
    )
    test_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )

    return train_dataloader, test_dataloader


def prepare_model(cfg, train_dataloader):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_encoder)

    model = ClipModel(
        cfg.model.motion_encoder,
        cfg.model.text_encoder,
        cfg.train.motion_encoder_pretrained,
        cfg.train.train_motion_encoder,
        cfg.train.train_text_encoder,
        cfg.model.motion_embedding_dims,
        cfg.model.text_embedding_dims,
        cfg.model.projection_dims,
        dropout=0.5 if cfg.dataset.dataset_name == "HumanML3D" else 0.0,
        pretrained_motion_path=cfg.model.pretrained_motion_path,
    )
    model.to(device)

    parameters = [
        {
            "params": model.motion_encoder.parameters(),
            "lr": cfg.train.optimizer.motion_lr * cfg.dataset.motion_lr_factor,
        },
        {
            "params": model.text_encoder.parameters(),
            "lr": cfg.train.optimizer.text_lr * cfg.dataset.text_lr_factor,
        },
        {
            "params": itertools.chain(
                model.motion_projection.parameters(),
                model.text_projection.parameters(),
            ),
            "lr": cfg.train.optimizer.head_lr * cfg.dataset.head_lr_factor,
        },
    ]
    optimizer = optim.Adam(parameters)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, len(train_dataloader) * cfg.train.epoch * 2
    )
    return model, optimizer, scheduler, tokenizer


def train(
    cfg,
    train_dataloader,
    test_dataloader,
    eval_dataloader,
    model,
    tokenizer,
    optimizer,
    scheduler,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    best_te_loss = 1e5
    best_t2m_r1 = 0
    best_m2t_r1 = 0
    best_ep = -1
    for epoch in range(cfg.train.epoch):
        print(
            f"running epoch {epoch}, best test loss {best_te_loss} "
            f"best_t2m_r1 {best_t2m_r1} best_m2t_r1 {best_m2t_r1} after epoch {best_ep}"
        )
        step = 0
        tr_loss = 0
        logit_scales = []
        model.train()
        pbar = tqdm(train_dataloader, leave=False)
        for batch in pbar:
            step += 1
            optimizer.zero_grad()

            texts, motions, _, _ = batch
            motions = motions.to(device)

            texts = tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt"
            ).to(device)

            total_loss = model(motions, texts, return_loss=True)
            total_loss.backward()
            tr_loss += total_loss.item()
            optimizer.step()
            scheduler.step()

            scale = model.logit_scale.exp().detach().cpu().item()
            tau = 1.0 / scale
            logit_scales.append(scale)
            pbar.set_description(
                f"train CE: {total_loss.item():.4f} | scale: {scale:.2f} | tau: {tau:.4f}",
                refresh=True,
            )
        tr_loss /= step
        avg_scale = float(np.mean(logit_scales)) if logit_scales else 0.0
        avg_tau = 1.0 / avg_scale if avg_scale > 0 else 0.0

        step = 0
        te_loss = 0
        with torch.no_grad():
            model.eval()
            test_pbar = tqdm(test_dataloader, leave=False)
            for batch in test_pbar:
                step += 1
                texts, motions, _, _ = batch
                motions = motions.to(device)
                texts = tokenizer(
                    texts, padding=True, truncation=True, return_tensors="pt"
                ).to(device)

                total_loss = model(motions, texts, return_loss=True)

                te_loss += total_loss.item()
                test_pbar.set_description(
                    f"test batchCE: {total_loss.item()}", refresh=True
                )
            te_loss /= step

        if te_loss < best_te_loss:
            best_te_loss = te_loss

        torch.save(model.state_dict(), pjoin(cfg.checkpoints_dir, "last_model.pt"))

        t2m_r1, m2t_r1 = eval(
            cfg, eval_dataloader, model, tokenizer=tokenizer, verbose=False
        )

        log.info(
            f"epoch {epoch}, tr_loss {tr_loss:.4f}, te_loss {te_loss:.4f}, "
            f"t2m_r1 {t2m_r1:.4f}, m2t_r1 {m2t_r1:.4f}, "
            f"logit_scale(exp) avg {avg_scale:.2f}, tau avg {avg_tau:.4f}"
        )

        best_t2m_r1 = max(best_t2m_r1, t2m_r1)
        best_m2t_r1 = max(best_m2t_r1, m2t_r1)

        if best_t2m_r1 == t2m_r1:
            best_ep = epoch
            torch.save(model.state_dict(), pjoin(cfg.checkpoints_dir, "best_model.pt"))


if __name__ == "__main__":
    main()
