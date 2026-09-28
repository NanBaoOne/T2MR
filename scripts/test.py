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

import logging
import os
import sys
from os.path import join as pjoin

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.getcwd())
from datasets import TextMotionPatchDataset
from models.clip import ClipModel

log = logging.getLogger(__name__)


def get_csls_sim(sims, k=10):
    """
    Cross-Domain Similarity Local Scaling (CSLS).

    Subtracts the mean similarity of the k nearest neighbours of both the
    query and the gallery side, which penalizes hub samples.

    Args:
        sims: (N_query, N_gallery) similarity matrix
        k: number of neighbours
    """
    topk_sims_q, _ = torch.topk(torch.tensor(sims), k=k, dim=1)
    mean_sim_q = torch.mean(topk_sims_q, dim=1)

    topk_sims_g, _ = torch.topk(torch.tensor(sims).t(), k=k, dim=1)
    mean_sim_g = torch.mean(topk_sims_g, dim=1)

    csls_sims = 2 * sims - mean_sim_q[:, None].numpy() - mean_sim_g[None, :].numpy()
    return csls_sims


class FastReRanking:
    """
    Fast Re-ranking for cross-modal retrieval.
    Based on "Dynamic Visual Semantic Sub-Embeddings and Fast Re-Ranking".

    Applies a bidirectional softmax normalization to the similarity matrix,
    using the reversed retrieval direction as a prior. No nearest-neighbour
    search is required.
    """

    def __init__(self, gamma1=10.0, gamma2=10.0, lambda1=10.0, lambda2=10.0):
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def apply(self, similarity_matrix):
        """
        Args:
            similarity_matrix: (N_text, N_motion) similarity matrix

        Returns:
            i2t_sim: (N_motion, N_text) re-ranked motion-to-text similarities
            t2i_sim: (N_text, N_motion) re-ranked text-to-motion similarities
        """
        if not isinstance(similarity_matrix, torch.Tensor):
            similarity_matrix = torch.tensor(similarity_matrix)

        i2t_sim = self.softmax_rerank(similarity_matrix.T, self.gamma1, self.gamma2, dim=0)
        t2i_sim = self.softmax_rerank(similarity_matrix, self.lambda1, self.lambda2, dim=1)

        return i2t_sim, t2i_sim

    def softmax_rerank(self, A, scale1, scale2, dim):
        numerator = torch.exp(A * scale2)
        denominator = torch.sum(torch.exp(A * scale1), dim=dim, keepdim=True)
        denominator = torch.clamp(denominator, min=1e-8)
        return numerator / denominator


@hydra.main(version_base=None, config_name="test_config", config_path="../conf")
def main(cfg: DictConfig) -> None:
    saved_cfg = OmegaConf.load(pjoin(cfg.checkpoints_dir, ".hydra/config.yaml"))
    cfg = OmegaConf.merge(saved_cfg, {"eval": OmegaConf.to_container(cfg.eval, resolve=True)})
    print(OmegaConf.to_yaml(cfg))
    test_dataloader = prepare_test_dataset(cfg)
    model, tokenizer = prepare_test_model(cfg)
    eval(cfg, test_dataloader, model, tokenizer)


def prepare_test_dataset(cfg):
    mean = np.load(pjoin(cfg.dataset.data_root, "Mean_ntu.npy"))
    std = np.load(pjoin(cfg.dataset.data_root, "Std_ntu.npy"))

    if cfg.eval.eval_train:
        test_split_file = pjoin(cfg.dataset.data_root, "train.txt")
    else:
        test_split_file = pjoin(cfg.dataset.data_root, "test.txt")
    test_dataset = TextMotionPatchDataset(
        cfg, mean, std, test_split_file, eval_mode=True, fps=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.get("num_workers", 16),
    )
    return test_dataloader


def prepare_test_model(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_encoder)

    model = ClipModel(
        cfg.model.motion_encoder,
        cfg.model.text_encoder,
        False,
        False,
        False,
        cfg.model.motion_embedding_dims,
        cfg.model.text_embedding_dims,
        cfg.model.projection_dims,
        dropout=0.5 if cfg.dataset.dataset_name == "HumanML3D" else 0.0,
    )

    model_name = "best_model.pt" if cfg.eval.use_best_model else "last_model.pt"
    model_path = pjoin(cfg.checkpoints_dir, model_name)
    print(model_path)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 60)
    print("Model parameter statistics:")
    print(f"  total: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"  trainable: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    print("=" * 60)

    model.to(device)

    return model, tokenizer


def compute_ranks(sims, positives):
    ranks = np.zeros(sims.shape[0])
    for index, score in enumerate(tqdm(sims)):
        inds = np.argsort(score)[::-1]
        rank = 1e20
        for i in positives[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
    return ranks


def report_metrics(ranks, prefix, verbose=True):
    r1 = 0.0
    for k in [1, 2, 3, 5, 10]:
        r = 100.0 * len(np.where(ranks < k)[0]) / len(ranks)
        if k == 1:
            r1 = r
        if verbose:
            log.info(f"{prefix}_recall_top{k}_correct_composition: {r:.2f}")
    if verbose:
        log.info(f"{prefix}_recall_median_correct_composition: {np.median(ranks) + 1:.2f}")
    return r1


def eval(cfg, test_dataloader, model, tokenizer=None, verbose=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_motions_feat = []
    all_texts_feat = []
    all_captions = []

    with torch.no_grad():
        model.eval()
        test_pbar = tqdm(test_dataloader, leave=False)
        for batch in test_pbar:
            texts, motions, m_length, keys = batch
            motions = motions.to(device)

            texts_token = tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt"
            ).to(device)

            motion_features = model.encode_motion(motions)
            text_features = model.encode_text(texts_token)

            motion_features = motion_features / motion_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            all_motions_feat.append(motion_features.cpu().numpy())
            all_texts_feat.append(text_features.cpu().numpy())
            all_captions.extend(texts)

    all_motions_feat = np.vstack(all_motions_feat)
    all_texts_feat = np.vstack(all_texts_feat)
    all_captions = np.array(all_captions)

    # Following the HumanML3D evaluation protocol, every sample that shares the
    # same caption as the query counts as a valid match.
    caption_to_rows = dict()
    for row, caption in enumerate(all_captions):
        caption_to_rows.setdefault(caption, []).append(row)
    positives = [np.array(caption_to_rows[caption]) for caption in all_captions]

    raw_sims = all_texts_feat.dot(all_motions_feat.T)
    sims_t2m = get_csls_sim(raw_sims, k=10) * 100

    if cfg.eval.fast_reranking.enabled:
        fr = FastReRanking(
            gamma1=cfg.eval.fast_reranking.gamma1,
            gamma2=cfg.eval.fast_reranking.gamma2,
            lambda1=cfg.eval.fast_reranking.lambda1,
            lambda2=cfg.eval.fast_reranking.lambda2,
        )
        _, t2i_sim = fr.apply(torch.tensor(sims_t2m).T)
        sims_t2m = t2i_sim.numpy()

    # Text -> Motion
    ranks = compute_ranks(sims_t2m, positives)
    t2m_r1 = report_metrics(ranks, "t2m", verbose=verbose)

    # Motion -> Text
    ranks = compute_ranks(sims_t2m.T, positives)
    m2t_r1 = report_metrics(ranks, "m2t", verbose=verbose)

    return t2m_r1, m2t_r1


if __name__ == "__main__":
    main()
