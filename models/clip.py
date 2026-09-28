import numpy as np
import torch
import transformers
from torch import nn

from models.t2mr_encoder import T2MRMotionEncoder


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x += projected
        return self.layer_norm(x)


class TextEncoder(nn.Module):
    def __init__(self, model_name_or_path: str, trainable: bool = True) -> None:
        super().__init__()
        self.text_model = transformers.AutoModel.from_pretrained(model_name_or_path)
        for param in self.text_model.parameters():
            param.requires_grad = trainable

        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state

        return last_hidden_state[:, self.target_token_idx, :]


class ClipModel(nn.Module):
    def __init__(
            self,
            motion_encoder_alias="t2mr",
            text_encoder_alias="roberta-large",
            motion_encoder_pretrained: bool = True,
            motion_encoder_trainable: bool = True,
            text_encoder_trainable: bool = True,
            motion_embedding_dims: int = 384,
            text_embedding_dims: int = 1024,
            projection_dims: int = 256,
            dropout: float = 0.5,
            logit: float = 0.07,
            pretrained_motion_path=None,
    ):
        super().__init__()

        # Motion encoder for the NTU 25-joint skeleton
        self.motion_encoder = T2MRMotionEncoder(
            num_point=25,
            embed_dim=motion_embedding_dims,
            graph='graph.smpl_graph.Graph',
        )

        # Load the pre-trained motion encoder weights
        if motion_encoder_pretrained and pretrained_motion_path is not None:
            print("\nLoading pre-trained motion encoder weights...")
            matched, skipped = self.motion_encoder.load_pretrained(
                pretrained_motion_path,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                strict=False,
            )
            print(f"Pretrained loading: {matched} matched, {skipped} skipped\n")

        if not motion_encoder_trainable:
            for p in self.motion_encoder.parameters():
                p.requires_grad = False

        self.text_encoder = TextEncoder(text_encoder_alias, trainable=text_encoder_trainable)

        self.motion_projection = ProjectionHead(motion_embedding_dims, projection_dims, dropout)
        self.text_projection = ProjectionHead(text_embedding_dims, projection_dims, dropout)

        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / logit)))
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def encode_motion(self, motion):
        """
        Input:  motion (B, T, V, C) = (B, seqlen, 25, 3)
        Output: motion embeddings (B, projection_dims)
        """
        motion_features = self.motion_encoder(motion)
        motion_embeddings = self.motion_projection(motion_features)
        return motion_embeddings

    def encode_text(self, text):
        text_features = self.text_encoder(
            input_ids=text["input_ids"], attention_mask=text["attention_mask"]
        )
        text_embeddings = self.text_projection(text_features)
        return text_embeddings

    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(
            logits, torch.arange(len(logits), device=logits.device)
        )

    def clip_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        motion_loss = self.contrastive_loss(similarity.t())
        return (caption_loss + motion_loss) / 2.0

    def forward(self, motion, text, return_loss=False):
        motion_embeds = self.encode_motion(motion)
        text_embeds = self.encode_text(text)

        motion_embeds = motion_embeds / motion_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, motion_embeds.t()) * logit_scale
        logits_per_motion = logits_per_text.T

        if return_loss:
            return self.clip_loss(logits_per_text)
        else:
            return motion_embeds, text_embeds
