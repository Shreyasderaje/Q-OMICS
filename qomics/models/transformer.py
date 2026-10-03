"""Multi-omics transformer encoder with an attention-based biomarker readout.

Design rationale
----------------
A biomarker panel is not a set of independent scores: PTEN loss drives PI3K
signalling, which is visible in the phosphoproteome, which feeds back on
transcription.  A model that scores each gene separately cannot represent
those couplings.  Q-OMICS therefore treats the sample as a *sequence of
molecular tokens* - one token per gene, antibody or genomic event - and lets
self-attention learn which markers co-vary across layers.

Concretely:

1. Every feature is a scalar lifted to ``d_model`` by a per-modality linear
   map, so a token carries its value plus a modality identity.
2. A learnable ``[CLS]`` token aggregates the sequence, and an optional
   *fusion token* carries the quantum-kernel decision score, which is how the
   quantum and classical branches actually interact.
3. Self-attention is read back through **attention rollout** (Abnar &
   Zuidema, 2020), the standard way to attribute a transformer's decision
   to individual input tokens, giving a per-feature importance that is
   ranked into a named biomarker panel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# Encoder block
# --------------------------------------------------------------------------


@dataclass
class AttentionOutput:
    """Attention matrices collected across layers for attribution."""

    matrices: list[torch.Tensor]
    token_modality: list[str]
    token_names: list[str]
    cls_index: int

    def rollout(self) -> torch.Tensor:
        """Attention rollout: cumulative attention from ``[CLS]`` to tokens.

        Naive per-layer attention ignores the residual stream, so a token's
        effective influence is attenuated at every layer and vanishes by
        layer three.  Rollout fixes this by adding the identity to each
        attention map, normalising rows, and composing across depth.
        """
        if not self.matrices:
            return torch.zeros(1)
        result = None
        for attn in self.matrices:
            a = attn.mean(dim=1)  # average over heads -> (batch, tokens, tokens)
            a = a + torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
            a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            result = a if result is None else torch.bmm(a, result)
        assert result is not None
        return result[:, self.cls_index, :]


class EncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer that exposes its attention map."""

    def __init__(self, d_model: int, n_heads: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, need_weights: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        h = self.norm1(x)
        attn_out, weights = self.attn(
            h, h, h, need_weights=need_weights, average_attn_weights=False
        )
        x = x + self.drop(attn_out)
        x = x + self.ff(self.norm2(x))
        return x, (weights if need_weights else None)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


class MultiOmicsTransformer(nn.Module):
    """Hybrid quantum-classical encoder for stage-stratified classification.

    Parameters
    ----------
    block_sizes:
        Number of tokens contributed by each modality.
    n_fusion:
        Width of the vector produced by the quantum branch.  When greater
        than zero a fusion token is appended to the sequence.
    """

    def __init__(
        self,
        block_sizes: dict[str, int],
        *,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
        n_classes: int = 2,
        n_fusion: int = 0,
        max_tokens: int = 640,
    ) -> None:
        super().__init__()
        self.block_sizes = dict(block_sizes)
        self.block_names = list(block_sizes)
        self.n_fusion = int(n_fusion)
        self.d_model = d_model

        total = sum(block_sizes.values()) + 1 + self.n_fusion
        if total > max_tokens:
            log_msg = (
                f"sequence of {total} tokens exceeds max_tokens={max_tokens}; "
                "reduce max_features_per_block in the preprocessor"
            )
            raise ValueError(log_msg)

        # Scalar feature -> token embedding, one map per modality so that the
        # model can learn a different lift for expression, dosage and
        # abundance rather than sharing one projection.
        self.lifts = nn.ModuleDict(
            {name: nn.Linear(1, d_model) for name in self.block_names}
        )
        self.modality_embedding = nn.Parameter(torch.randn(len(self.block_names), d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.type_embedding = nn.Parameter(torch.randn(3, d_model) * 0.02)  # gene / cls / fusion

        if self.n_fusion:
            self.fusion_proj = nn.Sequential(
                nn.Linear(self.n_fusion, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )

        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, dim_feedforward, dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

        self._index_maps = self._build_index_maps()
        self.apply(self._init_weights)

    # -- token bookkeeping -------------------------------------------------

    def _build_index_maps(self) -> dict[str, torch.Tensor]:
        """Token index slice for each modality, for readout bookkeeping."""
        maps: dict[str, torch.Tensor] = {}
        cursor = 1  # token 0 is [CLS]
        for i, name in enumerate(self.block_names):
            size = self.block_sizes[name]
            maps[name] = torch.arange(cursor, cursor + size)
            cursor += size
        maps["__fusion__"] = torch.arange(cursor, cursor + self.n_fusion)
        return maps

    @property
    def n_tokens(self) -> int:
        return sum(self.block_sizes.values()) + 1 + self.n_fusion

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # -- forward -----------------------------------------------------------

    def forward(
        self,
        blocks: dict[str, torch.Tensor],
        fusion: torch.Tensor | None = None,
        *,
        need_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run the encoder.

        Parameters
        ----------
        blocks:
            ``{modality: (batch, n_tokens_of_modality)}`` float tensors.
        fusion:
            ``(batch, n_fusion)`` quantum-branch features, required when the
            model was constructed with ``n_fusion > 0``.
        need_attention:
            Also return per-layer attention maps for biomarker attribution.
        """
        batch = self._batch_size(blocks)
        tokens: list[torch.Tensor] = [self.cls_token.expand(batch, -1, -1)]
        types: list[torch.Tensor] = [self._type(0, batch, 1)]

        for i, name in enumerate(self.block_names):
            values = blocks[name].unsqueeze(-1)  # (B, T, 1)
            lifted = self.lifts[name](values)  # (B, T, d)
            lifted = lifted + self.modality_embedding[i]
            tokens.append(lifted)
            types.append(self._type(1, batch, lifted.shape[1]))

        if self.n_fusion:
            if fusion is None:
                raise ValueError("model expects a fusion vector but none was supplied")
            projected = self.fusion_proj(fusion).unsqueeze(1)  # (B, 1, d)
            tokens.append(projected)
            types.append(self._type(2, batch, 1))

        x = torch.cat(tokens, dim=1)
        x = x + torch.cat(types, dim=1)

        matrices: list[torch.Tensor] = []
        for layer in self.layers:
            x, weights = layer(x, need_weights=need_attention)
            if weights is not None:
                matrices.append(weights)

        x = self.final_norm(x)
        cls = x[:, 0]
        mean_pool = x.mean(dim=1)
        logits = self.head(torch.cat([cls, mean_pool], dim=1))

        out: dict[str, torch.Tensor] = {"logits": logits, "token_embeddings": x}
        if need_attention:
            out["attention"] = torch.stack(matrices, dim=1)  # (B, layers, heads, T, T)
        return out

    @staticmethod
    def _batch_size(blocks: dict[str, torch.Tensor]) -> int:
        return next(iter(blocks.values())).shape[0]

    def _type(self, index: int, batch: int, length: int) -> torch.Tensor:
        """Broadcast a token-type embedding to ``(batch, length, d_model)``.

        The parameter is indexed to a 1-D vector first; ``expand`` cannot add
        leading dimensions on its own, so the singleton axis is made explicit.
        """
        return self.type_embedding[index].view(1, 1, -1).expand(batch, length, -1)

    # -- attribution -------------------------------------------------------

    def token_importance(self, attention: torch.Tensor) -> torch.Tensor:
        """Per-token importance from attention rollout, shape ``(batch, T)``."""
        rollout = AttentionOutput(matrices=[attention[:, i] for i in range(attention.shape[1])],
                                  token_modality=[], token_names=[], cls_index=0).rollout()
        return rollout

    def block_importance(self, importance: torch.Tensor) -> dict[str, torch.Tensor]:
        """Aggregate token importance per modality block."""
        out: dict[str, torch.Tensor] = {}
        for name, idx in self._index_maps.items():
            if name == "__fusion__":
                continue
            if idx.numel() == 0:
                continue
            out[name] = importance[:, idx].mean(dim=1)
        return out


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Focal loss for the imbalanced stage-I / stage-II-IV split.

    Early-stage cases are the minority class and the clinically important
    one; plain cross-entropy collapses onto the majority class, and class
    weights alone make the rare class dominate the gradient.  Focal loss
    down-weights easy examples so the encoder is forced to spend capacity on
    the boundary that matters.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    if label_smoothing > 0:
        nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        smooth = -log_probs.mean(dim=-1)
        nll = (1.0 - label_smoothing) * nll + label_smoothing * smooth
    else:
        nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

    pt = torch.exp(-nll)
    alpha_t = torch.where(targets == 1, torch.full_like(pt, alpha), torch.full_like(pt, 1.0 - alpha))
    loss = alpha_t * ((1.0 - pt) ** gamma) * nll
    return loss.mean()


def count_weights(y: np.ndarray) -> torch.Tensor:
    """Inverse-frequency class weights, normalised to mean one."""
    y = np.asarray(y).ravel()
    n_classes = int(y.max()) + 1
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts = np.clip(counts, 1.0, None)
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "MultiOmicsTransformer",
    "EncoderLayer",
    "AttentionOutput",
    "focal_loss",
    "count_weights",
    "parameter_count",
]
