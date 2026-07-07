from __future__ import annotations

import torch
import torch.nn as nn

FUSION_REGISTRY: dict[str, dict] = {}


def register_fusion(name: str, default_output_dim: int = 512):
    def decorator(cls):
        FUSION_REGISTRY[name] = {"class": cls, "output_dim": default_output_dim}
        return cls
    return decorator


class BaseFusionNetwork(nn.Module):
    output_dim: int = 512

    def __init__(self, cnn_dim: int = 512, vit_dim: int = 0, rnn_dim: int = 256, output_dim: int = 512):
        super().__init__()
        self.cnn_dim = cnn_dim
        self.vit_dim = vit_dim
        self.rnn_dim = rnn_dim
        self.output_dim = output_dim

    def forward(self, cnn: torch.Tensor, vit: torch.Tensor | None, rnn: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def _mlp_block(in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.3) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.LayerNorm(hidden_dim),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
        nn.ReLU(inplace=True),
        nn.LayerNorm(out_dim),
    )


@register_fusion("Standard FC", default_output_dim=512)
class StandardFCFusion(BaseFusionNetwork):
    def __init__(self, cnn_dim=512, vit_dim=0, rnn_dim=256, output_dim=512):
        super().__init__(cnn_dim, vit_dim, rnn_dim, output_dim)
        total = cnn_dim + vit_dim + rnn_dim
        self.net = nn.Sequential(
            nn.Linear(total, 1024),
            nn.ReLU(inplace=True),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim),
        )

    def forward(self, cnn, vit, rnn):
        features = [cnn, rnn]
        if vit is not None:
            features.insert(1, vit)
        return self.net(torch.cat(features, dim=1))


@register_fusion("MLP", default_output_dim=512)
class MLPFusion(BaseFusionNetwork):
    def __init__(self, cnn_dim=512, vit_dim=0, rnn_dim=256, output_dim=512):
        super().__init__(cnn_dim, vit_dim, rnn_dim, output_dim)
        total = cnn_dim + vit_dim + rnn_dim
        self.net = nn.Sequential(
            nn.Linear(total, 2048),
            nn.ReLU(inplace=True),
            nn.LayerNorm(2048),
            nn.Dropout(0.4),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.LayerNorm(1024),
            nn.Dropout(0.4),
            nn.Linear(1024, output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim),
        )

    def forward(self, cnn, vit, rnn):
        features = [cnn, rnn]
        if vit is not None:
            features.insert(1, vit)
        return self.net(torch.cat(features, dim=1))


@register_fusion("Deep Dense Fusion", default_output_dim=512)
class DeepDenseFusionNetwork(BaseFusionNetwork):
    def __init__(self, cnn_dim=512, vit_dim=0, rnn_dim=256, output_dim=512):
        super().__init__(cnn_dim, vit_dim, rnn_dim, output_dim)
        proj_dim = max(cnn_dim, rnn_dim, vit_dim)
        self.cnn_proj = nn.Linear(cnn_dim, proj_dim)
        self.rnn_proj = nn.Linear(rnn_dim, proj_dim)
        if vit_dim > 0:
            self.vit_proj = nn.Linear(vit_dim, proj_dim)
        num_modalities = 3 if vit_dim > 0 else 2
        self.pre_fusion = nn.Sequential(
            nn.Linear(proj_dim * num_modalities, 1024),
            nn.ReLU(inplace=True),
            nn.LayerNorm(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(proj_dim * num_modalities, num_modalities),
            nn.Softmax(dim=1),
        )
        self.out = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim),
        )

    def forward(self, cnn, vit, rnn):
        proj = [self.cnn_proj(cnn), self.rnn_proj(rnn)]
        if vit is not None:
            proj.insert(1, self.vit_proj(vit))
        concat = torch.cat(proj, dim=1)
        fused = self.pre_fusion(concat)
        weights = self.gate(concat)
        stacked = torch.stack(proj, dim=1)
        gated = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        return self.out(torch.cat([fused, gated], dim=1))


@register_fusion("Cross-Modal Attention", default_output_dim=512)
class CrossModalAttentionFusion(BaseFusionNetwork):
    def __init__(self, cnn_dim=512, vit_dim=0, rnn_dim=256, output_dim=512):
        super().__init__(cnn_dim, vit_dim, rnn_dim, output_dim)
        proj_dim = output_dim
        self.cnn_proj = nn.Linear(cnn_dim, proj_dim)
        self.rnn_proj = nn.Linear(rnn_dim, proj_dim)
        if vit_dim > 0:
            self.vit_proj = nn.Linear(vit_dim, proj_dim)
        self.num_modalities = 3 if vit_dim > 0 else 2
        self.cross_attn = nn.MultiheadAttention(proj_dim, num_heads=8, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(proj_dim)
        self.fusion = nn.Sequential(
            nn.Linear(proj_dim * self.num_modalities, proj_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(proj_dim),
            nn.Dropout(0.2),
        )

    def forward(self, cnn, vit, rnn):
        projected = [self.cnn_proj(cnn), self.rnn_proj(rnn)]
        if vit is not None:
            projected.insert(1, self.vit_proj(vit))
        tokens = torch.stack(projected, dim=1)
        attended, _ = self.cross_attn(tokens, tokens, tokens)
        attended = self.norm(tokens + attended)
        b, n, d = attended.shape
        flat = attended.reshape(b, n * d)
        return self.fusion(flat)


@register_fusion("Self-Attention Fusion", default_output_dim=512)
class SelfAttentionFusion(BaseFusionNetwork):
    def __init__(self, cnn_dim=512, vit_dim=0, rnn_dim=256, output_dim=512):
        super().__init__(cnn_dim, vit_dim, rnn_dim, output_dim)
        total = cnn_dim + vit_dim + rnn_dim
        self.proj = nn.Linear(total, output_dim)
        self.self_attn = nn.MultiheadAttention(output_dim, num_heads=4, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(output_dim)
        self.out = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim),
            nn.Dropout(0.2),
        )

    def forward(self, cnn, vit, rnn):
        features = [cnn, rnn]
        if vit is not None:
            features.insert(1, vit)
        concat = torch.cat(features, dim=1)
        x = self.proj(concat).unsqueeze(1)
        attended, _ = self.self_attn(x, x, x)
        x = self.norm(x + attended).squeeze(1)
        return self.out(x)


@register_fusion("Late Fusion", default_output_dim=512)
class LateFusion(BaseFusionNetwork):
    def __init__(self, cnn_dim=512, vit_dim=0, rnn_dim=256, output_dim=512):
        super().__init__(cnn_dim, vit_dim, rnn_dim, output_dim)
        self.cnn_branch = _mlp_block(cnn_dim, cnn_dim, output_dim)
        self.rnn_branch = _mlp_block(rnn_dim, rnn_dim, output_dim)
        if vit_dim > 0:
            self.vit_branch = _mlp_block(vit_dim, vit_dim, output_dim)
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * (3 if vit_dim > 0 else 2), output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim),
        )

    def forward(self, cnn, vit, rnn):
        outputs = [self.cnn_branch(cnn), self.rnn_branch(rnn)]
        if vit is not None:
            outputs.insert(1, self.vit_branch(vit))
        return self.fusion(torch.cat(outputs, dim=1))


@register_fusion("Early Fusion", default_output_dim=512)
class EarlyFusion(BaseFusionNetwork):
    def __init__(self, cnn_dim=512, vit_dim=0, rnn_dim=256, output_dim=512):
        super().__init__(cnn_dim, vit_dim, rnn_dim, output_dim)
        total = cnn_dim + vit_dim + rnn_dim
        self.net = nn.Sequential(
            nn.Linear(total, output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim),
            nn.Dropout(0.2),
        )

    def forward(self, cnn, vit, rnn):
        features = [cnn, rnn]
        if vit is not None:
            features.insert(1, vit)
        return self.net(torch.cat(features, dim=1))


@register_fusion("Hybrid Fusion", default_output_dim=512)
class HybridFusion(BaseFusionNetwork):
    def __init__(self, cnn_dim=512, vit_dim=0, rnn_dim=256, output_dim=512):
        super().__init__(cnn_dim, vit_dim, rnn_dim, output_dim)
        total = cnn_dim + vit_dim + rnn_dim
        self.early = nn.Linear(total, output_dim)
        self.cnn_proj = nn.Linear(cnn_dim, output_dim)
        self.rnn_proj = nn.Linear(rnn_dim, output_dim)
        if vit_dim > 0:
            self.vit_proj = nn.Linear(vit_dim, output_dim)
        self.gate = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim),
        )

    def forward(self, cnn, vit, rnn):
        features = [cnn, rnn]
        if vit is not None:
            features.insert(1, vit)
        early = self.early(torch.cat(features, dim=1))
        late = self.cnn_proj(cnn) + (self.vit_proj(vit) if vit is not None else 0) + self.rnn_proj(rnn)
        g = self.gate(torch.cat([early, late], dim=1))
        return self.out(g * early + (1 - g) * late)


@register_fusion("Multimodal Fusion", default_output_dim=512)
class MultimodalFusion(BaseFusionNetwork):
    def __init__(self, cnn_dim=512, vit_dim=0, rnn_dim=256, output_dim=512):
        super().__init__(cnn_dim, vit_dim, rnn_dim, output_dim)
        self.cnn_net = _mlp_block(cnn_dim, 1024, output_dim, 0.3)
        self.rnn_net = _mlp_block(rnn_dim, 512, output_dim, 0.3)
        if vit_dim > 0:
            self.vit_net = _mlp_block(vit_dim, 1024, output_dim, 0.3)
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * (3 if vit_dim > 0 else 2), output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim),
        )

    def forward(self, cnn, vit, rnn):
        outputs = [self.cnn_net(cnn), self.rnn_net(rnn)]
        if vit is not None:
            outputs.insert(1, self.vit_net(vit))
        return self.fusion(torch.cat(outputs, dim=1))


@register_fusion("Attention-based Fusion", default_output_dim=512)
class AttentionBasedFusion(BaseFusionNetwork):
    def __init__(self, cnn_dim=512, vit_dim=0, rnn_dim=256, output_dim=512):
        super().__init__(cnn_dim, vit_dim, rnn_dim, output_dim)
        self.attn = nn.Sequential(
            nn.Linear(cnn_dim + vit_dim + rnn_dim, 128),
            nn.Tanh(),
            nn.Linear(128, (3 if vit_dim > 0 else 2)),
            nn.Softmax(dim=1),
        )
        self.proj_cnn = nn.Linear(cnn_dim, output_dim)
        self.proj_rnn = nn.Linear(rnn_dim, output_dim)
        if vit_dim > 0:
            self.proj_vit = nn.Linear(vit_dim, output_dim)
        self.out = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim),
        )

    def forward(self, cnn, vit, rnn):
        features = [cnn, rnn]
        if vit is not None:
            features.insert(1, vit)
        concat = torch.cat(features, dim=1)
        weights = self.attn(concat)
        proj = [self.proj_cnn(cnn), self.proj_rnn(rnn)]
        if vit is not None:
            proj.insert(1, self.proj_vit(vit))
        stacked = torch.stack(proj, dim=1)
        weighted = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        return self.out(weighted)


class _GatedBase(BaseFusionNetwork):
    def __init__(self, cnn_dim=512, vit_dim=0, rnn_dim=256, output_dim=512):
        super().__init__(cnn_dim, vit_dim, rnn_dim, output_dim)
        self.cnn_proj = nn.Linear(cnn_dim, output_dim)
        self.rnn_proj = nn.Linear(rnn_dim, output_dim)
        if vit_dim > 0:
            self.vit_proj = nn.Linear(vit_dim, output_dim)
        self.num_mod = 3 if vit_dim > 0 else 2


@register_fusion("Gated Fusion", default_output_dim=512)
class GatedFusion(_GatedBase):
    def __init__(self, cnn_dim=512, vit_dim=0, rnn_dim=256, output_dim=512):
        super().__init__(cnn_dim, vit_dim, rnn_dim, output_dim)
        gate_input_dim = output_dim * self.num_mod
        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.num_mod),
            nn.Softmax(dim=1),
        )
        self.fusion = nn.Sequential(
            nn.Linear(gate_input_dim + output_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim),
        )

    def forward(self, cnn, vit, rnn):
        proj = [self.cnn_proj(cnn), self.rnn_proj(rnn)]
        if vit is not None:
            proj.insert(1, self.vit_proj(vit))
        concat = torch.cat(proj, dim=1)
        weights = self.gate(concat)
        stacked = torch.stack(proj, dim=1)
        gated_sum = torch.sum(stacked * weights.unsqueeze(-1), dim=1)
        return self.fusion(torch.cat([concat, gated_sum], dim=1))


@register_fusion("Residual Fusion", default_output_dim=512)
class ResidualFusion(BaseFusionNetwork):
    def __init__(self, cnn_dim=512, vit_dim=0, rnn_dim=256, output_dim=512):
        super().__init__(cnn_dim, vit_dim, rnn_dim, output_dim)
        self.proj = nn.Linear(cnn_dim + vit_dim + rnn_dim, output_dim)
        self.res_block = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim),
            nn.Dropout(0.2),
            nn.Linear(output_dim, output_dim),
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, cnn, vit, rnn):
        features = [cnn, rnn]
        if vit is not None:
            features.insert(1, vit)
        x = self.proj(torch.cat(features, dim=1))
        residual = x
        x = self.res_block(x)
        return self.norm(residual + x)


@register_fusion("Bilinear Fusion", default_output_dim=512)
class BilinearFusion(BaseFusionNetwork):
    def __init__(self, cnn_dim=512, vit_dim=0, rnn_dim=256, output_dim=512):
        super().__init__(cnn_dim, vit_dim, rnn_dim, output_dim)
        self.cnn_proj = nn.Linear(cnn_dim, output_dim)
        self.rnn_proj = nn.Linear(rnn_dim, output_dim)
        if vit_dim > 0:
            self.vit_proj = nn.Linear(vit_dim, output_dim)
        self.bilinear = nn.Bilinear(output_dim, output_dim, output_dim)
        self.out = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim),
            nn.Dropout(0.2),
        )

    def forward(self, cnn, vit, rnn):
        c = self.cnn_proj(cnn)
        r = self.rnn_proj(rnn)
        if vit is not None:
            v = self.vit_proj(vit)
            bilinear = self.bilinear(c, v) + self.bilinear(v, r) + self.bilinear(c, r)
        else:
            bilinear = self.bilinear(c, r)
        return self.out(bilinear)


@register_fusion("Tensor Fusion (TFN)", default_output_dim=512)
class TensorFusion(BaseFusionNetwork):
    def __init__(self, cnn_dim=512, vit_dim=0, rnn_dim=256, output_dim=512):
        super().__init__(cnn_dim, vit_dim, rnn_dim, output_dim)
        self.cnn_proj = nn.Linear(cnn_dim, output_dim)
        self.rnn_proj = nn.Linear(rnn_dim, output_dim)
        if vit_dim > 0:
            self.vit_proj = nn.Linear(vit_dim, output_dim)
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * output_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim),
        )

    def forward(self, cnn, vit, rnn):
        c = self.cnn_proj(cnn)
        r = self.rnn_proj(rnn)
        outer = torch.bmm(c.unsqueeze(2), r.unsqueeze(1))
        b, d, _ = outer.shape
        x = outer.reshape(b, d * d)
        if vit is not None:
            v = self.vit_proj(vit)
            outer_vc = torch.bmm(v.unsqueeze(2), c.unsqueeze(1))
            outer_vr = torch.bmm(v.unsqueeze(2), r.unsqueeze(1))
            x = x + outer_vc.reshape(b, d * d) + outer_vr.reshape(b, d * d)
        return self.fusion(x)


@register_fusion("Multimodal Transformer Fusion", default_output_dim=512)
class MultimodalTransformerFusion(BaseFusionNetwork):
    def __init__(self, cnn_dim=512, vit_dim=0, rnn_dim=256, output_dim=512):
        super().__init__(cnn_dim, vit_dim, rnn_dim, output_dim)
        self.cnn_proj = nn.Linear(cnn_dim, output_dim)
        self.rnn_proj = nn.Linear(rnn_dim, output_dim)
        if vit_dim > 0:
            self.vit_proj = nn.Linear(vit_dim, output_dim)
        num_mod = 3 if vit_dim > 0 else 2
        encoder_layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=8, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.cls_token = nn.Parameter(torch.randn(1, 1, output_dim))
        self.out = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(output_dim),
        )

    def forward(self, cnn, vit, rnn):
        projected = [self.cnn_proj(cnn), self.rnn_proj(rnn)]
        if vit is not None:
            projected.insert(1, self.vit_proj(vit))
        tokens = torch.stack(projected, dim=1)
        b = tokens.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        encoded = self.transformer(tokens)
        return self.out(encoded[:, 0, :])


@register_fusion("Adaptive Fusion", default_output_dim=512)
class AdaptiveFusion(BaseFusionNetwork):
    def __init__(self, cnn_dim=512, vit_dim=0, rnn_dim=256, output_dim=512):
        super().__init__(cnn_dim, vit_dim, rnn_dim, output_dim)
        self.cnn_net = _mlp_block(cnn_dim, 512, output_dim, 0.2)
        self.rnn_net = _mlp_block(rnn_dim, 256, output_dim, 0.2)
        if vit_dim > 0:
            self.vit_net = _mlp_block(vit_dim, 512, output_dim, 0.2)
        num_mod = 3 if vit_dim > 0 else 2
        self.adapt = nn.Sequential(
            nn.Linear(output_dim * num_mod, 128),
            nn.ReLU(),
            nn.Linear(128, num_mod * output_dim),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Dropout(0.2),
        )

    def forward(self, cnn, vit, rnn):
        outputs = [self.cnn_net(cnn), self.rnn_net(rnn)]
        if vit is not None:
            outputs.insert(1, self.vit_net(vit))
        stacked = torch.stack(outputs, dim=1)
        b, n, d = stacked.shape
        adapt_params = self.adapt(stacked.reshape(b, n * d)).reshape(b, n, d)
        weighted = stacked * torch.sigmoid(adapt_params)
        return self.out(weighted.sum(dim=1))


class FusionNetworkFactory:
    @staticmethod
    def list_models() -> list[str]:
        return sorted(FUSION_REGISTRY.keys())

    @staticmethod
    def get_info(model_name: str) -> dict:
        normalized = _normalize_name(model_name)
        if normalized not in FUSION_REGISTRY:
            raise KeyError(f"Unknown Fusion model: {model_name}. Available: {sorted(FUSION_REGISTRY.keys())}")
        info = FUSION_REGISTRY[normalized]
        return {"name": normalized, "output_dim": info["output_dim"]}

    @staticmethod
    def create(model_name: str, cnn_dim: int = 512, vit_dim: int = 0, rnn_dim: int = 256, output_dim: int = 512) -> BaseFusionNetwork:
        normalized = _normalize_name(model_name)
        if normalized not in FUSION_REGISTRY:
            raise KeyError(f"Unknown Fusion model: {model_name}. Available: {sorted(FUSION_REGISTRY.keys())}")
        info = FUSION_REGISTRY[normalized]
        return info["class"](cnn_dim=cnn_dim, vit_dim=vit_dim, rnn_dim=rnn_dim, output_dim=output_dim)


def _normalize_name(name: str) -> str:
    mapping = {
        "standard_fc": "Standard FC",
        "standardfc": "Standard FC",
        "fc": "Standard FC",
        "mlp": "MLP",
        "deep_dense_fusion": "Deep Dense Fusion",
        "deepdensefusion": "Deep Dense Fusion",
        "cross_modal_attention": "Cross-Modal Attention",
        "crossmodalattention": "Cross-Modal Attention",
        "self_attention_fusion": "Self-Attention Fusion",
        "selfattentionfusion": "Self-Attention Fusion",
        "late_fusion": "Late Fusion",
        "latefusion": "Late Fusion",
        "early_fusion": "Early Fusion",
        "earlyfusion": "Early Fusion",
        "hybrid_fusion": "Hybrid Fusion",
        "hybridfusion": "Hybrid Fusion",
        "multimodal_fusion": "Multimodal Fusion",
        "multimodalfusion": "Multimodal Fusion",
        "attention_based_fusion": "Attention-based Fusion",
        "attentionbasedfusion": "Attention-based Fusion",
        "gated_fusion": "Gated Fusion",
        "gatedfusion": "Gated Fusion",
        "residual_fusion": "Residual Fusion",
        "residualfusion": "Residual Fusion",
        "bilinear_fusion": "Bilinear Fusion",
        "bilinearfusion": "Bilinear Fusion",
        "tensor_fusion": "Tensor Fusion (TFN)",
        "tensorfusion": "Tensor Fusion (TFN)",
        "tfn": "Tensor Fusion (TFN)",
        "multimodal_transformer_fusion": "Multimodal Transformer Fusion",
        "multimodaltransformerfusion": "Multimodal Transformer Fusion",
        "adaptive_fusion": "Adaptive Fusion",
        "adaptivefusion": "Adaptive Fusion",
    }
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    return mapping.get(key, name)
