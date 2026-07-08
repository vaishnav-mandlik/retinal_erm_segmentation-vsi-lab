from typing import Dict, Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation


class SegFormerWrap(torch.nn.Module):
    """Wrap HF SegFormer to output (B,C,H,W) logits resized to input size."""
    def __init__(self, variant: str, num_labels: int):
        super().__init__()
        if SegformerForSemanticSegmentation is None:
            raise RuntimeError("transformers not installed for SegFormer.")
        id_map = {"segformer_b0":"nvidia/segformer-b0-finetuned-ade-512-512",
                  "segformer_b2":"nvidia/segformer-b2-finetuned-ade-512-512"}
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            id_map[variant], num_labels=num_labels, ignore_mismatched_sizes=True
        )
    def forward(self, x):
        B, C, H, W = x.shape
        out = self.model(pixel_values=x)
        logits = out.logits  # (B, C, H/4, W/4) typically
        return F.interpolate(logits, size=(H,W), mode="bilinear", align_corners=False)


class MultiTaskSegWithPhase(torch.nn.Module):
    """
    Wrap SMP model to extract encoder top feature for phase head.
    Works with Unet++ and DeepLabV3+ in SMP (have .encoder/.decoder).
    """
    def __init__(self, smp_model: torch.nn.Module, n_phases: int, enc_feats_hint: int = 512):
        super().__init__()
        self.seg = smp_model
        # try to infer encoder out channels
        ch = enc_feats_hint
        try:
            ch = self.seg.encoder.out_channels[-1]
        except Exception:
            pass
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.phase_head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(ch, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, n_phases)
        )

    @staticmethod
    def _decode(decoder, features):
        """
        SMP backbones/decoders differ:
          - some expect decoder(*features)
          - others expect decoder(features)
        Try unpack first, fall back to single arg.
        """
        try:
            return decoder(*features)
        except TypeError:
            return decoder(features)

    def forward(self, x):
        feats = self.seg.encoder(x)          # list of feature maps
        enc_top = feats[-1]                   # top-level encoder feature
        dec = self._decode(self.seg.decoder, feats)
        logits = self.seg.segmentation_head(dec)
        phase_logits = self.phase_head(self.pool(enc_top))
        return logits, phase_logits


def _build_model_from_cfg(cfg: Dict, out_channels: int):
    t = cfg["train"]
    model_name = t.get("model","unetpp").lower()
    if model_name in ("unetpp","deeplabv3p"):
        model = _build_smp_model(model_name, t.get("encoder_name","resnet34"), t.get("encoder_weights","imagenet"), out_channels)

        # Check if model loads pretrained weights
        check_pretrained(model, t)

        # With multihead phase
        if t.get("phase_head",{}).get("enabled", False) and len(cfg["train"].get("phase_labels",[]))>0:
            model = MultiTaskSegWithPhase(model, n_phases=len(cfg["train"]["phase_labels"]),
                                          enc_feats_hint=int(t.get("phase_head",{}).get("enc_feats_hint",512)))
            print("Using Multi task Phase Head extn.....")
        return model
    if model_name in ("segformer_b0","segformer_b2"):
        model = SegFormerWrap(model_name, num_labels=out_channels)
        # phase head not supported here in this drop-in (keep SMP path for Track C)
        return model
    raise ValueError(f"Unknown model: {model_name}")


def check_pretrained(model, t):
    print(f"[enc] requested encoder_weights={t.get('encoder_weights', '<unset>')}")
    # 2) If using timm under the hood, this exposes the pretrained recipe
    pcfg = getattr(model.encoder, "pretrained_cfg", None)
    print("[enc] pretrained_cfg:", None if pcfg is None else pcfg.get("tag", pcfg.get("hf_hub_id", "unknown")))
    # 3) Sanity: distance to a fresh Kaiming init (should be large if pretrained)
    with torch.no_grad():
        w = model.encoder.conv1.weight.detach().cpu()
        torch.manual_seed(123)
        w_rand = torch.empty_like(w);
        torch.nn.init.kaiming_normal_(w_rand, nonlinearity="relu")
        delta = torch.norm(w - w_rand).item()
        print(f"[enc] L2 delta vs fresh Kaiming: {delta:.4f}")


def _build_smp_model(model_name: str, encoder_name: str, encoder_weights: Optional[str], out_channels: int):
    if smp is None:
        raise RuntimeError("segmentation_models_pytorch not installed. Please install to use SMP backbones.")
    if model_name == "unetpp":
        return smp.UnetPlusPlus(encoder_name=encoder_name,
                                encoder_weights=encoder_weights,
                                in_channels=3, classes=out_channels, activation=None)
    if model_name == "deeplabv3p":
        return smp.DeepLabV3Plus(encoder_name=encoder_name,
                                 encoder_weights=encoder_weights,
                                 in_channels=3, classes=out_channels, activation=None)
    raise ValueError(f"Unsupported SMP model: {model_name}")
