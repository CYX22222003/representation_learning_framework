"""Frozen Phase 3 price framework matrix."""

from __future__ import annotations

from phase3_encoder_common import DEFAULT_PRICE_ROOT

BASE = ("statistical", "transformed", "vae_mlp", "contrastive_cnn", "byol_cnn")
PRICE_CONFIGS = {
    "H0": BASE,
    "HC-SL": ("statistical", "transformed", "vae_mlp", "contrastive_lstm", "byol_cnn"),
    "HC-ST": ("statistical", "transformed", "vae_mlp", "contrastive_transformer", "byol_cnn"),
    "HC-AL": BASE + ("contrastive_lstm",),
    "HC-AT": BASE + ("contrastive_transformer",),
    "HC-DC": BASE + ("contrastive_cnn@dup1",),
    "HC-ALT": BASE + ("contrastive_lstm", "contrastive_transformer"),
    "HC-DD": BASE + ("contrastive_cnn@dup1", "contrastive_cnn@dup2"),
    "HB-SL": ("statistical", "transformed", "vae_mlp", "contrastive_cnn", "byol_lstm"),
    "HB-ST": ("statistical", "transformed", "vae_mlp", "contrastive_cnn", "byol_transformer"),
    "HB-AL": BASE + ("byol_lstm",),
    "HB-AT": BASE + ("byol_transformer",),
    "HB-DC": BASE + ("byol_cnn@dup1",),
    "HB-ALT": BASE + ("byol_lstm", "byol_transformer"),
    "HB-DD": BASE + ("byol_cnn@dup1", "byol_cnn@dup2"),
}


def source_branch(name: str) -> str:
    return name.split("@", 1)[0]


def run_root(config_id: str):
    return DEFAULT_PRICE_ROOT / config_id / "seed0"
