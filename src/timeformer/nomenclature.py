"""
Public-facing names for the Timeformer ablation chain.

The short model IDs are kept as stable experimental identifiers because they
appear in checkpoints, run directories, and historical result files. These
labels are for tables, plots, and paper text.
"""

MODEL_LABELS = {
    "Static": "Static Transformer",
    "Additive": "Additive Time-Conditioned Transformer",
    "Joint": "Token-Time Transformer",
    "Timeformer": "Memory-Augmented Timeformer",
    "Joint_ref": "Token-Time Transformer (reference)",
    "Timeformer_learned": "Memory-Augmented Timeformer (learned memory)",
    "Timeformer_oracle": "Oracle-Memory Timeformer",
    "Timeformer_shuffled": "Shuffled-Memory Timeformer",
    "Timeformer_nohistory": "No-History Timeformer",
}

ABLATION_LABELS = {
    "delta_time_conditioning": "Delta time-conditioning",
    "delta_token_time_interaction": "Delta token-time interaction",
    "delta_memory": "Delta memory",
    "delta_spurious_memory": "Delta spurious memory",
}

ABLATION_DISPLAY = {
    "delta_time_conditioning": (
        "Additive Time-Conditioned Transformer - Static Transformer"
    ),
    "delta_token_time_interaction": (
        "Token-Time Transformer - Additive Time-Conditioned Transformer"
    ),
    "delta_memory": "Memory-Augmented Timeformer - Token-Time Transformer",
    "delta_spurious_memory": (
        "Shuffled-memory Timeformer - Token-Time Transformer"
    ),
}

LEGACY_ABLATION_ALIASES = {
    "delta_time_conditioning": "ΔT_global",
    "delta_token_time_interaction": "ΔT_inter",
    "delta_memory": "ΔA",
    "delta_spurious_memory": "ΔA_spurious",
}


def model_label(model_id: str) -> str:
    """Return the public label for a canonical model ID."""
    return MODEL_LABELS.get(model_id, model_id)
