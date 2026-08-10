"""Regression checks for the v3.3 gated global spectral branch."""

from __future__ import annotations

from argparse import Namespace

import torch

from mcar.models.residual_mlp_cnn import ResidualMLPCNN
from mcar.training.train_mlp_cnn_v3 import make_optimizer, training_stage


def main() -> None:
    torch.manual_seed(20260809)
    source = ResidualMLPCNN(
        mlp_width=16,
        mlp_block_count=1,
        cnn_channels=8,
    )
    torch.nn.init.normal_(source.cnn.output.weight, std=0.05)
    torch.nn.init.normal_(source.cnn.output.bias, std=0.05)

    candidate = ResidualMLPCNN(
        mlp_width=16,
        mlp_block_count=1,
        cnn_channels=8,
        global_context_attention=True,
        global_attention_width=16,
        global_attention_heads=4,
        global_attention_blocks=1,
        global_attention_stride=4,
    )
    incompatible = candidate.load_state_dict(source.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(
        key.startswith(("global_context.", "global_gate."))
        for key in incompatible.missing_keys
    )

    point_features = torch.randn(3, 2, 129, 7)
    # Ear-independent direction and frequency features match real input layout.
    point_features[:, 1, :, 2:6] = point_features[:, 0, :, 2:6]
    with torch.no_grad():
        source_prediction = source(point_features)[0]
        candidate_prediction = candidate(point_features)[0]
    torch.testing.assert_close(
        candidate_prediction,
        source_prediction,
        rtol=0.0,
        atol=0.0,
    )

    optimizer, trainable = make_optimizer(
        candidate,
        learning_rate=1e-4,
        weight_decay=1e-5,
        global_context_only=True,
    )
    assert [group["name"] for group in optimizer.param_groups] == [
        "global_context"
    ]
    assert all(not parameter.requires_grad for parameter in candidate.mlp.parameters())
    assert all(not parameter.requires_grad for parameter in candidate.cnn.parameters())
    assert all(
        parameter.requires_grad
        for parameter in candidate.global_context.parameters()
    )
    assert len(trainable) == len(
        list(candidate.global_context.parameters())
        + list(candidate.global_gate.parameters())
    )

    stage = training_stage(
        Namespace(
            global_context_attention=True,
            freeze_local_cnn=True,
            unfreeze_mlp=False,
            dual_sampling_strict_ild=True,
            ild_loss_mode="strict_hrir",
            horizontal_only=False,
        )
    )
    assert stage.endswith("horizontal_hrir_ild_v33")
    print(
        {
            "status": "passed",
            "initial_output_exactly_matches_v32": True,
            "training_stage": stage,
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in trainable
            ),
        }
    )


if __name__ == "__main__":
    main()
