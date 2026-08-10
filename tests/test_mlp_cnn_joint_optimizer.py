"""Regression checks for optional low-rate MLP+CNN joint fine-tuning."""

from __future__ import annotations

from argparse import Namespace

from mcar.models.residual_mlp_cnn import ResidualMLPCNN
from mcar.training.train_mlp_cnn_v3 import make_optimizer, training_stage


def main() -> None:
    frozen_model = ResidualMLPCNN(
        mlp_width=16,
        mlp_block_count=1,
        cnn_channels=8,
    )
    frozen_optimizer, frozen_parameters = make_optimizer(
        frozen_model,
        learning_rate=1e-5,
        weight_decay=1e-5,
    )
    assert len(frozen_optimizer.param_groups) == 1
    assert frozen_optimizer.param_groups[0]["name"] == "cnn"
    assert all(
        not parameter.requires_grad
        for parameter in frozen_model.mlp.parameters()
    )
    assert len(frozen_parameters) == len(list(frozen_model.cnn.parameters()))

    joint_model = ResidualMLPCNN(
        mlp_width=16,
        mlp_block_count=1,
        cnn_channels=8,
    )
    joint_optimizer, joint_parameters = make_optimizer(
        joint_model,
        learning_rate=1e-5,
        weight_decay=1e-5,
        unfreeze_mlp=True,
        mlp_learning_rate=1e-6,
    )
    assert [group["name"] for group in joint_optimizer.param_groups] == [
        "cnn",
        "mlp",
    ]
    assert [group["lr"] for group in joint_optimizer.param_groups] == [
        1e-5,
        1e-6,
    ]
    assert all(
        parameter.requires_grad for parameter in joint_model.mlp.parameters()
    )
    assert len(joint_parameters) == len(list(joint_model.parameters()))

    stage = training_stage(
        Namespace(
            unfreeze_mlp=True,
            dual_sampling_strict_ild=True,
            ild_loss_mode="strict_hrir",
            horizontal_only=False,
        )
    )
    assert stage == "joint_mlp_cnn_global_magnitude_horizontal_hrir_ild_v321"
    print(
        {
            "status": "passed",
            "parameter_groups": [
                group["name"] for group in joint_optimizer.param_groups
            ],
            "cnn_learning_rate": joint_optimizer.param_groups[0]["lr"],
            "mlp_learning_rate": joint_optimizer.param_groups[1]["lr"],
            "training_stage": stage,
        }
    )


if __name__ == "__main__":
    main()
