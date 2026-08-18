"""Regression checks for checkpoint-free joint MLP+CNN training."""

from __future__ import annotations

from argparse import Namespace
import math

import torch

from mcar.models.residual_mlp_cnn import ResidualMLPCNN
from mcar.training.train_mlp_cnn_v3 import (
    infer_mlp_architecture_from_arguments,
    make_optimizer,
    make_scheduler,
    training_stage,
)


def main() -> None:
    assert infer_mlp_architecture_from_arguments(
        {"width": 128, "block_count": 3}
    ) == (128, 3)
    assert infer_mlp_architecture_from_arguments(
        {"mlp_width": 128, "mlp_block_count": 3}
    ) == (128, 3)

    torch.manual_seed(20260809)
    model = ResidualMLPCNN(
        mlp_width=16,
        mlp_block_count=1,
        cnn_channels=8,
    )
    hidden_weight_before = model.mlp.input[0].weight.detach().clone()
    assert torch.count_nonzero(hidden_weight_before).item() > 0
    model.mlp.zero_initialize_output()
    assert torch.count_nonzero(model.mlp.output.weight).item() == 0
    assert torch.count_nonzero(model.mlp.output.bias).item() == 0
    assert torch.count_nonzero(model.cnn.output.weight).item() == 0
    assert torch.count_nonzero(model.cnn.output.bias).item() == 0

    # The widest CNN block reflect-pads by 24 bins on each side.
    features = torch.randn(3, 2, 65, 7)
    prediction, base, delta = model(features)
    assert torch.count_nonzero(prediction).item() == 0
    assert torch.count_nonzero(base).item() == 0
    assert torch.count_nonzero(delta).item() == 0

    optimizer, trainable = make_optimizer(
        model,
        learning_rate=3e-4,
        weight_decay=1e-5,
        unfreeze_mlp=True,
        mlp_learning_rate=1e-3,
    )
    assert len(trainable) == len(list(model.parameters()))
    assert [group["name"] for group in optimizer.param_groups] == [
        "cnn",
        "mlp",
    ]
    scheduler = make_scheduler(optimizer, epochs=60, warmup_epochs=2)
    assert all(
        math.isclose(actual, expected, rel_tol=1e-12)
        for actual, expected in zip(
            [group["lr"] for group in optimizer.param_groups],
            [3e-5, 1e-4],
        )
    )
    scheduler.step()
    assert [group["lr"] for group in optimizer.param_groups] == [
        0.000165,
        0.00055,
    ]
    scheduler.step()
    assert [group["lr"] for group in optimizer.param_groups] == [3e-4, 1e-3]

    loss = model(features)[0].square().mean() + model(features)[0].mean()
    loss.backward()
    assert model.mlp.output.weight.grad is not None
    assert model.cnn.output.weight.grad is not None
    assert torch.isfinite(model.mlp.output.weight.grad).all()
    assert torch.isfinite(model.cnn.output.weight.grad).all()

    stage = training_stage(
        Namespace(
            random_initialize_mlp=True,
            unfreeze_mlp=True,
            dual_sampling_strict_ild=True,
            ild_loss_mode="strict_hrir",
            horizontal_only=False,
        )
    )
    assert stage == (
        "scratch_joint_mlp_cnn_global_magnitude_horizontal_hrir_ild_v32"
    )
    print(
        {
            "status": "passed",
            "initial_prediction_nonzero": torch.count_nonzero(prediction).item(),
            "parameter_groups": [
                group["name"] for group in optimizer.param_groups
            ],
            "training_stage": stage,
        }
    )


if __name__ == "__main__":
    main()
