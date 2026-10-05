"""The hand-written Adam against ``torch.optim.Adam``, and its state resizing."""

from __future__ import annotations

import torch

from microsplat.optim import ADAM_BETA1, ADAM_BETA2, ADAM_EPS, Adam


def _quadratic(params: dict[str, torch.Tensor]) -> torch.Tensor:
    return (params["a"] ** 2).sum() + ((params["b"] - 1.5) ** 4).sum()


def _initial(seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "a": torch.randn((6, 3), dtype=torch.float64, generator=generator),
        "b": torch.randn((6,), dtype=torch.float64, generator=generator),
    }


def test_updates_match_torch_adam_step_by_step():
    rates = {"a": 0.05, "b": 0.01}
    ours = {k: v.clone().requires_grad_(True) for k, v in _initial().items()}
    theirs = {k: v.clone().requires_grad_(True) for k, v in _initial().items()}
    adam = Adam(ours)
    reference = torch.optim.Adam(
        [{"params": [theirs[k]], "lr": rates[k]} for k in theirs],
        betas=(ADAM_BETA1, ADAM_BETA2),
        eps=ADAM_EPS,
    )
    for _ in range(25):
        _quadratic(ours).backward()
        adam.step(ours, rates)
        reference.zero_grad()
        _quadratic(theirs).backward()
        reference.step()
        for name in ours:
            torch.testing.assert_close(ours[name], theirs[name], rtol=1e-12, atol=1e-14)


def test_step_clears_the_gradients():
    params = {k: v.requires_grad_(True) for k, v in _initial().items()}
    adam = Adam(params)
    _quadratic(params).backward()
    adam.step(params, {"a": 0.1, "b": 0.1})
    assert all(p.grad is None for p in params.values())


def test_first_step_moves_every_entry_by_the_learning_rate():
    # With bias correction the first Adam update is lr * sign(gradient).
    params = {k: v.requires_grad_(True) for k, v in _initial().items()}
    before = {k: v.detach().clone() for k, v in params.items()}
    adam = Adam(params)
    _quadratic(params).backward()
    adam.step(params, {"a": 0.1, "b": 0.02})
    torch.testing.assert_close((params["a"] - before["a"]).abs(), torch.full_like(before["a"], 0.1))
    torch.testing.assert_close(
        (params["b"] - before["b"]).abs(), torch.full_like(before["b"], 0.02)
    )


def test_resize_keeps_selected_moments_and_zero_fills_new_rows():
    params = {k: v.requires_grad_(True) for k, v in _initial().items()}
    adam = Adam(params)
    _quadratic(params).backward()
    adam.step(params, {"a": 0.1, "b": 0.1})
    kept_first = adam.first["a"].clone()
    kept_second = adam.second["b"].clone()

    keep = torch.tensor([True, False, True, True, False, True])
    adam.resize(keep, added=3)

    assert adam.first["a"].shape == (7, 3)
    assert adam.second["b"].shape == (7,)
    torch.testing.assert_close(adam.first["a"][:4], kept_first[keep])
    torch.testing.assert_close(adam.second["b"][:4], kept_second[keep])
    assert not adam.first["a"][4:].any()
    assert not adam.second["b"][4:].any()
    assert adam.steps == 1
