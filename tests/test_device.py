"""Device selection and seeding. Owned by #1."""

from __future__ import annotations

import torch

from pr_edge.device import device_name, get_device, set_seed


def test_auto_returns_a_usable_device():
    device = get_device("auto")
    assert device.type in {"cuda", "mps", "cpu"}
    torch.zeros(1, device=device)  # must actually be usable


def test_cpu_is_always_available():
    assert get_device("cpu").type == "cpu"


def test_unavailable_accelerator_falls_back_rather_than_crashing():
    """A member without a GPU must still be able to run everything."""
    if not torch.cuda.is_available():
        assert get_device("cuda").type == "cpu"


def test_device_name_is_non_empty():
    assert device_name(get_device("auto"))


def test_set_seed_makes_training_reproducible():
    """Two identical short runs must produce bit-identical losses.

    This is what lets four people compare numbers that have to line up in one
    table, and what makes a rerun of a config reproduce its own curves.
    """

    def run() -> list[float]:
        set_seed(1234)
        model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = torch.nn.CrossEntropyLoss()

        losses = []
        for _ in range(5):
            x = torch.randn(4, 8)
            y = torch.randint(0, 3, (4,))
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return losses

    assert run() == run()
