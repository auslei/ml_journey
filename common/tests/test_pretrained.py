import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from common import pretrained


def test_random_init_head_shape_and_seed():
    a, used_a = pretrained.load_resnet18(3, pretrained=False, seed=1)
    b, _ = pretrained.load_resnet18(3, pretrained=False, seed=1)
    assert not used_a
    assert a.fc.out_features == 3
    assert torch.equal(a.fc.weight, b.fc.weight)


def test_download_failure_falls_back(monkeypatch):
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(4, 2)

    def fake_resnet18(weights=None):
        if weights is not None:
            raise OSError("no network")
        return Tiny()

    monkeypatch.setattr(pretrained.models, "resnet18", fake_resnet18)
    model, used = pretrained.load_resnet18(5, pretrained=True)
    assert not used
    assert model.fc.out_features == 5


def test_rejects_bad_num_classes():
    with pytest.raises(ValueError, match="num_classes"):
        pretrained.load_resnet18(0)
