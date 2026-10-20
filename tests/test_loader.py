import pytest
from torch import nn

from nanovllm.utils.loader import load_model


def test_load_model_rejects_missing_checkpoint_weights(tmp_path):
    model = nn.Linear(4, 4)
    with pytest.raises(FileNotFoundError, match="Refusing to run with randomly initialized model parameters"):
        load_model(model, str(tmp_path))


def test_load_model_reports_incomplete_weight_download(tmp_path):
    partial = tmp_path / ".cache" / "download" / "weights.incomplete"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"partial")
    model = nn.Linear(4, 4)
    with pytest.raises(FileNotFoundError, match="incomplete download artifact"):
        load_model(model, str(tmp_path))
