from datasets import ClassLabel, Dataset, DatasetDict, Features, Value
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from rank_suite import config
from rank_suite.data import prepare_task_data


def tokenizer():
    words = ["[PAD]", "[UNK]", "Premise", ":", "Hypothesis", "trainmarker", "evalmarker", "b"]
    backend = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]")


def test_training_holdout_never_reads_official_validation(monkeypatch, tmp_path):
    features = Features({"sentence1": Value("string"), "sentence2": Value("string"),
                         "label": ClassLabel(names=["yes", "no"])})
    def split(n, marker):
        return Dataset.from_dict({"sentence1": [marker] * n, "sentence2": ["b"] * n,
                                  "label": [i % 2 for i in range(n)]}, features=features)
    raw = DatasetDict(train=split(60, "trainmarker"), validation=split(13, "evalmarker"))
    calls = []
    def load(*args, **kwargs):
        calls.append(kwargs)
        return raw
    monkeypatch.setattr("datasets.load_dataset", load)
    tok = tokenizer()
    td = prepare_task_data(tok, task_keys=["rte"], n_train=40,
                            selection_dir=tmp_path / "calibration", calibration=True)["rte"]
    assert (td.n_train, td.n_eval) == (32, 8)
    from rank_suite.utils import load_json
    selection = load_json(tmp_path / "calibration/data_selection_rte.json")
    assert len(selection["training_indices"]) == 32
    assert len(selection["holdout_indices"]) == 8
    assert not set(selection["training_indices"]) & set(selection["holdout_indices"])
    assert set(selection["training_indices"]) | set(selection["holdout_indices"]) == set(selection["selected_indices"])
    assert calls[0]["revision"] == config.DATA_REVISION
    eval_marker = tok.convert_tokens_to_ids("evalmarker")
    assert all(eval_marker not in row["input_ids"] for row in td.eval_dataset)
    normal = prepare_task_data(tok, task_keys=["rte"], n_train=40,
                               selection_dir=tmp_path / "main")["rte"]
    assert (normal.n_train, normal.n_eval) == (40, 13)
    assert all(eval_marker in row["input_ids"] for row in normal.eval_dataset)
    # Pair boundaries are explicit even with a tokenizer having no SEP postprocessor.
    assert tok.convert_tokens_to_ids("Hypothesis") in normal.train_dataset[0]["input_ids"]
    with pytest.raises(ValueError, match="only 60"):
        prepare_task_data(tok, task_keys=["rte"], n_train=100)
