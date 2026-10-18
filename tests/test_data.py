import json

import numpy as np
import pytest
from tokenizers import Tokenizer

from learn_llm.data import DTYPE, EOT, TokenData


def test_packing_and_shift(token_data):
    tokenizer = Tokenizer.from_file(str(token_data / "tokenizer.json"))
    tokens = np.fromfile(token_data / "train.bin", dtype=DTYPE)
    assert tokens[-1] == tokenizer.token_to_id(EOT)
    assert tokens.max() < tokenizer.get_vocab_size()
    data = TokenData(token_data, block_size=8)
    x, y = data.batch("train", 2, "cpu")
    assert x.shape == y.shape == (2, 8)
    np.testing.assert_array_equal(x[:, 1:], y[:, :-1])


def test_corrupt_metadata_rejected(token_data):
    path = token_data / "meta.json"
    meta = json.loads(path.read_text())
    meta["train_tokens"] += 1
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="token 数"):
        TokenData(token_data, block_size=8)


def test_too_short_rejected(token_data):
    with pytest.raises(ValueError, match="太短"):
        TokenData(token_data, block_size=1_000_000)
