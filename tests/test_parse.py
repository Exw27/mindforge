import pytest
from mindforge.main import parse_model_name

@pytest.mark.parametrize(
    "inp,exp",
    [
        ("gpt2", ("gpt2", None)),
        ("org/model:Q4_K_M", ("org/model", "Q4_K_M")),
    ],
)
def test_parse_model_name_ok(inp, exp):
    assert parse_model_name(inp) == exp

@pytest.mark.parametrize("bad", ["/abs", "./rel", "../up", "..\\up", "\\abs"])
def test_parse_model_name_rejects_paths(bad):
    with pytest.raises(ValueError):
        parse_model_name(bad)
