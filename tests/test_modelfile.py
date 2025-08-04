import io
import json
import textwrap
import pytest
from mindforge.modelfile import parse_modelfile


def write_tmp(tmp_path, content):
    p = tmp_path / "Modelfile"
    p.write_text(textwrap.dedent(content))
    return p


def test_parse_modelfile_happy(tmp_path):
    p = write_tmp(
        tmp_path,
        (
            "\n"
            "FROM org/model\n"
            "TAGS [\"gguf\",\"q4\"]\n"
            "PARAMS device=cpu dtype=float32 quant=Q4_K_M temperature=0.7 top_p=1.0\n"
            "SYSTEM \"\"\"\nYou are a helpful AI assistant.\n\"\"\"\n"
        ),
    )
    cfg = parse_modelfile(str(p))
    assert cfg["from"] == "org/model"
    assert cfg["tags"] == ["gguf", "q4"]
    assert cfg["params"]["device"] == "cpu"
    assert "system" in cfg and "helpful" in cfg["system"]


def test_parse_modelfile_requires_from(tmp_path):
    p = write_tmp(
        tmp_path,
        (
            "\n"
            "TAGS [\"gguf\"]\n"
            "SYSTEM \"\"\"\nhi\n\"\"\"\n"
        ),
    )
    with pytest.raises(ValueError):
        parse_modelfile(str(p))


def test_parse_modelfile_invalid_tags(tmp_path):
    p = write_tmp(
        tmp_path,
        """
        FROM org/model
        TAGS notjson
        """,
    )
    with pytest.raises(ValueError):
        parse_modelfile(str(p))
