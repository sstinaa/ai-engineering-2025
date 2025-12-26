import pandas as pd

from eda_cli.core import compute_quality_flags


def test_has_constant_columns():
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "const": [7, 7, 7],
        }
    )
    flags = compute_quality_flags(df)
    assert flags["has_constant_columns"] is True


def test_has_many_zero_values():
    df = pd.DataFrame(
        {
            "x": [0, 0, 0, 10],
            "y": [1, 2, 3, 4],
        }
    )
    flags = compute_quality_flags(df, zero_share_threshold=0.5)
    assert flags["has_many_zero_values"] is True
