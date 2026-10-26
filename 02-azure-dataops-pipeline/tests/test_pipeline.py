import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import extract, transform, validate


def test_transform_adds_revenue_and_standardizes_region():
    df = transform(extract(n=30, seed=1))
    assert 'revenue' in df.columns
    assert df['region'].notna().all()
    assert df['region'].str.istitle().all()


def test_validation_flags_negative_quantity():
    df = transform(extract(n=30, seed=1))
    assert any('quantity must be positive' in issue for issue in validate(df))
