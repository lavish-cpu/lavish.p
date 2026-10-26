from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / 'data'


def extract(n=500, seed=7):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'order_id': np.arange(1, n + 1),
        'order_date': pd.date_range('2025-01-01', periods=n, freq='D').astype(str),
        'region': rng.choice(['north', 'south', 'east', 'west', None], n),
        'quantity': rng.integers(1, 12, n),
        'unit_price': np.round(rng.uniform(100, 5000, n), 2),
    })
    if n > 10:
        df.loc[3, 'quantity'] = -2
        df.loc[8, 'region'] = None
    return df


def transform(df):
    out = df.copy()
    out['region'] = out['region'].fillna('unknown').str.strip().str.title()
    out['order_date'] = pd.to_datetime(out['order_date'], errors='coerce')
    out['quantity'] = pd.to_numeric(out['quantity'], errors='coerce')
    out['unit_price'] = pd.to_numeric(out['unit_price'], errors='coerce')
    out['revenue'] = out['quantity'] * out['unit_price']
    return out


def validate(df):
    issues = []
    if df['order_id'].isna().any() or df['order_id'].duplicated().any():
        issues.append('order_id must be present and unique')
    if (df['quantity'] <= 0).any():
        issues.append('quantity must be positive')
    if (df['unit_price'] < 0).any():
        issues.append('unit_price cannot be negative')
    if df['order_date'].isna().any():
        issues.append('order_date contains invalid values')
    return issues


def main():
    DATA.mkdir(exist_ok=True)
    raw = extract()
    raw.to_csv(DATA / 'raw_sales.csv', index=False)
    curated = transform(raw)
    issues = validate(curated)
    print(f'Rows extracted: {len(raw)}')
    print(f'Quality issues: {issues or "none"}')
    # Keep only valid records; persist rejected rows for audit.
    rejected_mask = (curated['quantity'] <= 0) | curated['order_date'].isna() | curated['unit_price'].isna()
    curated.loc[rejected_mask].to_csv(DATA / 'rejected_sales.csv', index=False)
    curated.loc[~rejected_mask].to_csv(DATA / 'curated_sales.csv', index=False)
    print(f'Rows curated: {(~rejected_mask).sum()}')

if __name__ == '__main__':
    main()
