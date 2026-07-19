import io
from typing import Dict, List, Optional

import pandas as pd

from models import BusinessMetrics


COLUMN_KEYWORDS: Dict[str, List[str]] = {

    "sales": ["sales"],
    "revenue": ["revenue", "total revenue", "income"],
    "profit": ["profit", "net profit", "margin"],
    "expenses": ["expense", "cost", "expenditure"],
    "order_value": ["order value", "order amount", "order total"],
    "customer": ["customer"],
    "product": ["product", "item", "sku"],
    "region": ["region", "location", "state", "country", "city"],
}


def load_dataset(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Load a CSV or Excel file (given as bytes) into a DataFrame."""
    buffer = io.BytesIO(file_bytes)
    if filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(buffer)
    return pd.read_csv(buffer)


def validate_dataset(df: pd.DataFrame) -> List[str]:
    """Basic sanity checks. Returns a list of warning strings (empty if clean)."""
    warnings: List[str] = []
    if df.empty:
        warnings.append("The dataset has no rows.")
    if df.shape[1] == 0:
        warnings.append("The dataset has no columns.")
    return warnings


def find_column(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    """Find the first column whose name contains any of the given keywords (case-insensitive)."""
    for col in df.columns:
        col_lower = str(col).lower()
        for kw in keywords:
            if kw in col_lower:
                return col
    return None


def detect_dataset_type(df: pd.DataFrame) -> str:
    """Classify the dataset as Sales, Financial, or Business Intelligence using column-name heuristics."""
    cols = " ".join(str(c).lower() for c in df.columns)

    has_sales = any(kw in cols for kw in COLUMN_KEYWORDS["sales"] + COLUMN_KEYWORDS["product"])
    has_financial = any(
        kw in cols for kw in COLUMN_KEYWORDS["revenue"] + COLUMN_KEYWORDS["profit"] + COLUMN_KEYWORDS["expenses"]
    )

    if has_financial and not has_sales:
        return "Financial"
    if has_sales and not has_financial:
        return "Sales"
    if has_sales and has_financial:
        sales_score = cols.count("sales") + cols.count("product")
        financial_score = cols.count("revenue") + cols.count("profit")
        return "Sales" if sales_score >= financial_score else "Financial"
    return "Business Intelligence"


def calculate_business_metrics(df: pd.DataFrame) -> BusinessMetrics:
    """Calculate headline business metrics wherever a matching column exists in the dataset."""
    metrics = BusinessMetrics()

    sales_col = find_column(df, COLUMN_KEYWORDS["sales"])
    if sales_col is not None and pd.api.types.is_numeric_dtype(df[sales_col]):
        metrics.total_sales = float(df[sales_col].sum())

    revenue_col = find_column(df, COLUMN_KEYWORDS["revenue"])
    if revenue_col is not None and pd.api.types.is_numeric_dtype(df[revenue_col]):
        metrics.total_revenue = float(df[revenue_col].sum())

    profit_col = find_column(df, COLUMN_KEYWORDS["profit"])
    if profit_col is not None and pd.api.types.is_numeric_dtype(df[profit_col]):
        metrics.total_profit = float(df[profit_col].sum())

    expenses_col = find_column(df, COLUMN_KEYWORDS["expenses"])
    if expenses_col is not None and pd.api.types.is_numeric_dtype(df[expenses_col]):
        metrics.total_expenses = float(df[expenses_col].sum())

    order_value_col = find_column(df, COLUMN_KEYWORDS["order_value"])
    if order_value_col is not None and pd.api.types.is_numeric_dtype(df[order_value_col]):
        metrics.average_order_value = float(df[order_value_col].mean())
    elif revenue_col is not None and pd.api.types.is_numeric_dtype(df[revenue_col]):
        metrics.average_order_value = float(df[revenue_col].mean())

    customer_col = find_column(df, COLUMN_KEYWORDS["customer"])
    if customer_col is not None:
        metrics.total_customers = int(df[customer_col].nunique())

    product_col = find_column(df, COLUMN_KEYWORDS["product"])
    if product_col is not None:
        metrics.total_products = int(df[product_col].nunique())

    return metrics


def get_preview(df: pd.DataFrame, rows: int = 10) -> pd.DataFrame:
    """Return the first N rows for display in the UI."""
    return df.head(rows)


def get_statistics(df: pd.DataFrame) -> Dict[str, object]:
    """Return basic dataset statistics: shape, missing values, and duplicate row count."""
    missing = df.isnull().sum()
    return {
        "row_count": len(df),
        "column_count": df.shape[1],
        "missing_values": missing[missing > 0].to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }
