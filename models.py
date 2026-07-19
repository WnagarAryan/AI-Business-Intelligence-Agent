from typing import List, Optional
from pydantic import BaseModel, Field


class DatasetSummary(BaseModel):
    """Initial AI-generated overview of an uploaded dataset."""

    dataset_type: str = Field(description="Sales, Financial, or Business Intelligence")
    summary: str = Field(description="Plain-English overview of what the dataset contains")
    data_quality_notes: List[str] = Field(
        default_factory=list, description="Observations about missing values, duplicates, or quality issues"
    )
    suggested_questions: List[str] = Field(
        default_factory=list, description="Example business questions the user could ask about this dataset"
    )


class BusinessMetrics(BaseModel):
    """Headline business metrics calculated by Pandas. Fields are None if the dataset has no matching column."""

    total_sales: Optional[float] = None
    total_revenue: Optional[float] = None
    total_profit: Optional[float] = None
    total_expenses: Optional[float] = None
    average_order_value: Optional[float] = None
    total_customers: Optional[int] = None
    total_products: Optional[int] = None


class AnalysisResponse(BaseModel):
    """Final AI-generated response to a user's business question."""

    answer: str = Field(description="Direct, business-friendly answer to the user's question")
    explanation: str = Field(description="Explanation of how the numbers support the answer")
    recommendation: str = Field(description="One or two actionable recommendations based on this result")
