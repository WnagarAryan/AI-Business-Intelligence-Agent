from langchain_core.prompts import PromptTemplate

summary_template = """You are an expert Business Intelligence Analyst.
Look at the dataset description below and produce a structured overview
following the DatasetSummary schema.

Dataset type (auto-detected): {dataset_type}
Columns: {columns}
Row count: {row_count}
Missing values by column: {missing_values}
Duplicate rows: {duplicate_rows}
Sample rows (first 5, as JSON):
{sample_rows}

Business metrics already calculated (by Pandas):
{metrics}

Explain in plain English what this dataset contains, note any data quality
issues, and suggest a few useful business questions the user could ask.
"""

summary_prompt = PromptTemplate(
    template=summary_template,
    input_variables=[
        "dataset_type",
        "columns",
        "row_count",
        "missing_values",
        "duplicate_rows",
        "sample_rows",
        "metrics",
    ],
)

qa_template = """You are an expert Business Intelligence Analyst.
Pandas has already performed the calculation below. Turn it into a
business-friendly response following the AnalysisResponse schema.

User question:
{question}

Analysis performed: {analysis_type}
Raw result (JSON, calculated by Pandas):
{result_json}

Give a direct answer, a short explanation of how the numbers support it,
and one or two actionable recommendations.
"""

qa_prompt = PromptTemplate(
    template=qa_template,
    input_variables=["question", "analysis_type", "result_json"],
)


recommendation_template = """You are an expert Business Intelligence Consultant.
The user wants general recommendations for their business, based on the
metrics below. Respond following the AnalysisResponse schema.

User question:
{question}

Dataset summary:
{dataset_summary}

Business metrics (calculated by Pandas):
{metrics}

Give a short answer, a brief explanation grounded in the metrics above,
and one or two concrete, actionable recommendations.
"""

recommendation_prompt = PromptTemplate(
    template=recommendation_template,
    input_variables=["question", "dataset_summary", "metrics"],
)
