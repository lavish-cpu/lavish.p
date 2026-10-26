# Azure DataOps — Data Quality Pipeline

A local-first ETL starter that creates sample sales records, validates them, standardizes columns, and writes a curated CSV. The README also outlines how to map the stages to Azure Data Factory, Azure Data Lake Storage, and Azure Databricks.

## Run

    python -m venv .venv
    # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    python pipeline.py

Output files are written to `data/`. No Azure credentials are required for the local demo.

## Azure extension
1. Land raw files in an ADLS Gen2 raw container.
2. Trigger an ADF pipeline on file arrival or schedule.
3. Run the same validation/transformation logic in Databricks with PySpark.
4. Write curated Delta tables and monitor failures with Azure Monitor.
5. Store secrets in Key Vault; use managed identity and least-privilege access.
