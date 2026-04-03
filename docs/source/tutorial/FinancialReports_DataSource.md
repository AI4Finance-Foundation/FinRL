# Data Source Suggestion: FinancialReports.eu for RL Trading Signals

## Overview

[FinancialReports.eu](https://financialreports.eu) provides API access to **14M+ regulatory filings** from 35 official sources across 30+ countries. Filing events (annual reports, M&A announcements, ESG disclosures) are known alpha signals that can enhance RL agent feature sets for international trading strategies.

## Why This Fits FinRL

FinRL agents learn from market data features. FinancialReports.eu adds:

- **Filing event signals** — annual report releases, interim results, M&A announcements as timestamped events
- **11 standardized filing categories** — Financial Reporting, ESG, M&A/Partnerships/Legal, Debt/Equity Info, etc. — usable as categorical features
- **Non-US market coverage** — 35 regulators across 30+ countries for international trading strategies
- **33,000+ companies** with ISIN identifiers for cross-referencing with price data
- **Markdown endpoint** — extract filing text for NLP-based feature engineering

## Integration Approaches

### 1. Filing Events as RL Features

```python
import requests
import pandas as pd

headers = {"X-API-Key": "your-api-key"}

# Fetch filing events for a company (e.g., Siemens)
resp = requests.get("https://api.financialreports.eu/filings/",
    headers=headers,
    params={
        "company_isin": "DE0007236101",  # Siemens ISIN
        "categories": "2",               # Financial Reporting
        "page_size": 50
    }
)
filings = resp.json()["results"]

# Convert to DataFrame for RL feature engineering
df = pd.DataFrame([{
    "date": f["release_datetime"],
    "category": f["filing_type"]["category"]["name"],
    "type": f["filing_type"]["name"],
} for f in filings])

# Use as event features alongside price data in RL environment
```

### 2. MCP Server for AI Agent Integration

FinancialReports.eu offers an [MCP server](https://financialreports.eu) compatible with Claude.ai and other AI platforms, enabling RL agents to query filings interactively.

### 3. Python SDK

```bash
pip install financial-reports-generated-client
```

```python
from financial_reports_client import Client
from financial_reports_client.api.filings import filings_list

client = Client(base_url="https://api.financialreports.eu")
client = client.with_headers({"X-API-Key": "your-api-key"})

filings = filings_list.sync(client=client, company_isin="DE0007236101", categories="2")
```

## API Details

| Property | Value |
|---|---|
| **Base URL** | `https://api.financialreports.eu` |
| **API Docs** | [docs.financialreports.eu](https://docs.financialreports.eu/) |
| **Authentication** | API key via `X-API-Key` header |
| **Python SDK** | `pip install financial-reports-generated-client` |
| **Rate Limiting** | Burst limit + monthly quota |
| **Companies** | 33,230+ |
| **Total Filings** | 14,135,359+ |
| **Sources** | 35 official regulators |

## Feature Engineering Ideas

| Filing Signal | RL Feature |
|---|---|
| Annual report release date | Binary event indicator |
| Filing category (11 types) | One-hot encoded categorical |
| Filing frequency changes | Anomaly detection signal |
| M&A announcements | Merger/acquisition event flag |
| ESG disclosures | ESG sentiment score (via Markdown + NLP) |
| Multi-country filings | Cross-market correlation features |
