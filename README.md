# Azure Capacity Checker

A Streamlit tool that validates Azure VM SKU availability against live Azure data, tests real deployment capacity, and recommends verified alternatives when SKUs are blocked or exhausted.

Built for cloud migration teams working with large rightsizing exports (1000+ servers) who need to know — before migration day — which VMs will actually deploy.

## What It Does

1. **Upload** a rightsizing export (Excel from Azure Migrate / Dr Migrate, or CSV/JSON)
2. **Catalogue check** — queries the Azure Resource SKUs API to verify every VM size is available in your target region and subscription
3. **Live capacity check** _(optional)_ — submits ARM deployment validation requests to test whether physical hardware is actually available right now (no VMs are created)
4. **Alternative recommendations** — when a SKU is blocked or exhausted, scores and ranks alternatives by family similarity, vCPU/memory match, generation, and disk support
5. **Pricing comparison** — fetches PAYG, 1-year RI, and 3-year RI pricing from the Azure Retail Prices API for both current and alternative SKUs
6. **Export** — download an updated output with blocked SKUs swapped for verified alternatives (mirrored rightsizing Excel format for Excel inputs, flat `.xlsx` for CSV/JSON inputs)

## Quick Start

```bash
# Clone
git clone https://github.com/adamswbrown/AzureCapacityChecker.git
cd AzureCapacityChecker

# Install dependencies
pip install -r requirements.txt

# Run
streamlit run app/app.py
```

The app opens at `http://localhost:8501`.

## Authentication

The tool needs read access to the Azure Resource SKUs API. Four auth methods are supported:

| Method | When to use |
|---|---|
| **Default (CLI / Managed Identity / Env Vars)** | Local development or pre-configured runtime identities |
| **Device Code (Browser Login)** | Recommended for hosted deployments (including Streamlit Cloud) |
| **Browser (Interactive)** | Local development only (requires the app host to open a browser window) |
| **Service Principal** | CI/CD or shared environments — provide tenant, client ID, and secret |

If you're running this in a hosted environment, `Default` often fails unless credentials are preconfigured, and `Interactive Browser` typically fails because the server cannot open a local browser. Use `Device Code` or `Service Principal`.

In hosted Streamlit deployments, the auth dropdown defaults to **Device Code** automatically. You can override this with `AZURE_AUTH_DEFAULT=device_code` or `AZURE_AUTH_DEFAULT=default`.

The live capacity check also requires **Contributor** access to a resource group (it creates and validates ARM deployments without actually provisioning VMs).

### Required Azure permissions

- `Microsoft.Compute/skus/read` — SKU catalogue
- `Microsoft.Resources/deployments/validate/action` — capacity validation (optional)
- `Microsoft.Resources/resourceGroups/write` — auto-create the validation resource group (optional)

## Input Formats

Supported upload file types: `.xlsx`, `.xls`, `.csv`, `.json`.

### Excel (Azure Migrate / Dr Migrate export)

The standard format. Expects a **Servers** sheet with headers on row 6 and columns including:

| Column | Maps to |
|---|---|
| Server | Machine name |
| Target Azure Region | Deployment region |
| Chosen SKU | The VM size to validate |
| Current Cores | vCPU count |
| Current RAM (MB) | Memory |
| Chosen Compute Cost Monthly | Current PAYG cost (used as baseline for cost deltas) |
| Chosen 1 Year RI Cost Monthly | Current 1yr RI cost |
| Chosen 3 Year RI Cost Monthly | Current 3yr RI cost |

A **Disks** sheet (also headers on row 6) is parsed if present.

### CSV / JSON (alternate flat format)

Use this when you're not uploading a full Azure Migrate / Dr Migrate rightsizing export.

Required columns:

- `MachineName`
- `Region`
- `RecommendedSKU`

Optional columns:

- `vCPU`
- `MemoryGB`
- `VMFamily`

The parser also accepts common header variants (for example `machine_name`, `server`, `location`, `vm_size`).

See:

- `data/example_dataset.csv`
- `data/example_dataset.json`

## How the Analysis Works

### Phase 1: Catalogue Check

Queries `Microsoft.Compute/skus` for your subscription. Each SKU is classified:

- **OK** — available, no restrictions
- **RISK** — available but not in all availability zones
- **BLOCKED** — not offered in the region, restricted on subscription, or doesn't exist

### Phase 2: Live Capacity Validation

For SKUs that passed the catalogue check, submits ARM `deployments/validate` requests. This tests real physical capacity without creating any resources. Results:

- **Capacity Verified** — hardware is available right now
- **Capacity Failed** — catalogue says OK but no physical hardware (quota exhaustion, regional sellout)

### Phase 3: Alternative Scoring

Blocked and capacity-failed SKUs are scored against all available SKUs using a weighted algorithm:

| Factor | Weight | Logic |
|---|---|---|
| VM Family match | 35 | Same family = full score |
| vCPU match | 25 | Exact = 25, within 2x = partial |
| Memory match | 15 | Exact = 15, within 2x = partial |
| Generation | 15 | Same or newer gen preferred |
| Size tier | 5 | Penalise large jumps |
| Constrained bonus | 5 | Constrained variants when disk count is low |

Alternatives are capacity-checked too (Phase 2 runs on them), so the top recommendation is the best-scoring SKU with verified capacity.

### Phase 4: Pricing

Fetches retail pricing from the free [Azure Retail Prices API](https://learn.microsoft.com/en-us/rest/api/cost-management/retail-prices/azure-retail-prices) (no auth required):

- **Current SKU cost**: uses the Excel assessment data when available, falls back to API
- **Alternative SKU cost**: PAYG, 1-year RI, 3-year RI from the API
- **Cost deltas**: shown inline on server rows and in the alternatives detail table

## UI Overview

### Executive Summary

Donut chart showing deployment readiness breakdown, key metrics (total servers, vCPUs, memory), and a region/status heatmap.

### Grouped View

Servers grouped by VM family. Each family shows outcome-focused pills:

- **deploy** — capacity verified, ready to go
- **catalogue OK** — in the catalogue but not live-tested
- **zone limited** — available but not in all AZs
- **need alternative** — blocked or no capacity, with count of verified alternatives

Click any server to expand its detail panel showing the full assessment, alternatives comparison table (with pricing across all three tiers), and attached disks.

### Export

Two download options:

- **Updated rightsizing export (.xlsx)** — same format as the input file with the "Chosen SKU" column updated to the best alternative where needed, plus advisory columns (Original SKU, Capacity Status, Verdict, Reason)
- **Full analysis CSV** — flat export with all analysis data including pricing

If the input file is CSV/JSON (or mirrored Excel reconstruction fails), the `.xlsx` download is a flat sheet with:
`Server`, `Target Azure Region`, `Chosen SKU`, `Original SKU`, `Capacity Status`, `Verdict`, and `Reason`.

## Project Structure

```
├── app/
│   ├── app.py              # Streamlit UI (main entry point)
│   └── config.py           # Regions, limits, display names
├── azure_client/
│   ├── auth.py             # Azure authentication (CLI, browser, SP)
│   ├── deployment_validator.py  # ARM deployment validation
│   ├── pricing_service.py  # Azure Retail Prices API client
│   └── sku_service.py      # Resource SKUs API client
├── engine/
│   ├── alternatives.py     # Alternative SKU scoring algorithm
│   ├── analyzer.py         # Main analysis orchestrator
│   ├── capacity_validator.py  # Two-phase capacity validation
│   └── disk_analyzer.py    # Disk SKU analysis
├── models/
│   ├── disk.py             # Disk data model
│   ├── machine.py          # Machine data model + SKU family parser
│   └── result.py           # Analysis result + summary models
├── parsers/
│   └── dataset_parser.py   # Excel/CSV/JSON input parser
├── data/
│   ├── example_dataset.csv
│   └── example_dataset.json
├── .streamlit/
│   └── config.toml         # Dark theme configuration
└── requirements.txt
```

## Requirements

- Python 3.10+
- An Azure subscription with access to the target regions
- `az login` or a service principal for authentication

## License

MIT
