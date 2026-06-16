# infra/

Infrastructure, deployment, and environment configuration.

## Deploy to Azure

Pick a cost tier:

| Tier | Button | What you get |
|------|--------|--------------|
| **Free / Lite** | [![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fthreadkeeper%2Fgaia-robot%2Fmain%2Finfra%2Fazuredeploy.free.json) | Cosmos **free tier**, **scale-to-zero** Container App (0.25 vCPU / 0.5 GiB, 0–1 replicas), smallest model capacity (10). Lowest cost — best for trying Gaia out. |
| **Hobby** | [![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fthreadkeeper%2Fgaia-robot%2Fmain%2Finfra%2Fazuredeploy.hobby.json) | Always-on Container App (0.5 vCPU / 1 GiB, 1–3 replicas), mid-size model capacity (50). |

Both buttons nest [azuredeploy.json](azuredeploy.json), the base ARM template
(deploy it directly to pick a custom `tier`/parameters). It provisions:

- the **Cosmos DB for NoSQL** account (with `EnableNoSQLVectorSearch` and
  `EnableNoSQLFullTextSearch`) and the `gaia` database;
- an **Azure AI Foundry** account and project, with the **`model-router`** model
  deployed (`GlobalStandard`);
- a **Container App** (plus its managed environment and a Log Analytics
  workspace), pre-wired with `COSMOS_ENDPOINT`, `COSMOS_DATABASE`,
  `FOUNDRY_ENDPOINT`, and `MODEL_ROUTER_DEPLOYMENT` environment variables.

Template outputs include `tier`, `cosmosEndpoint` (use it for `COSMOS_ENDPOINT`),
`foundryEndpoint`, `modelRouterDeployment`, and `containerAppFqdn`.

Key template parameters: `tier` (default `free`), `modelRouterVersion`
(default `2025-05-19`), `modelRouterCapacity` (default `0` = use the tier
default), `containerImage` (default a hello-world image — replace with your own
once published).

## Cost / pricing breakdown

> Azure prices vary by region and change over time. The figures below are
> approximate USD pay-as-you-go rates for guidance only — always confirm with
> the [Azure Pricing Calculator](https://azure.microsoft.com/pricing/calculator/).
> The **Free / Lite** tier is designed to have **no fixed idle cost**: you only
> pay for what you actually use (model tokens and any Cosmos throughput/storage
> beyond the free allowances).

| Resource | Free tier (Lite) | Pay-as-you-go cost | Notes |
|----------|------------------|--------------------|-------|
| **Cosmos DB for NoSQL** account | ✅ `enableFreeTier` on (both tiers) | ~$0.008 / 100 RU/s per hour + ~$0.25/GB-month | Free tier covers the **first 1,000 RU/s + 25 GB** per **subscription** (one free-tier account max). `cosmos_create.py` makes 5 containers at 400 RU/s each (2,000 RU/s) — that **exceeds** the free 1,000 RU/s, so ~1,000 RU/s is billable. Lower `COSMOS_THROUGHPUT` (e.g. 200) or use shared database throughput to stay free. |
| **Azure AI Foundry** account (`AIServices`, S0) | ✅ No fixed/hourly fee | $0 idle | You are billed only per model token, not for the account itself. |
| **`model-router` deployment** (`GlobalStandard`) | ⚠️ Pay-per-token (no idle cost) | Per 1K input/output tokens (varies by routed model) | Consumption-based; **cannot be made free**, but costs nothing when not called. Capacity is a rate limit (TPM), not a charge. |
| **Container App** | ✅ Free when scaled to zero | Free monthly grant: 180K vCPU-s + 360K GiB-s + 2M requests, then ~$0.000024/vCPU-s | **Lite** uses `minReplicas: 0` → idles at $0. **Hobby** uses `minReplicas: 1` (always-on) which exceeds the free grant and costs ~$15–30/mo. |
| **Container Apps managed environment** | ✅ Free | $0 | No charge for the environment itself; you pay only for the apps. |
| **Log Analytics workspace** (`PerGB2018`) | ✅ First 5 GB/mo free | ~$2.76/GB after 5 GB | Retention is set to **4 days** (the minimum) to minimise stored-data cost. At Gaia's low log volume this is effectively free. |

**Bottom line:** the **Free / Lite** button has **no fixed monthly cost** while
idle. The only unavoidable usage-based charges are (1) AI model tokens when the
`model-router` is actually called, and (2) Cosmos throughput/storage beyond the
free-tier allowances. Everything else sits within Azure's free grants.

## cosmos_create.py

Provisions the five Cosmos DB for NoSQL containers from the Gaia physical
architecture diagram. The script is idempotent (safe to re-run): it creates the
database and each container only if they do not already exist.

| Container    | Partition / business key | Business key (uniqueness) |
|--------------|--------------------------|---------------------------|
| `GaiaKB`     | `entity`                 | entity + date             |
| `GaiaLH`     | `entity`                 | entity + date             |
| `UsersKB`    | `userId`                 | userId + date             |
| `UsersDL`    | `userId`                 | userId + date             |
| `GaiaCosmos` | `entity`                 | entity + date             |

Each container is created with:

- **Partition key** on the business key field (`/entity` or `/userId`) so vector
  and text queries can be filtered cheaply by entity / user.
- **Unique-key policy** on `/date`, giving the business key entity+date /
  userId+date (one record per partition per `yyyy-mm-dd`).
- **Vector embedding policy** on `/dataVector` — the vector representation of the
  record's `data` field — plus a **DiskANN vector index** on that path.
- **Full-text index** on the entity / userId text field (the additional text
  index), in addition to the partition-key filtering.

### Account prerequisites (one-time, control plane)

The target Cosmos account must exist and have these capabilities enabled:

```sh
az cosmosdb update -n <account> -g <resource-group> \
  --capabilities EnableNoSQLVectorSearch EnableNoSQLFullTextSearch
```

### Run it

```powershell
cd infra
python -m venv .venv
.venv\Scripts\Activate.ps1          # PowerShell (Windows)
pip install -r requirements.txt

# Configure: copy the sample and fill in your values.
Copy-Item .env.sample .env
# Edit .env (at minimum set COSMOS_ENDPOINT). The script loads .env automatically.

python cosmos_create.py
```

The script reads configuration from `infra/.env` (git-ignored) if present;
real environment variables override the file. Use [.env.sample](.env.sample) as
the template.

### Configuration (environment variables)

| Variable             | Required | Default | Purpose                          |
|----------------------|----------|---------|----------------------------------|
| `COSMOS_ENDPOINT`    | yes      | —       | Cosmos account URL               |
| `COSMOS_KEY`         | no       | —       | Account key (else Azure AD used) |
| `COSMOS_DATABASE`    | no       | `gaia`  | Database name                    |
| `COSMOS_VECTOR_DIMS` | no       | `1536`  | Embedding dimensions             |
| `COSMOS_THROUGHPUT`  | no       | `400`   | RU/s per container               |
