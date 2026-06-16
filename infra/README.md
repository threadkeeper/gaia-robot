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
  deployed (SKU from `modelRouterSku`; default `auto`) and, when
  `deployEmbeddingModel` is `true`, a high-performance **`text-embedding`**
  deployment used to populate the Cosmos `/dataVector` field;
- a **Container App** (plus its managed environment and a Log Analytics
  workspace), pre-wired with `COSMOS_ENDPOINT`, `COSMOS_DATABASE`,
  `FOUNDRY_ENDPOINT`, `MODEL_ROUTER_DEPLOYMENT`, `EMBEDDING_DEPLOYMENT`, and
  `EMBEDDING_DIMENSIONS` environment variables.

Template outputs include `tier`, `cosmosEndpoint` (use it for `COSMOS_ENDPOINT`),
`foundryEndpoint`, `modelRouterDeployment`, `embeddingDeployment`,
`embeddingDimensions`, and `containerAppFqdn`.

Key template parameters: `tier` (default `free`), `modelRouterVersion`
(default `2025-05-19`), `modelRouterSku` (default `auto`: uses `Standard` in
`eastus`, `GlobalStandard` elsewhere), `modelRouterCapacity` (default `0` = use
the tier default), `deployEmbeddingModel` (default `false`), `embeddingModel`
(default `text-embedding-3-large`), `embeddingDimensions` (default `1536` — must
match `COSMOS_VECTOR_DIMS`), `containerImage` (default a hello-world image —
replace with your own once published).

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
| **`model-router` deployment** (`modelRouterSku`) | ⚠️ Pay-per-token (no idle cost) | Per 1K input/output tokens (varies by routed model) | Consumption-based; **cannot be made free**, but costs nothing when not called. Capacity is a rate limit (TPM), not a charge. |
| **Container App** | ✅ Free when scaled to zero | Free monthly grant: 180K vCPU-s + 360K GiB-s + 2M requests, then ~$0.000024/vCPU-s | **Lite** uses `minReplicas: 0` → idles at $0. **Hobby** uses `minReplicas: 1` (always-on) which exceeds the free grant and costs ~$15–30/mo. |
| **Container Apps managed environment** | ✅ Free | $0 | No charge for the environment itself; you pay only for the apps. |
| **Log Analytics workspace** (`PerGB2018`) | ✅ First 5 GB/mo free | ~$2.76/GB after 5 GB | Retention is set to **30 days** to stay within Azure's supported workspace limits while keeping logs available for testing. |

**Bottom line:** the **Free / Lite** button has **no fixed monthly cost** while
idle. The only unavoidable usage-based charges are (1) AI model tokens when the
`model-router` is actually called, and (2) Cosmos throughput/storage beyond the
free-tier allowances. Everything else sits within Azure's free grants.

## cosmos_create.py

Provisions the six Cosmos DB for NoSQL containers from the Gaia physical
architecture diagram. The script is idempotent (safe to re-run): it creates the
database and each container only if they do not already exist.

| Container         | Partition / business key | Business key (uniqueness) |
|-------------------|--------------------------|---------------------------|
| `GaiaKB`          | `entity`                 | entity + date             |
| `GaiaLH`          | `entity`                 | entity + date             |
| `UsersKB`         | `userId`                 | userId + date             |
| `UsersDL`         | `userId`                 | userId + date             |
| `GaiaCosmos`      | `entity`                 | entity + date             |
| `GaiaConnections` | `entity`                 | entity + timestamp        |

Each container is created with:

- **Partition key** on the business key field (`/entity` or `/userId`) so vector
  and text queries can be filtered cheaply by entity / user.
- **Unique-key policy** on the container's uniqueness field — `/date` for the
  daily-snapshot containers (one record per partition per `yyyy-mm-dd`), and
  `/timestamp` for the `GaiaConnections` ledger (one record per partition per
  ISO 8601 instant, so a single day can hold many ledger entries).
- **Vector embedding policy** on `/dataVector` — the vector representation of the
  record's `data` field — plus a **DiskANN vector index** on that path.
- **Full-text index** on the entity / userId text field (the additional text
  index), in addition to the partition-key filtering.

### Gaia Connections (emotional bank account)

`GaiaConnections` is a **ledger**: LLM Call 1 judges whether each user turn grows
or weakens the friendship and posts a **signed change** to a per-entity running
balance (Gaia's "emotional bank account"). Each ledger record carries:

| Field             | Meaning                                                |
|-------------------|--------------------------------------------------------|
| `entity`          | The entity / user the balance is tracked for (key).    |
| `timestamp`       | ISO 8601 instant of the change (uniqueness within key).|
| `changeAmount`    | Signed delta applied this turn (`+` gain, `-` loss).   |
| `previousBalance` | Balance before this change.                            |
| `newBalance`      | Balance after this change (`previousBalance + change`).|
| `notes`           | Why the change was made.                               |

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
| `GITHUB_TOKEN`       | no       | —       | GitHub token for local/dev mode  |

## embed_worker.py (vector backfill)

`cosmos_create.py` declares the `/dataVector` vector policy, but the migration
scripts do not compute embeddings. `embed_worker.py` fills that gap: it scans
**every** container for records whose `/dataVector` is missing or empty, embeds
their `data` text with the Foundry **`text-embedding`** deployment in batches,
and writes the vectors back with the Cosmos **patch** API — concurrently and
idempotently (re-running only picks up whatever is still missing).

The embedding logic lives in [embeddings.py](embeddings.py) as a reusable
`EmbeddingClient` (with `embed`, `embed_one`, and a lazy `embed_if_missing`).

### Run it

```powershell
cd infra
.venv\Scripts\Activate.ps1
pip install -r requirements.txt        # adds openai + aiohttp

# .env must set COSMOS_ENDPOINT, FOUNDRY_ENDPOINT and EMBEDDING_DEPLOYMENT.
python embed_worker.py                 # backfill every container
python embed_worker.py --dry-run       # count only, no model calls
python embed_worker.py --container GaiaKB --batch-size 32 --concurrency 8 --max 500
```

| Variable                   | Required | Default      | Purpose                                   |
|----------------------------|----------|--------------|-------------------------------------------|
| `FOUNDRY_ENDPOINT`         | yes      | —            | Foundry account endpoint                  |
| `EMBEDDING_DEPLOYMENT`     | yes      | —            | Embedding deployment name                 |
| `EMBEDDING_DIMENSIONS`     | no       | `1536`       | Vector length (must match the Cosmos policy) |
| `FOUNDRY_KEY`              | no       | —            | API key (else Azure AD is used)           |
| `AZURE_OPENAI_API_VERSION` | no       | `2024-10-21` | Azure OpenAI REST API version             |

> The embedding vector length **must** equal `COSMOS_VECTOR_DIMS` (the value the
> containers were created with), or Cosmos will reject the vector.
