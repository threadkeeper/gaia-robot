# infra/

Infrastructure, deployment, and environment configuration.

## Deploy to Azure

[![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Fthreadkeeper%2Fgaia-robot%2Fmain%2Finfra%2Fazuredeploy.json)

[azuredeploy.json](azuredeploy.json) is the ARM template behind the button. It
provisions:

- the **Cosmos DB for NoSQL** account (with `EnableNoSQLVectorSearch` and
  `EnableNoSQLFullTextSearch`) and the `gaia` database;
- an **Azure AI Foundry** account and project, with the **`model-router`** model
  deployed (`GlobalStandard`);
- a **Container App** (plus its managed environment and a Log Analytics
  workspace), pre-wired with `COSMOS_ENDPOINT`, `COSMOS_DATABASE`,
  `FOUNDRY_ENDPOINT`, and `MODEL_ROUTER_DEPLOYMENT` environment variables.

Template outputs include `cosmosEndpoint` (use it for `COSMOS_ENDPOINT`),
`foundryEndpoint`, `modelRouterDeployment`, and `containerAppFqdn`.

Key template parameters: `modelRouterVersion` (default `2025-05-19`),
`modelRouterCapacity` (default `50`), `containerImage` (default a hello-world
image — replace with your own once published).

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
