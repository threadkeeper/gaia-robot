"""Create the six Gaia Cosmos DB containers described in the architecture diagram.

This is an infrastructure provisioning *script* (not part of the Rust program).
Running it is idempotent: it creates the database and each container only if they
do not already exist, so it is safe to re-run.

It provisions six Azure Cosmos DB for NoSQL containers:

    Container         Business key field   Business key (uniqueness)
    ---------------   ------------------   -------------------------
    GaiaKB            entity               entity + date (yyyy-mm-dd)
    GaiaLH            entity               entity + date (yyyy-mm-dd)
    UsersKB           userId               userId + date (yyyy-mm-dd)
    UsersDL           userId               userId + date (yyyy-mm-dd)
    GaiaCosmos        entity               entity + date (yyyy-mm-dd)
    GaiaConnections   entity               entity + timestamp (ISO 8601)

``GaiaConnections`` is the emotional-bank-account *ledger*: LLM Call 1 decides
whether each user turn grows or weakens the friendship and posts a signed change
to a per-entity running balance. Because a ledger holds many entries per day, its
uniqueness is per *timestamp* (a full ISO 8601 instant) rather than per day.

Each container is configured with:

  * Partition key on the business key field (``/entity`` or ``/userId``). This
    is what lets vector and text queries be *filtered* cheaply by entity/user.
  * A unique-key policy on the container's uniqueness field (``/date`` for the
    daily-snapshot containers, ``/timestamp`` for the ledger) so there is at most
    one record per partition per that field.
  * A vector embedding policy on ``/dataVector`` -- the vector representation of
    the record's ``data`` field -- plus a DiskANN vector index on that path so
    the data can be searched by similarity.
  * A full-text index on the business key field so the entity/userId text can be
    searched in addition to being filtered.

Prerequisites (control-plane, done once, not by this data-plane script):
  * An Azure Cosmos DB for NoSQL account.
  * The account must have these capabilities enabled:
      - "EnableNoSQLVectorSearch"  (vector indexing)
      - "EnableNoSQLFullTextSearch" (full-text indexing)
    e.g. via:
      az cosmosdb update -n <account> -g <rg> \
        --capabilities EnableNoSQLVectorSearch EnableNoSQLFullTextSearch

Authentication (pick one, checked in this order):
  * COSMOS_KEY env var  -> uses the account key.
  * otherwise           -> uses Azure AD via DefaultAzureCredential
                           (recommended; e.g. `az login` or a managed identity).

Required environment variables:
  COSMOS_ENDPOINT        e.g. https://my-account.documents.azure.com:443/
Optional environment variables:
  COSMOS_KEY             account key (if omitted, Azure AD is used)
  COSMOS_DATABASE        database name (default: "gaia")
  COSMOS_VECTOR_DIMS     embedding dimensions (default: 1536)
  COSMOS_THROUGHPUT      RU/s per container (default: 400)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from azure.cosmos import CosmosClient, PartitionKey, exceptions

# --- .env loading ------------------------------------------------------------


def load_env_file(path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from a .env file into os.environ.

    Kept dependency-free (no python-dotenv) on purpose. Existing environment
    variables always win, so real env vars override the file. Lines that are
    blank or start with '#' are ignored, and surrounding quotes are stripped.
    """
    if not os.path.exists(path):
        return

    with open(path, encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            # Skip blanks and comments.
            if not line or line.startswith("#"):
                continue
            # Only split on the first '=' so values may contain '='.
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            # Don't clobber variables already set in the real environment.
            os.environ.setdefault(key, value)


# --- Configuration -----------------------------------------------------------


# The six containers from the diagram. ``key_field`` is the business/partition
# key: the entity tables key on "entity", the user tables key on "userId".
# ``unique_field`` is the path made unique *within* a partition; it is "date"
# for the daily-snapshot containers and "timestamp" for the connections ledger
# (a ledger needs many rows per day, so it keys on a full instant, not a day).
@dataclass(frozen=True)
class TableSpec:
    """Describes one Cosmos container to create."""

    name: str                  # container id, e.g. "GaiaKB"
    key_field: str             # business key / partition key field, e.g. "entity"
    unique_field: str = "date"  # per-partition unique path, e.g. "date" or "timestamp"


TABLES: list[TableSpec] = [
    TableSpec(name="GaiaKB", key_field="entity"),
    TableSpec(name="GaiaLH", key_field="entity"),
    TableSpec(name="UsersKB", key_field="userId"),
    TableSpec(name="UsersDL", key_field="userId"),
    TableSpec(name="GaiaCosmos", key_field="entity"),
    # Emotional-bank-account ledger: one record per (entity, timestamp).
    TableSpec(name="GaiaConnections", key_field="entity", unique_field="timestamp"),
]

# Path of the vector field. Every record stores the embedding of its ``data``
# field here; the container's policies declare and index this path.
VECTOR_PATH = "/dataVector"


# --- Policy builders ---------------------------------------------------------
# These are small pure functions so each policy is easy to read and test.


def build_vector_embedding_policy(dimensions: int) -> dict:
    """Vector embedding policy: declares ``/dataVector`` as the data's embedding.

    Cosine distance is the usual choice for text-embedding similarity.
    """
    return {
        "vectorEmbeddings": [
            {
                "path": VECTOR_PATH,
                "dataType": "float32",
                "distanceFunction": "cosine",
                "dimensions": dimensions,
            }
        ]
    }


def build_full_text_policy(key_field: str) -> dict:
    """Full-text policy: enables text search over the entity/userId field."""
    return {
        "defaultLanguage": "en-US",
        "fullTextPaths": [
            {"path": f"/{key_field}", "language": "en-US"},
        ],
    }


def build_indexing_policy(key_field: str) -> dict:
    """Indexing policy combining the standard, vector, and full-text indexes.

    * The vector path is excluded from normal indexing (vectors are indexed only
      by the dedicated DiskANN vector index).
    * ``vectorIndexes`` indexes ``/dataVector`` for similarity search; combined
      with the partition key it can be filtered by entity/userId.
    * ``fullTextIndexes`` adds the extra text index on the entity/userId field.
    """
    return {
        "indexingMode": "consistent",
        "automatic": True,
        "includedPaths": [{"path": "/*"}],
        "excludedPaths": [
            {"path": f"{VECTOR_PATH}/*"},
            {"path": '/"_etag"/?'},
        ],
        "vectorIndexes": [
            {"path": VECTOR_PATH, "type": "diskANN"},
        ],
        "fullTextIndexes": [
            {"path": f"/{key_field}"},
        ],
    }


def build_unique_key_policy(unique_field: str) -> dict:
    """Unique-key policy enforcing one record per partition per ``unique_field``.

    Uniqueness is scoped to the logical partition (the entity/userId), so this
    gives the business key, e.g. entity+date, userId+date, or entity+timestamp
    for the connections ledger.
    """
    return {"uniqueKeys": [{"paths": [f"/{unique_field}"]}]}


# --- Cosmos client -----------------------------------------------------------


def get_client(endpoint: str) -> CosmosClient:
    """Create a CosmosClient using an account key if present, else Azure AD."""
    key = os.environ.get("COSMOS_KEY")
    if key:
        # Key auth: simplest, but prefer Azure AD where possible.
        return CosmosClient(url=endpoint, credential=key)

    # Azure AD auth (recommended). Imported lazily so the azure-identity
    # dependency is only needed when actually using AAD.
    from azure.identity import DefaultAzureCredential

    return CosmosClient(url=endpoint, credential=DefaultAzureCredential())


def create_container(database, table: TableSpec, dimensions: int, throughput: int) -> None:
    """Create one container (if it does not already exist) with all policies."""
    print(f"  creating container '{table.name}' (key: {table.key_field}) ...")
    database.create_container_if_not_exists(
        id=table.name,
        # Partition key = business key field -> enables cheap filtering.
        partition_key=PartitionKey(path=f"/{table.key_field}"),
        indexing_policy=build_indexing_policy(table.key_field),
        vector_embedding_policy=build_vector_embedding_policy(dimensions),
        full_text_policy=build_full_text_policy(table.key_field),
        unique_key_policy=build_unique_key_policy(table.unique_field),
        offer_throughput=throughput,
    )
    print(f"  done: '{table.name}'")


# --- Entry point -------------------------------------------------------------


def main() -> int:
    """Provision the database and all six containers. Returns a process exit code."""
    # 0. Load .env (if present) so local config is picked up automatically.
    load_env_file()

    # 1. Read configuration from the environment.
    endpoint = os.environ.get("COSMOS_ENDPOINT")
    if not endpoint:
        print("ERROR: set COSMOS_ENDPOINT to your Cosmos account URL.", file=sys.stderr)
        return 1

    database_name = os.environ.get("COSMOS_DATABASE", "gaia")
    dimensions = int(os.environ.get("COSMOS_VECTOR_DIMS", "1536"))
    throughput = int(os.environ.get("COSMOS_THROUGHPUT", "400"))

    # 2. Connect and ensure the database exists.
    print(f"Connecting to {endpoint} ...")
    client = get_client(endpoint)

    try:
        print(f"Ensuring database '{database_name}' exists ...")
        database = client.create_database_if_not_exists(id=database_name)

        # 3. Create each container from the spec table.
        for table in TABLES:
            create_container(database, table, dimensions, throughput)
    except exceptions.CosmosHttpResponseError as error:
        # Most likely cause: the account is missing the vector/full-text
        # capabilities, or the credential lacks data-plane permissions.
        print(f"ERROR: Cosmos request failed: {error.message}", file=sys.stderr)
        return 1

    print(f"\nAll {len(TABLES)} containers are ready in database '{database_name}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
