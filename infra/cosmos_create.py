"""Create the six Gaia Cosmos DB containers described in the architecture diagram.

This is an infrastructure provisioning *script* (not part of the Rust program).
Running it is idempotent: it creates the database and each container only if they
do not already exist, so it is safe to re-run.

It provisions the Azure Cosmos DB for NoSQL containers below:

    Container              Business key field   Business key (uniqueness)
    --------------------   ------------------   -------------------------
    GaiaKB                 entity               entity + date (yyyy-mm-dd)
    GaiaDataLake           entity               entity + date (yyyy-mm-dd)
    GaiaDiary              entity               entity + date (yyyy-mm-dd)
    GaiaWebSearchHistory   entity               entity + timestamp (ISO 8601)
    GaiaConnections        entity               entity + timestamp (ISO 8601)
    DataLakeIndex          entity               entity + date (yyyy-mm-dd)

``DataLakeIndex`` is a derived semantic index over ``GaiaDataLake``. Each entry
stores only routing fields (source id, entity, day, source container) and a
*half-size* 768-dimension vector at ``/indexVector`` -- not the shared
``/dataVector`` -- with no ``/data`` payload. The Rust write path recreates it on
first write if it is missing; provisioning it here keeps a fresh account whole.

``GaiaWebSearchHistory`` is an append-only *log* of Gaia's web searches: it
stores the search query and its results rather than a ``/data`` text payload, so
it has a vector index (over the embedded query text) but no ``/data`` full-text
index. Like the ledger below, it can hold many entries per day, so its
uniqueness is per *timestamp*.

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
    the record's text content -- plus a DiskANN vector index on that path so
    every container can be searched by similarity.
  * Two composite (regular) indexes pairing the business key with the
    uniqueness field in *both* orders -- ``(entity|userId, date)`` and
    ``(date, entity|userId)`` for the daily-snapshot containers (and the
    ``timestamp`` equivalents for the log/ledger) -- so queries that lead with
    either field are served by a regular index. The default ``/*`` included path
    also range-indexes each field individually.
  * A full-text index on the business key field so the entity/userId text can be
    searched in addition to being filtered, plus a full-text index on ``/data``
    (the text payload) for every container that stores one -- i.e. all except
    the ``GaiaConnections`` ledger and the ``GaiaWebSearchHistory`` log -- so the
    content itself is keyword/full-text searchable.

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


# The five core containers from the diagram plus the derived DataLakeIndex.
# ``key_field`` is the business/partition
# key: the entity tables key on "entity".
# ``unique_field`` is the path made unique *within* a partition; it is "date"
# for the daily-snapshot containers and "timestamp" for the log/ledger
# (which need many rows per day, so they key on a full instant, not a day).
@dataclass(frozen=True)
class TableSpec:
    """Describes one Cosmos container to create."""

    name: str                  # container id, e.g. "GaiaKB"
    key_field: str             # business key / partition key field, e.g. "entity"
    unique_field: str = "date"  # per-partition unique path, e.g. "date" or "timestamp"
    has_data: bool = True       # whether the container stores a "/data" text payload
    # Optional vector overrides. When None, the container uses the shared
    # "/dataVector" path and the account-wide COSMOS_VECTOR_DIMS dimension. The
    # derived DataLakeIndex sets these to index its half-size "/indexVector".
    vector_path: str | None = None
    vector_dims: int | None = None


TABLES: list[TableSpec] = [
    TableSpec(name="GaiaKB", key_field="entity"),
    TableSpec(name="GaiaDataLake", key_field="entity"),
    TableSpec(name="GaiaDiary", key_field="entity"),
    # Append-only log of Gaia's web searches. It stores the query and results
    # rather than a "/data" text payload, so it has a vector index over its
    # embedded query text but no "/data" full-text index. Many searches happen
    # per day, so uniqueness is per timestamp.
    TableSpec(
        name="GaiaWebSearchHistory",
        key_field="entity",
        unique_field="timestamp",
        has_data=False,
    ),
    # Emotional-bank-account ledger: one record per (entity, timestamp). It
    # records signed deltas and notes, not a "/data" text payload, so it has no
    # full-text content to index.
    TableSpec(
        name="GaiaConnections",
        key_field="entity",
        unique_field="timestamp",
        has_data=False,
    ),
    # Derived semantic index over GaiaDataLake. Each entry stores only the
    # source id, partition, day, source container, and a *half-size* (768-d)
    # vector at "/indexVector" -- not "/dataVector" and no "/data" payload. The
    # Rust write path (write_data_controller) creates this on first write if it
    # is missing; provisioning it here keeps a fresh account self-consistent.
    TableSpec(
        name="DataLakeIndex",
        key_field="entity",
        unique_field="date",
        has_data=False,
        vector_path="/indexVector",
        vector_dims=768,
    ),
]

# Path of the vector field. Every container stores the embedding of its text
# content here; the container's policies declare and index this path.
VECTOR_PATH = "/dataVector"


# --- Policy builders ---------------------------------------------------------
# These are small pure functions so each policy is easy to read and test.


def build_vector_embedding_policy(dimensions: int, path: str = VECTOR_PATH) -> dict:
    """Vector embedding policy: declares ``path`` as the data's embedding.

    Cosine distance is the usual choice for text-embedding similarity.
    """
    return {
        "vectorEmbeddings": [
            {
                "path": path,
                "dataType": "float32",
                "distanceFunction": "cosine",
                "dimensions": dimensions,
            }
        ]
    }


def build_full_text_policy(key_field: str, include_data: bool = True) -> dict:
    """Full-text policy: enables text search over the entity/userId field.

    When the container stores a ``/data`` text payload (every container except
    the connections ledger), that field is included too so the actual content --
    not just the business key -- can be full-text searched.
    """
    full_text_paths = [{"path": f"/{key_field}", "language": "en-US"}]
    if include_data:
        full_text_paths.append({"path": "/data", "language": "en-US"})
    return {
        "defaultLanguage": "en-US",
        "fullTextPaths": full_text_paths,
    }


def build_indexing_policy(
    key_field: str, unique_field: str, include_data: bool = True,
    vector_path: str = VECTOR_PATH,
) -> dict:
    """Indexing policy combining the standard, vector, full-text, and composite indexes.

    * The vector path is excluded from normal indexing (vectors are indexed only
      by the dedicated DiskANN vector index).
    * ``vectorIndexes`` indexes ``/dataVector`` for similarity search over the
      record's text content; combined with the partition key it can be filtered
      by entity/userId. Every container gets one.
    * ``fullTextIndexes`` adds the extra text index on the entity/userId field,
      plus the ``/data`` payload when the container stores one, so keyword and
      full-text queries over the content are served by an index.
    * ``compositeIndexes`` pairs the business key with the uniqueness field
      (``date`` or ``timestamp``) in *both* orders so queries that lead with
      either the business key or the date/timestamp are served by a regular
      index. The default ``/*`` included path also range-indexes each field
      individually.
    """
    full_text_indexes = [{"path": f"/{key_field}"}]
    if include_data:
        full_text_indexes.append({"path": "/data"})
    return {
        "indexingMode": "consistent",
        "automatic": True,
        "includedPaths": [{"path": "/*"}],
        "excludedPaths": [
            # Vectors are served only by the DiskANN vector index below, so keep
            # the raw vector out of the normal range index.
            {"path": f"{vector_path}/*"},
            {"path": '/"_etag"/?'},
        ],
        "vectorIndexes": [
            {"path": vector_path, "type": "diskANN"},
        ],
        "fullTextIndexes": full_text_indexes,
        "compositeIndexes": [
            # Regular index on (entity|userId, date|timestamp): filter by the
            # business key, range/order by the uniqueness field.
            [
                {"path": f"/{key_field}", "order": "ascending"},
                {"path": f"/{unique_field}", "order": "ascending"},
            ],
            # The reverse order (date|timestamp, entity|userId): filter/range by
            # the uniqueness field first, then by the business key. Cosmos
            # composite indexes are order-sensitive, so both directions are
            # declared to serve queries that lead with either field.
            [
                {"path": f"/{unique_field}", "order": "ascending"},
                {"path": f"/{key_field}", "order": "ascending"},
            ],
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
    # Per-table vector overrides: the derived DataLakeIndex indexes a half-size
    # "/indexVector"; every other container uses the shared "/dataVector".
    vector_path = table.vector_path or VECTOR_PATH
    vector_dims = table.vector_dims or dimensions
    database.create_container_if_not_exists(
        id=table.name,
        # Partition key = business key field -> enables cheap filtering.
        partition_key=PartitionKey(path=f"/{table.key_field}"),
        indexing_policy=build_indexing_policy(
            table.key_field, table.unique_field, table.has_data, vector_path
        ),
        vector_embedding_policy=build_vector_embedding_policy(vector_dims, vector_path),
        full_text_policy=build_full_text_policy(table.key_field, table.has_data),
        unique_key_policy=build_unique_key_policy(table.unique_field),
        offer_throughput=throughput,
    )
    print(f"  done: '{table.name}'")


# --- Entry point -------------------------------------------------------------


def main() -> int:
    """Provision the database and all containers. Returns a process exit code."""
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
