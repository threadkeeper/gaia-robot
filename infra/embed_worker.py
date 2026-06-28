"""Backfill empty ``/dataVector`` embeddings across every Gaia Cosmos container.

This is an infrastructure *worker* (not part of the Rust program). It scans all
of the Cosmos containers for records whose ``data`` has never been embedded
(``/dataVector`` missing or empty), sends their text to the Foundry embedding
model in batches, and writes the resulting vectors back -- concurrently and
idempotently. It is safe to re-run: already-embedded records are skipped by the
query itself, so a second run only picks up whatever is still missing.

Pipeline per container:
  1. Query for documents lacking a usable vector (cheap; uses IS_DEFINED /
     ARRAY_LENGTH so already-embedded rows never leave Cosmos).
  2. Group the rows into batches and embed each batch in one model call.
  3. Patch each document's ``/dataVector`` (partial update -- no full rewrite).

Concurrency is bounded by a semaphore so we never have more than ``--concurrency``
embedding calls in flight at once.

Authentication (each service picks one, checked in this order):
  * Cosmos:    COSMOS_KEY env var -> account key, else Azure AD.
  * Embedding: FOUNDRY_KEY / AZURE_OPENAI_API_KEY -> API key, else Azure AD.

Required environment variables (see ``.env.sample``):
  COSMOS_ENDPOINT        e.g. https://my-account.documents.azure.com:443/
  FOUNDRY_ENDPOINT       e.g. https://<account>.cognitiveservices.azure.com/
  EMBEDDING_DEPLOYMENT   embedding deployment name (template default: "text-embedding")
Optional environment variables:
  COSMOS_KEY, COSMOS_DATABASE (default "gaia"), COSMOS_VECTOR_DIMS,
  EMBEDDING_DIMENSIONS, FOUNDRY_KEY, AZURE_OPENAI_API_VERSION.

Usage:
  python infra/embed_worker.py                 # backfill every container
  python infra/embed_worker.py --container GaiaKB
  python infra/embed_worker.py --dry-run       # count only, no model calls
  python infra/embed_worker.py --batch-size 32 --concurrency 8 --max 500
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from embeddings import EmbeddingClient, TEXT_FIELD, VECTOR_FIELD

# --- Configuration -----------------------------------------------------------

# Container name -> partition/business key field. Mirrors infra/cosmos_create.py
# TABLES: the entity tables key on "entity", the user tables on "userId". We need
# the key field both to read each row's partition value and to target the patch.
#
# Only the containers that actually store a "/data" text payload appear here,
# because the backfill embeds "/data" into "/dataVector". The "/data"-less
# containers from cosmos_create.py (GaiaWebSearchHistory and GaiaConnections,
# both has_data=False) are deliberately omitted: they have nothing to embed.
CONTAINERS: dict[str, str] = {
    "GaiaKB": "entity",
    "GaiaDataLake": "entity",
    "GaiaDiary": "entity",
}

# Select only the fields we need: id, the partition value (aliased "pk"), and the
# text to embed. The WHERE clause leaves already-embedded rows in Cosmos.
QUERY_TEMPLATE = (
    "SELECT c.id, c.{key_field} AS pk, c.{text_field} AS data FROM c "
    "WHERE (NOT IS_DEFINED(c.{vector_field})) OR ARRAY_LENGTH(c.{vector_field}) = 0"
)


# --- .env loading ------------------------------------------------------------


def load_env_file(path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from a .env file into os.environ.

    Kept dependency-free (no python-dotenv) on purpose, matching the other infra
    scripts. Existing environment variables always win, blanks and '#' comments
    are skipped, and surrounding quotes are stripped.
    """
    if not os.path.exists(path):
        return

    with open(path, encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


# --- Cosmos helpers ----------------------------------------------------------


def get_cosmos_client(endpoint: str):
    """Create an async CosmosClient using an account key if present, else AAD.

    Returned object is an async context manager; the caller closes it.
    """
    from azure.cosmos.aio import CosmosClient

    key = os.environ.get("COSMOS_KEY")
    if key:
        return CosmosClient(url=endpoint, credential=key)

    # Azure AD auth (recommended). The async Cosmos client needs an *async*
    # credential, imported lazily so azure-identity is only required for AAD.
    from azure.identity.aio import DefaultAzureCredential

    return CosmosClient(url=endpoint, credential=DefaultAzureCredential())


def chunked(items: list, size: int) -> list[list]:
    """Split a list into consecutive chunks of at most ``size`` elements."""
    return [items[i : i + size] for i in range(0, len(items), size)]


async def fetch_missing(container, key_field: str, limit: int) -> list[dict]:
    """Return rows lacking a vector: dicts with ``id``, ``pk`` and ``data``.

    Rows whose text is blank are dropped here -- there is nothing to embed -- and
    at most ``limit`` rows are returned (0 means no cap).
    """
    query = QUERY_TEMPLATE.format(
        key_field=key_field, text_field=TEXT_FIELD, vector_field=VECTOR_FIELD
    )

    rows: list[dict] = []
    async for item in container.query_items(query=query):
        text = str(item.get("data", "")).strip()
        pk = item.get("pk")
        # Skip rows we cannot act on: no text to embed, or no partition value.
        if not text or pk is None:
            continue
        rows.append({"id": item["id"], "pk": pk, "data": text})
        if limit and len(rows) >= limit:
            break
    return rows


async def patch_vector(container, doc: dict, vector: list[float]) -> None:
    """Write one document's ``/dataVector`` via a partial (patch) update.

    Uses the "add" op so it both creates the field and overwrites any empty
    placeholder, keeping the operation idempotent.
    """
    await container.patch_item(
        item=doc["id"],
        partition_key=doc["pk"],
        patch_operations=[{"op": "add", "path": f"/{VECTOR_FIELD}", "value": vector}],
    )


async def process_container(
    database,
    name: str,
    key_field: str,
    embedder: EmbeddingClient,
    *,
    batch_size: int,
    concurrency: int,
    limit: int,
    dry_run: bool,
) -> int:
    """Backfill one container; returns the number of records embedded.

    Embedding calls are bounded by ``concurrency`` via a semaphore so a large
    backlog cannot open an unbounded number of simultaneous model requests.
    """
    container = database.get_container_client(name)
    rows = await fetch_missing(container, key_field, limit)

    if not rows:
        print(f"  {name}: nothing to backfill")
        return 0

    if dry_run:
        print(f"  {name}: {len(rows)} record(s) would be embedded (dry-run)")
        return 0

    print(f"  {name}: embedding {len(rows)} record(s) ...")
    semaphore = asyncio.Semaphore(concurrency)

    async def handle_batch(batch: list[dict]) -> int:
        # Limit how many batches embed at once.
        async with semaphore:
            vectors = await embedder.embed([row["data"] for row in batch])
            # Write the whole batch back concurrently.
            await asyncio.gather(
                *(patch_vector(container, row, vec) for row, vec in zip(batch, vectors))
            )
        return len(batch)

    counts = await asyncio.gather(
        *(handle_batch(batch) for batch in chunked(rows, batch_size))
    )
    embedded = sum(counts)
    print(f"  {name}: embedded {embedded} record(s)")
    return embedded


# --- Orchestration -----------------------------------------------------------


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line options for the backfill worker."""
    parser = argparse.ArgumentParser(description="Backfill empty Cosmos /dataVector embeddings.")
    parser.add_argument(
        "--container",
        choices=sorted(CONTAINERS),
        help="Only process this container (default: all).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Records per embedding API call (default: 16).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Maximum concurrent embedding calls (default: 4).",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=0,
        dest="limit",
        help="Cap records processed per container (0 = no cap).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count missing embeddings; do not call the model or write back.",
    )
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> int:
    """Run the backfill across the selected container(s); returns total embedded."""
    endpoint = os.environ.get("COSMOS_ENDPOINT", "").strip()
    if not endpoint:
        print("ERROR: COSMOS_ENDPOINT is not set (see infra/.env.sample).", file=sys.stderr)
        raise SystemExit(2)

    database_name = os.environ.get("COSMOS_DATABASE", "gaia")
    targets = {args.container: CONTAINERS[args.container]} if args.container else CONTAINERS

    print(f"Backfilling embeddings in database '{database_name}' ...")

    cosmos = get_cosmos_client(endpoint)
    # Build the embedding client unless this is a dry run (which makes no calls).
    embedder = None if args.dry_run else EmbeddingClient.from_env()

    total = 0
    try:
        async with cosmos:
            database = cosmos.get_database_client(database_name)
            for name, key_field in targets.items():
                total += await process_container(
                    database,
                    name,
                    key_field,
                    embedder,
                    batch_size=args.batch_size,
                    concurrency=args.concurrency,
                    limit=args.limit,
                    dry_run=args.dry_run,
                )
    finally:
        if embedder is not None:
            await embedder.aclose()

    print(f"Done. Embedded {total} record(s) total.")
    return total


def main(argv: list[str] | None = None) -> None:
    """Entry point: load .env, parse args, and run the async backfill."""
    load_env_file()
    args = parse_args(sys.argv[1:] if argv is None else argv)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
