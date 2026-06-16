"""Async client for the Gaia Foundry text-embedding model.

This wraps the high-performance embedding deployment provisioned by
``infra/azuredeploy.json`` (``deployEmbeddingModel=true``) and turns record text
into the vector stored at Cosmos path ``/dataVector``.

The class is deliberately small and reusable:

  * :meth:`EmbeddingClient.embed` embeds a *batch* of strings in one API call.
  * :meth:`EmbeddingClient.embed_if_missing` populates a single Cosmos document
    *lazily* -- it only calls the model when the document has no usable vector
    yet, so it is safe to call on every record and cheap on already-embedded
    ones. This is what the backfill worker (``infra/embed_worker.py``) uses.

Authentication mirrors the rest of ``infra`` (checked in this order):
  * ``FOUNDRY_KEY`` / ``AZURE_OPENAI_API_KEY`` env var -> API-key auth.
  * otherwise -> Azure AD via ``DefaultAzureCredential`` (e.g. ``az login`` or a
    managed identity) against the Cognitive Services scope.

Required environment variables (see ``.env.sample``):
  FOUNDRY_ENDPOINT       e.g. https://<account>.cognitiveservices.azure.com/
  EMBEDDING_DEPLOYMENT   the deployment name (template default: "text-embedding")
Optional environment variables:
  EMBEDDING_DIMENSIONS   vector length; defaults to COSMOS_VECTOR_DIMS or 1536.
                         Must match the Cosmos /dataVector policy. Set to 0 for
                         models that do not support dimension reduction (e.g.
                         text-embedding-ada-002).
  FOUNDRY_KEY            API key (if omitted, Azure AD is used).
  AZURE_OPENAI_API_VERSION  REST API version (default below).
"""

from __future__ import annotations

import os
from typing import Optional

# Cosmos document fields this client reads/writes.
VECTOR_FIELD = "dataVector"
TEXT_FIELD = "data"

# Default Azure OpenAI REST API version. >= 2024-02-01 is required for the
# ``dimensions`` parameter used by the text-embedding-3 models.
DEFAULT_API_VERSION = "2024-10-21"

# Azure AD scope for data-plane calls to a Cognitive Services / Foundry account.
COGNITIVE_SCOPE = "https://cognitiveservices.azure.com/.default"

# Conservative per-input character cap. text-embedding-3 accepts ~8191 tokens
# per input; the migrated day-aggregate documents can be far larger, so we
# truncate to stay well under the limit (roughly 4 chars/token).
DEFAULT_MAX_CHARS = 24_000


def _friendly_import(module: str):
    """Import an optional dependency, with an actionable error if it is absent."""
    try:
        return __import__(module)
    except ImportError as error:  # pragma: no cover - exercised only without deps
        raise ImportError(
            f"the '{module}' package is required for embeddings. Install it with:\n"
            "  pip install -r infra/requirements.txt"
        ) from error


class EmbeddingClient:
    """Async wrapper over the Foundry text-embedding deployment.

    Use it as an async context manager so the underlying HTTP client is closed::

        async with EmbeddingClient.from_env() as client:
            vectors = await client.embed(["hello", "world"])
    """

    def __init__(
        self,
        endpoint: str,
        deployment: str,
        dimensions: Optional[int] = 1536,
        *,
        api_key: Optional[str] = None,
        api_version: str = DEFAULT_API_VERSION,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> None:
        """Create a client bound to one embedding deployment.

        ``dimensions`` is sent to the model so the returned vector length matches
        the Cosmos ``/dataVector`` policy. Pass ``None`` (or 0) to omit it for
        models that do not support dimension reduction.
        """
        if not endpoint:
            raise ValueError("endpoint is required (set FOUNDRY_ENDPOINT)")
        if not deployment:
            raise ValueError("deployment is required (set EMBEDDING_DEPLOYMENT)")

        self.deployment = deployment
        self.dimensions = dimensions if dimensions else None
        self.max_chars = max_chars

        # Build the async Azure OpenAI client lazily so importing this module
        # never requires the optional 'openai' / 'azure-identity' packages.
        openai = _friendly_import("openai")

        if api_key:
            # Simplest auth path: a deployment / account key.
            self._client = openai.AsyncAzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
            )
        else:
            # Azure AD (recommended). Imported lazily so azure-identity is only
            # needed when actually using AAD.
            identity = _friendly_import("azure.identity")
            token_provider = identity.get_bearer_token_provider(
                identity.DefaultAzureCredential(), COGNITIVE_SCOPE
            )
            self._client = openai.AsyncAzureOpenAI(
                azure_endpoint=endpoint,
                azure_ad_token_provider=token_provider,
                api_version=api_version,
            )

    @classmethod
    def from_env(cls) -> "EmbeddingClient":
        """Build a client from environment variables (see module docstring)."""
        endpoint = os.environ.get("FOUNDRY_ENDPOINT", "").strip()
        deployment = os.environ.get("EMBEDDING_DEPLOYMENT", "").strip()

        # Default the vector length to the Cosmos policy so the two always agree.
        raw_dims = os.environ.get("EMBEDDING_DIMENSIONS") or os.environ.get(
            "COSMOS_VECTOR_DIMS"
        )
        dimensions = int(raw_dims) if raw_dims else 1536

        api_key = (
            os.environ.get("FOUNDRY_KEY") or os.environ.get("AZURE_OPENAI_API_KEY") or ""
        ).strip() or None
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION)

        return cls(
            endpoint=endpoint,
            deployment=deployment,
            dimensions=dimensions,
            api_key=api_key,
            api_version=api_version,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings, returning one vector per input (in order).

        Each input is truncated to ``max_chars`` first. Inputs must be non-empty;
        callers that may have blanks should pre-filter (the worker does).
        """
        if not texts:
            return []

        prepared = [self._prepare(text) for text in texts]

        # text-embedding-3 models accept the optional ``dimensions`` parameter;
        # only send it when configured so older models still work.
        kwargs = {"model": self.deployment, "input": prepared}
        if self.dimensions:
            kwargs["dimensions"] = self.dimensions

        response = await self._client.embeddings.create(**kwargs)

        # The API returns items with an ``index``; sort to be order-safe.
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]

    async def embed_one(self, text: str) -> list[float]:
        """Embed a single string and return its vector."""
        vectors = await self.embed([text])
        return vectors[0]

    async def embed_if_missing(self, doc: dict) -> bool:
        """Lazily populate ``doc['dataVector']`` if it is missing and has text.

        Returns ``True`` when a new embedding was computed and assigned, ``False``
        when the document already had a vector or had no text to embed. Mutates
        ``doc`` in place so the caller can persist it.
        """
        if not needs_embedding(doc):
            return False

        text = str(doc.get(TEXT_FIELD, "")).strip()
        if not text:
            # Nothing to embed; leave the document untouched.
            return False

        doc[VECTOR_FIELD] = await self.embed_one(text)
        return True

    def _prepare(self, text: str) -> str:
        """Normalise and length-cap one input before sending it to the model."""
        # Embedding models reject empty inputs; collapse to a single space so a
        # caller that did not pre-filter still gets a (meaningless) vector rather
        # than an API error. The worker pre-filters, so this is just a guard.
        cleaned = str(text).strip() or " "
        if len(cleaned) > self.max_chars:
            return cleaned[: self.max_chars]
        return cleaned

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()

    async def __aenter__(self) -> "EmbeddingClient":
        return self

    async def __aexit__(self, *_exc_info) -> None:
        await self.aclose()


def needs_embedding(doc: dict) -> bool:
    """True when a Cosmos document has no usable ``/dataVector`` yet.

    Treats a missing field, ``None``, or an empty array as "needs embedding".
    """
    vector = doc.get(VECTOR_FIELD)
    if vector is None:
        return True
    # An empty list means the field exists but was never populated.
    return len(vector) == 0
