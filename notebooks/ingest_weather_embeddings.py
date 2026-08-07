# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest Weather Documents -> Vector Embeddings (Lakebase)
# MAGIC
# MAGIC This notebook is the weather-domain counterpart to
# MAGIC `ingest_ticker_news_embeddings.py`. It:
# MAGIC 1. Reads rows from `weather_documents` (populated by `POST /weather/sync`
# MAGIC    in `weather_client.py`) that don't have embeddings yet.
# MAGIC 2. Chunks `narrative_text` with a sliding window (`CHUNK_SIZE=800`,
# MAGIC    `CHUNK_OVERLAP=100` chars) - most NWS alert/forecast text is short
# MAGIC    enough to fit in a single chunk, but combined alert
# MAGIC    description+instruction text can run long, so chunking still matters
# MAGIC    sometimes.
# MAGIC 3. Embeds each chunk with `sentence-transformers/all-MiniLM-L6-v2`
# MAGIC    (384-dim) - the SAME model as the news pipeline, so both tables stay
# MAGIC    queryable with the same distance operator (`vector_cosine_ops`).
# MAGIC 4. Upserts into `weather_embeddings` (id, document_id, chunk_index,
# MAGIC    chunk_text, embedding vector(384), model_name, created_at) via
# MAGIC    `psycopg2` + `execute_values`, casting straight to `::vector` -
# MAGIC    NO Spark, NO `spark.write.jdbc` anywhere in this notebook, since that
# MAGIC    path is unreliable against this Lakebase instance.
# MAGIC
# MAGIC Unlike the news notebook, this one uses `lakebase.py`'s `get_connection()`
# MAGIC directly (same secret scope/key it already resolves) instead of manually
# MAGIC re-parsing the connection URL.

# COMMAND ----------

# DBTITLE 1,Install required packages
# MAGIC %pip install --upgrade --force-reinstall 'databricks-sdk>=0.118.0' sentence-transformers psycopg sqlalchemy

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config
# MAGIC
# MAGIC Widgets let you override table names, chunk sizing, and the embedding
# MAGIC model without editing the notebook - useful when running this as a
# MAGIC scheduled Databricks Job.

# COMMAND ----------

dbutils.widgets.text("documents_table_name", "weather_documents", "Source table (weather documents)")
dbutils.widgets.text("embeddings_table_name", "weather_embeddings", "Destination table (chunk vectors)")
dbutils.widgets.text("embedding_model", "sentence-transformers/all-MiniLM-L6-v2", "Embedding model")
dbutils.widgets.text("chunk_size", "800", "narrative_text chunk size (chars)")
dbutils.widgets.text("chunk_overlap", "100", "narrative_text chunk overlap (chars)")
dbutils.widgets.text("batch_size", "32", "Embedding batch size")

DOCUMENTS_TABLE_NAME = dbutils.widgets.get("documents_table_name")
EMBEDDINGS_TABLE_NAME = dbutils.widgets.get("embeddings_table_name")
EMBEDDING_MODEL_NAME = dbutils.widgets.get("embedding_model")
CHUNK_SIZE = int(dbutils.widgets.get("chunk_size"))
CHUNK_OVERLAP = int(dbutils.widgets.get("chunk_overlap"))
BATCH_SIZE = int(dbutils.widgets.get("batch_size"))

# Different sentence-transformers models emit different vector sizes, and the
# pgvector column type (VECTOR(N)) must match exactly. Switch on the model
# name so swapping EMBEDDING_MODEL_NAME via the widget above fails loudly
# instead of silently writing mismatched vectors.
match EMBEDDING_MODEL_NAME:
    case "sentence-transformers/all-MiniLM-L6-v2":
        EMBEDDING_DIM = 384
    case "sentence-transformers/all-MiniLM-L12-v2":
        EMBEDDING_DIM = 384
    case "sentence-transformers/all-mpnet-base-v2":
        EMBEDDING_DIM = 768
    case "BAAI/bge-small-en-v1.5":
        EMBEDDING_DIM = 384
    case "BAAI/bge-base-en-v1.5":
        EMBEDDING_DIM = 768
    case _:
        raise ValueError(
            f"Unknown embedding model {EMBEDDING_MODEL_NAME!r} - add its output "
            "dimension to the match/case block above before running this notebook. "
            "If you change models, `weather_embeddings.embedding` must also be "
            "recreated as vector(<new dim>) - see the DDL note in this notebook's header."
        )

print(f"Using model {EMBEDDING_MODEL_NAME!r} -> {EMBEDDING_DIM}-dim vectors")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Database Setup Instructions
# MAGIC
# MAGIC Before running this notebook, make sure both tables already exist in
# MAGIC Lakebase:
# MAGIC 1. `weather_documents` (created by `weather_client.py`'s `ensure_table()`
# MAGIC    or run by hand).
# MAGIC 2. `weather_embeddings` - requires the `pgvector` extension and an HNSW
# MAGIC    index. See this notebook's accompanying SQL:
# MAGIC
# MAGIC ```sql
# MAGIC CREATE EXTENSION IF NOT EXISTS vector;
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS weather_embeddings (
# MAGIC     id              TEXT PRIMARY KEY,  -- document_id || '_' || chunk_index
# MAGIC     document_id     TEXT NOT NULL REFERENCES weather_documents(id) ON DELETE CASCADE,
# MAGIC     chunk_index     INTEGER NOT NULL,
# MAGIC     chunk_text      TEXT NOT NULL,
# MAGIC     embedding       vector(384) NOT NULL,
# MAGIC     model_name      TEXT NOT NULL,
# MAGIC     created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
# MAGIC );
# MAGIC
# MAGIC CREATE INDEX IF NOT EXISTS idx_weather_embeddings_hnsw
# MAGIC     ON weather_embeddings USING hnsw (embedding vector_cosine_ops);
# MAGIC
# MAGIC CREATE INDEX IF NOT EXISTS idx_weather_embeddings_document_id
# MAGIC     ON weather_embeddings (document_id);
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Connect to Lakebase
# MAGIC
# MAGIC Reuses `lakebase.py`'s `get_connection()` context manager directly -
# MAGIC same secret scope/key resolution it already does, no re-parsing of the
# MAGIC connection URL needed here.

# COMMAND ----------

# DBTITLE 1,Import lakebase connection helper
import base64
from urllib.parse import urlparse

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()


def get_lakebase_url() -> str:
    secret = w.secrets.get_secret(scope="database", key="lakebase-weather-url")
    # Secret is double base64 encoded, decode twice
    first_decode = base64.b64decode(secret.value).decode("utf-8")
    return base64.b64decode(first_decode).decode("utf-8")


lakebase_url = get_lakebase_url()
parsed = urlparse(lakebase_url)

# Extract connection details directly from the secret URL
db_host = parsed.hostname
db_port = parsed.port or 5432
db_name = parsed.path.lstrip('/')
db_user = parsed.username
db_password = parsed.password

print(f"Connection details:")
print(f"  Host: {db_host}:{db_port}")
print(f"  Database: {db_name}")
print(f"  User: {db_user}")
print(f"  Using raw credentials from secret (no OAuth)")

# COMMAND ----------

# DBTITLE 1,Fetch Weather Data from NWS API
# MAGIC %md
# MAGIC ## Fetch Weather Data from NWS API
# MAGIC
# MAGIC Before embedding, we need weather documents to embed. This section fetches
# MAGIC active alerts and forecasts from api.weather.gov for configured locations
# MAGIC and inserts them into `weather_documents`.
# MAGIC
# MAGIC The NWS API provides:
# MAGIC - **Active alerts** (watches, warnings, advisories) via `/alerts/active`
# MAGIC - **Gridpoint forecasts** (period-by-period forecast text) via `/gridpoints/{gridId}/{x},{y}/forecast`
# MAGIC
# MAGIC Each location is a "lat,lon" coordinate pair (e.g. "41.8781,-87.6298").
# MAGIC The Census geocoder for "City, ST" requires full street addresses, so
# MAGIC lat/lon is more reliable. Defaults: Chicago, Austin, Seattle.

# COMMAND ----------

# DBTITLE 1,Configure weather sync locations
dbutils.widgets.text(
    "weather_locations",
    "41.8781,-87.6298; 30.2672,-97.7431; 47.6062,-122.3321",
    "Weather locations (lat,lon pairs; semicolon-separated)"
)
dbutils.widgets.text("weather_limit", "50", "Max alerts/forecast periods per location")

WEATHER_LOCATIONS_RAW = dbutils.widgets.get("weather_locations")
WEATHER_LIMIT = int(dbutils.widgets.get("weather_limit"))

# Parse semicolon-separated locations
WEATHER_LOCATIONS = [loc.strip() for loc in WEATHER_LOCATIONS_RAW.split(";") if loc.strip()]

print(f"Will sync {len(WEATHER_LOCATIONS)} locations: {WEATHER_LOCATIONS}")

# COMMAND ----------

# DBTITLE 1,Import weather sync utilities
import hashlib
import os
from datetime import datetime, timezone
from typing import Any

import requests

# NWS API config
BASE_URL = os.environ.get("NWS_API_BASE_URL", "https://api.weather.gov")
USER_AGENT = os.environ.get("NWS_USER_AGENT", "weather-intelligence-notebook (genie-databricks)")
CENSUS_GEOCODER_URL = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
DEFAULT_TIMEOUT = 30


class WeatherClient:
    """Thin wrapper around the NWS API."""
    def __init__(self, base_url: str = BASE_URL, timeout: int = DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "application/geo+json",
        })

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        resp = self._session.get(f"{self.base_url}{path}", params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def resolve_point(self, lat: float, lon: float) -> dict:
        data = self.get(f"/points/{lat},{lon}")
        return data.get("properties", {})

    def get_active_alerts(self, lat: float, lon: float) -> list[dict]:
        data = self.get("/alerts/active", params={"point": f"{lat},{lon}"})
        return data.get("features", [])

    def get_forecast(self, grid_id: str, grid_x: int, grid_y: int) -> dict:
        data = self.get(f"/gridpoints/{grid_id}/{grid_x},{grid_y}/forecast")
        return data.get("properties", {})


def geocode_location(location: str) -> tuple[float, float]:
    """Resolve location string to (lat, lon). Accepts 'lat,lon' or 'City, ST'."""
    # Check if it's already lat,lon
    parts = [p.strip() for p in location.split(",")]
    if len(parts) == 2:
        try:
            lat, lon = float(parts[0]), float(parts[1])
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
        except ValueError:
            pass

    # Geocode via Census Bureau
    resp = requests.get(
        CENSUS_GEOCODER_URL,
        params={"address": location, "benchmark": "Public_AR_Current", "format": "json"},
        timeout=DEFAULT_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    matches = data.get("result", {}).get("addressMatches", [])
    if not matches:
        raise ValueError(f"Could not geocode location: {location!r}")
    coords = matches[0]["coordinates"]
    return float(coords["y"]), float(coords["x"])


def normalize_alert(feature: dict, location: str) -> dict:
    """Normalize an NWS alert feature to a document record."""
    props = feature.get("properties", {})
    return {
        "id": feature.get("id") or props.get("id"),
        "location": location,
        "source_type": "alert",
        "headline": props.get("event"),
        "narrative_text": props.get("description") or props.get("instruction"),
        "issued_at": props.get("sent"),
        "effective_at": props.get("effective"),
        "payload": feature,
        "synced_at": datetime.now(timezone.utc).isoformat(),
    }


def normalize_forecast_period(period: dict, location: str, updated_at: str | None) -> dict:
    """Normalize a forecast period to a document record."""
    dedup_source = f"{location}|forecast|{period.get('number')}|{period.get('startTime')}"
    doc_id = "forecast-" + hashlib.sha256(dedup_source.encode("utf-8")).hexdigest()[:24]
    return {
        "id": doc_id,
        "location": location,
        "source_type": "forecast",
        "headline": period.get("name"),
        "narrative_text": period.get("detailedForecast"),
        "issued_at": updated_at,
        "effective_at": period.get("startTime"),
        "payload": period,
        "synced_at": datetime.now(timezone.utc).isoformat(),
    }


print("✓ Weather sync utilities loaded")

# COMMAND ----------

# DBTITLE 1,Fetch weather data from NWS API
from datetime import datetime, timezone

client = WeatherClient()
all_weather_docs = []

# Use hardcoded coordinates (Chicago, Austin, Seattle) for now
# Update the weather_locations widget in the UI to change these
test_locations = [
    ("41.8781,-87.6298", "Chicago, IL"),
    ("30.2672,-97.7431", "Austin, TX"),
    ("47.6062,-122.3321", "Seattle, WA"),
]

for coords, location_name in test_locations:
    location = coords
    try:
        print(f"\nSyncing {location}...")
        
        # Geocode location to lat/lon
        lat, lon = geocode_location(location)
        print(f"  Resolved to ({lat:.4f}, {lon:.4f})")
        
        # Fetch active alerts
        alerts = client.get_active_alerts(lat, lon)
        for feature in alerts[:WEATHER_LIMIT]:
            all_weather_docs.append(normalize_alert(feature, location))
        print(f"  Found {len(alerts[:WEATHER_LIMIT])} alerts")
        
        # Fetch gridpoint forecast
        point = client.resolve_point(lat, lon)
        grid_id = point.get("gridId")
        grid_x = point.get("gridX")
        grid_y = point.get("gridY")
        
        if grid_id and grid_x is not None and grid_y is not None:
            forecast = client.get_forecast(grid_id, grid_x, grid_y)
            updated_at = forecast.get("updated")
            periods = forecast.get("periods", [])
            
            for period in periods[:WEATHER_LIMIT]:
                all_weather_docs.append(
                    normalize_forecast_period(period, location, updated_at)
                )
            print(f"  Found {len(periods[:WEATHER_LIMIT])} forecast periods")
        else:
            print(f"  No forecast grid available for this location")
            
    except Exception as e:
        print(f"  ⚠️  Error syncing {location}: {e}")
        continue

print(f"\n" + "="*60)
print(f"Fetched {len(all_weather_docs)} total weather documents")
print("="*60)

# COMMAND ----------

# DBTITLE 1,Insert weather documents into Lakebase
import json
from sqlalchemy import create_engine, text

if all_weather_docs:
    # Build connection URL
    connection_url = get_lakebase_url()
    engine = create_engine(connection_url.replace("postgresql://", "postgresql+psycopg://"))
    
    # Ensure weather_documents table exists
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS weather_documents (
        id              TEXT PRIMARY KEY,
        location        TEXT NOT NULL,
        source_type     TEXT NOT NULL,
        headline        TEXT,
        narrative_text  TEXT,
        issued_at       TIMESTAMPTZ,
        effective_at    TIMESTAMPTZ,
        payload         JSONB,
        synced_at       TIMESTAMPTZ NOT NULL
    )
    """
    
    upsert_sql = """
    INSERT INTO weather_documents
        (id, location, source_type, headline, narrative_text, issued_at, effective_at, payload, synced_at)
    VALUES (:id, :location, :source_type, :headline, :narrative_text, :issued_at, :effective_at, :payload, :synced_at)
    ON CONFLICT (id) DO UPDATE SET
        location       = EXCLUDED.location,
        source_type    = EXCLUDED.source_type,
        headline       = EXCLUDED.headline,
        narrative_text = EXCLUDED.narrative_text,
        issued_at      = EXCLUDED.issued_at,
        effective_at   = EXCLUDED.effective_at,
        payload        = EXCLUDED.payload,
        synced_at      = EXCLUDED.synced_at
    """
    
    with engine.begin() as conn:
        # Create table if needed
        conn.execute(text(create_table_sql))
        
        # Upsert all documents
        for doc in all_weather_docs:
            conn.execute(
                text(upsert_sql),
                {
                    "id": doc["id"],
                    "location": doc["location"],
                    "source_type": doc["source_type"],
                    "headline": doc.get("headline"),
                    "narrative_text": doc.get("narrative_text"),
                    "issued_at": doc.get("issued_at"),
                    "effective_at": doc.get("effective_at"),
                    "payload": json.dumps(doc.get("payload")),
                    "synced_at": doc["synced_at"],
                }
            )
    
    print(f"✅ Upserted {len(all_weather_docs)} weather documents into {DOCUMENTS_TABLE_NAME}")
else:
    print("No weather documents to insert.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load unembedded weather documents
# MAGIC
# MAGIC A document is "unembedded" if `weather_embeddings` has no rows with a
# MAGIC matching `document_id` yet. Re-running this notebook is therefore safe -
# MAGIC already-embedded documents are skipped, not re-processed.

# COMMAND ----------

# DBTITLE 1,Query unembedded documents via psycopg2
import base64
from databricks.sdk import WorkspaceClient
from sqlalchemy import create_engine, text

DOCUMENTS_TABLE_NAME = dbutils.widgets.get("documents_table_name")
EMBEDDINGS_TABLE_NAME = dbutils.widgets.get("embeddings_table_name")

UNEMBEDDED_QUERY = f"""
    SELECT id, location, source_type, headline, narrative_text
    FROM {DOCUMENTS_TABLE_NAME} wd
    WHERE narrative_text IS NOT NULL
      AND TRIM(narrative_text) != ''\n      AND NOT EXISTS (
          SELECT 1 FROM {EMBEDDINGS_TABLE_NAME} we WHERE we.document_id = wd.id
      )
"""

# Get connection URL using double-decode (secret is double base64 encoded)
connection_url = get_lakebase_url()

# Use psycopg (v3) driver to avoid psycopg2 C extension crashes
engine = create_engine(connection_url.replace("postgresql://", "postgresql+psycopg://"))

with engine.connect() as conn:
    result = conn.execute(text(UNEMBEDDED_QUERY))
    unembedded_docs = [dict(row._mapping) for row in result]

print(f"Found {len(unembedded_docs)} unembedded documents in {DOCUMENTS_TABLE_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunk narrative_text
# MAGIC
# MAGIC Same sliding-window pattern as the news pipeline's article-body
# MAGIC chunking: `CHUNK_SIZE=800` / `CHUNK_OVERLAP=100` characters. If
# MAGIC `narrative_text` is shorter than `CHUNK_SIZE` (true for most NWS
# MAGIC forecast periods and many short alerts), this naturally produces exactly
# MAGIC one chunk containing the full text - no separate "short text" branch
# MAGIC needed. Longer combined alert `description` + `instruction` text is
# MAGIC split into overlapping windows so no passage is lost to a single
# MAGIC truncated chunk.

# COMMAND ----------

# DBTITLE 1,Chunk narrative_text per document


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Sliding-window chunker. Always returns at least one chunk for non-empty text."""
    text = text.strip()
    if not text:
        return []

    step = max(chunk_size - chunk_overlap, 1)
    chunks = []
    for start in range(0, len(text), step):
        piece = text[start : start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(text):
            break
    return chunks


chunk_rows = []  # list of dicts: id, document_id, chunk_index, chunk_text
for doc in unembedded_docs:
    pieces = chunk_text(doc["narrative_text"], CHUNK_SIZE, CHUNK_OVERLAP)
    for chunk_index, piece in enumerate(pieces):
        chunk_rows.append(
            {
                "id": f"{doc['id']}_{chunk_index}",
                "document_id": doc["id"],
                "chunk_index": chunk_index,
                "chunk_text": piece,
            }
        )

print(
    f"Chunked {len(unembedded_docs)} documents into {len(chunk_rows)} chunks "
    f"({sum(1 for r in chunk_rows if r['chunk_index'] > 0)} came from multi-chunk documents)"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute embeddings
# MAGIC
# MAGIC Loads the sentence-transformers model once and applies it in batches.
# MAGIC No Spark / pandas UDF distribution here - the news notebook uses that
# MAGIC for very large ticker-news volumes, but weather documents are a small
# MAGIC enough batch per sync that plain in-process batching is simpler and
# MAGIC plenty fast. (If your volume grows enough to need it, wrap the batch
# MAGIC loop in a `concurrent.futures.ThreadPoolExecutor` - the model's `encode()`
# MAGIC call itself already releases the GIL during the heavy tensor ops, so
# MAGIC threading multiple batches concurrently is safe.)

# COMMAND ----------

# DBTITLE 1,Compute chunk embeddings
import os

from sentence_transformers import SentenceTransformer

os.environ["HF_HOME"] = "/tmp/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/tmp/.cache/huggingface"

if chunk_rows:
    print(f"Loading embedding model {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder="/tmp/.cache/huggingface")

    print(f"Computing {len(chunk_rows)} chunk embeddings...")
    texts = [r["chunk_text"] for r in chunk_rows]
    all_vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        vectors = model.encode(batch, show_progress_bar=False)
        all_vectors.extend(vectors.tolist())
        if (i + BATCH_SIZE) % (BATCH_SIZE * 4) == 0:
            print(f"  Processed {min(i + BATCH_SIZE, len(texts))}/{len(texts)} chunks")

    for row, vector in zip(chunk_rows, all_vectors):
        row["embedding"] = vector

    print(f"Computed {len(chunk_rows)} embeddings using {EMBEDDING_MODEL_NAME}")
else:
    print("No unembedded chunks - skipping model load.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upsert embeddings into Lakebase
# MAGIC
# MAGIC Written via `psycopg2.extras.execute_values` for batched throughput.
# MAGIC Each embedding is passed as a pgvector text literal (`'[0.1,0.2,...]'`)
# MAGIC and cast directly with `%s::vector` in the same statement - no
# MAGIC intermediate `double precision[]` column or follow-up `UPDATE ... ::vector`
# MAGIC pass needed, since we control the DDL here and can commit to `vector(384)`
# MAGIC from the start.

# COMMAND ----------

# DBTITLE 1,Insert chunk embeddings using psycopg2 + execute_values
from datetime import datetime, timezone
from sqlalchemy import create_engine, text

if chunk_rows:
    now = datetime.now(timezone.utc)
    
    # Use get_lakebase_url() to get connection URL
    connection_url = get_lakebase_url()
    engine = create_engine(connection_url.replace("postgresql://", "postgresql+psycopg://"))
    
    insert_sql = f"""
        INSERT INTO {EMBEDDINGS_TABLE_NAME} (
            id, document_id, chunk_index, chunk_text, embedding, model_name, created_at
        ) VALUES (:id, :document_id, :chunk_index, :chunk_text, CAST(:embedding AS vector), :model_name, :created_at)
        ON CONFLICT (id) DO UPDATE SET
            chunk_text  = EXCLUDED.chunk_text,
            embedding   = EXCLUDED.embedding,
            model_name  = EXCLUDED.model_name,
            created_at  = EXCLUDED.created_at
    """
    
    with engine.begin() as conn:
        for row in chunk_rows:
            embedding_str = "[" + ",".join(str(float(x)) for x in row["embedding"]) + "]"
            conn.execute(
                text(insert_sql),
                {
                    "id": row["id"],
                    "document_id": row["document_id"],
                    "chunk_index": row["chunk_index"],
                    "chunk_text": row["chunk_text"],
                    "embedding": embedding_str,
                    "model_name": EMBEDDING_MODEL_NAME,
                    "created_at": now,
                }
            )

    print(f"✅ Upserted {len(chunk_rows)} chunk embeddings into {EMBEDDINGS_TABLE_NAME}")
else:
    print("No chunk embeddings to write.")