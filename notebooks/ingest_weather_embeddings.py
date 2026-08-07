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
# MAGIC %pip uninstall -y psycopg2 psycopg2-binary
# MAGIC %pip install -q 'databricks-sdk>=0.118.0' sentence-transformers psycopg2-binary

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
from lakebase import get_connection

# Sanity-check the connection and confirm the source table is reachable.
with get_connection() as _conn:
    with _conn.cursor() as _cur:
        _cur.execute(f"SELECT COUNT(*) AS n FROM {DOCUMENTS_TABLE_NAME}")
        _count = _cur.fetchone()["n"]
        print(f"✅ Connected. {DOCUMENTS_TABLE_NAME} has {_count} rows.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load unembedded weather documents
# MAGIC
# MAGIC A document is "unembedded" if `weather_embeddings` has no rows with a
# MAGIC matching `document_id` yet. Re-running this notebook is therefore safe -
# MAGIC already-embedded documents are skipped, not re-processed.

# COMMAND ----------

# DBTITLE 1,Query unembedded documents via psycopg2
UNEMBEDDED_QUERY = f"""
    SELECT id, location, source_type, headline, narrative_text
    FROM {DOCUMENTS_TABLE_NAME} wd
    WHERE narrative_text IS NOT NULL
      AND TRIM(narrative_text) != ''
      AND NOT EXISTS (
          SELECT 1 FROM {EMBEDDINGS_TABLE_NAME} we WHERE we.document_id = wd.id
      )
"""

with get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute(UNEMBEDDED_QUERY)
        unembedded_docs = cur.fetchall()  # list[dict], thanks to RealDictCursor

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

from psycopg2.extras import execute_values

if chunk_rows:
    now = datetime.now(timezone.utc)
    insert_data = [
        (
            row["id"],
            row["document_id"],
            row["chunk_index"],
            row["chunk_text"],
            "[" + ",".join(str(float(x)) for x in row["embedding"]) + "]",
            EMBEDDING_MODEL_NAME,
            now,
        )
        for row in chunk_rows
    ]

    insert_sql = f"""
        INSERT INTO {EMBEDDINGS_TABLE_NAME} (
            id, document_id, chunk_index, chunk_text, embedding, model_name, created_at
        ) VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            chunk_text  = EXCLUDED.chunk_text,
            embedding   = EXCLUDED.embedding,
            model_name  = EXCLUDED.model_name,
            created_at  = EXCLUDED.created_at
    """
    template = "(%s, %s, %s, %s, %s::vector, %s, %s)"

    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, insert_sql, insert_data, template=template, page_size=200)
        conn.commit()  # get_connection() does not auto-commit - see lakebase.py

    print(f"✅ Upserted {len(insert_data)} chunk embeddings into {EMBEDDINGS_TABLE_NAME}")
else:
    print("No chunk embeddings to write.")
