"""
Client for the National Weather Service (NWS) API (api.weather.gov).

Resolves locations (city/state strings or lat/lon pairs) to NWS grid points,
fetches active alerts + forecasts, normalizes them into document records, and
upserts them into a Lakebase `weather_documents` table.

Geocoding note: api.weather.gov has no geocoding endpoint of its own, so
"City, ST" strings are resolved to lat/lon via the free US Census Bureau
geocoder (no API key required). Lat/lon pairs are used as-is.

Forecast discussions (AFD text products) are NOT fetched here — that's a
separate /products endpoint keyed by forecast office, not by lat/lon, and
was explicitly out of scope for this version. Instead, "forecast" documents
come from the standard gridpoint forecast (`.../forecast`), one document per
forecast period (e.g. "Tonight", "Wednesday").

ASSUMPTIONS CALLED OUT (please verify against your actual codebase):
  - Uses `from lakebase import get_connection` directly (real import, not a
    reconstruction). Note that lakebase.get_connection() does NOT
    auto-commit — only lakebase.run_write() does — so ensure_table() and
    upsert_documents() call conn.commit() explicitly, same as this module's
    writes should.
  - The Flask endpoint is written as a standalone Blueprint. Wire it into
    your existing app the same way /news/sync is wired in (import + register,
    or copy the route into the existing blueprint/module).
  - Upsert conflict target assumes `id` is the primary key / unique
    constraint on weather_documents.
"""

import hashlib
import os
from datetime import datetime, timezone
from typing import Any, Iterable

import requests
from flask import Blueprint, jsonify, request
from psycopg2.extras import execute_values

from lakebase import get_connection

# Lazy-load sentence transformer for vector search
_embedding_model = None

def get_embedding_model():
    """Lazy-load the sentence transformer model for vector search."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return _embedding_model

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

_BASE_URL = os.environ.get("NWS_API_BASE_URL", "https://api.weather.gov")
_DEFAULT_TIMEOUT = 30

# NWS API *requires* a descriptive User-Agent (ideally app name + contact
# email/URL) — requests without one are frequently throttled/blocked.
_USER_AGENT = os.environ.get(
    "NWS_USER_AGENT", "weather-sync-app (set NWS_USER_AGENT env var)"
)

_CENSUS_GEOCODER_URL = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"


_CREATE_TABLE_SQL = """
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
);
"""

_UPSERT_SQL = """
INSERT INTO weather_documents
    (id, location, source_type, headline, narrative_text, issued_at, effective_at, payload, synced_at)
VALUES %s
ON CONFLICT (id) DO UPDATE SET
    location       = EXCLUDED.location,
    source_type    = EXCLUDED.source_type,
    headline       = EXCLUDED.headline,
    narrative_text = EXCLUDED.narrative_text,
    issued_at      = EXCLUDED.issued_at,
    effective_at   = EXCLUDED.effective_at,
    payload        = EXCLUDED.payload,
    synced_at      = EXCLUDED.synced_at;
"""


def ensure_table() -> None:
    # lakebase.get_connection() does not auto-commit (only run_write does),
    # so DDL/DML issued here must commit explicitly.
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_CREATE_TABLE_SQL)
        conn.commit()


def upsert_documents(docs: Iterable[dict]) -> int:
    """Upsert normalized document records into weather_documents. Returns count upserted."""
    import json

    rows = [
        (
            d["id"],
            d["location"],
            d["source_type"],
            d.get("headline"),
            d.get("narrative_text"),
            d.get("issued_at"),
            d.get("effective_at"),
            json.dumps(d.get("payload")),
            d["synced_at"],
        )
        for d in docs
    ]
    if not rows:
        return 0

    ensure_table()
    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, _UPSERT_SQL, rows)
        conn.commit()
    return len(rows)


# --------------------------------------------------------------------------
# Geocoding: "City, ST" -> (lat, lon) via US Census Bureau geocoder.
# --------------------------------------------------------------------------


def _looks_like_lat_lon(location: str) -> tuple[float, float] | None:
    parts = [p.strip() for p in location.split(",")]
    if len(parts) == 2:
        try:
            lat, lon = float(parts[0]), float(parts[1])
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
        except ValueError:
            pass
    return None


def geocode_location(location: str) -> tuple[float, float]:
    """
    Resolve a location string to (lat, lon).

    Accepts either "lat,lon" (e.g. "41.8781,-87.6298") or a free-form
    address/city-state string (e.g. "Chicago, IL"), which is resolved via
    the free Census Bureau geocoder (no API key required).
    """
    coords = _looks_like_lat_lon(location)
    if coords:
        return coords

    resp = requests.get(
        _CENSUS_GEOCODER_URL,
        params={"address": location, "benchmark": "Public_AR_Current", "format": "json"},
        timeout=_DEFAULT_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    matches = data.get("result", {}).get("addressMatches", [])
    if not matches:
        raise ValueError(f"Could not geocode location: {location!r}")

    coords = matches[0]["coordinates"]
    return float(coords["y"]), float(coords["x"])  # y=lat, x=lon


# --------------------------------------------------------------------------
# WeatherClient — thin wrapper around api.weather.gov
# --------------------------------------------------------------------------


class WeatherClient:
    """Thin wrapper around the NWS API with a User-Agent-configured session."""

    def __init__(self, base_url: str | None = None, timeout: int = _DEFAULT_TIMEOUT):
        self.base_url = (base_url or _BASE_URL).rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": _USER_AGENT,
                "Accept": "application/geo+json",
            }
        )

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        resp = self._session.get(f"{self.base_url}{path}", params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def resolve_point(self, lat: float, lon: float) -> dict:
        """GET /points/{lat},{lon} -> gridId/gridX/gridY, forecast URLs, office, etc."""
        data = self.get(f"/points/{lat},{lon}")
        return data.get("properties", {})

    def get_active_alerts(self, lat: float, lon: float) -> list[dict]:
        """GET /alerts/active?point={lat},{lon} -> list of alert Features."""
        data = self.get("/alerts/active", params={"point": f"{lat},{lon}"})
        return data.get("features", [])

    def get_forecast(self, grid_id: str, grid_x: int, grid_y: int) -> dict:
        """GET /gridpoints/{gridId}/{gridX},{gridY}/forecast -> forecast properties."""
        data = self.get(f"/gridpoints/{grid_id}/{grid_x},{grid_y}/forecast")
        return data.get("properties", {})


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_alert(feature: dict, location: str) -> dict:
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
        "synced_at": _now_iso(),
    }


def normalize_forecast_period(period: dict, location: str, updated_at: str | None) -> dict:
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
        "synced_at": _now_iso(),
    }


# --------------------------------------------------------------------------
# Sync orchestration
# --------------------------------------------------------------------------


def sync_location(client: WeatherClient, location: str, limit: int = 50) -> list[dict]:
    """Fetch + normalize alerts and forecast periods for a single location."""
    lat, lon = geocode_location(location)

    docs: list[dict] = []

    alerts = client.get_active_alerts(lat, lon)
    for feature in alerts[:limit]:
        docs.append(normalize_alert(feature, location))

    point = client.resolve_point(lat, lon)
    grid_id, grid_x, grid_y = point.get("gridId"), point.get("gridX"), point.get("gridY")
    if grid_id is not None and grid_x is not None and grid_y is not None:
        forecast = client.get_forecast(grid_id, grid_x, grid_y)
        updated_at = forecast.get("updated")
        periods = forecast.get("periods", [])
        for period in periods[:limit]:
            docs.append(normalize_forecast_period(period, location, updated_at))

    return docs


def sync_weather(locations: list[str], limit: int = 50) -> int:
    """Sync alerts + forecasts for all given locations into weather_documents. Returns count synced."""
    client = WeatherClient()
    all_docs: list[dict] = []
    for location in locations:
        all_docs.extend(sync_location(client, location, limit=limit))
    return upsert_documents(all_docs)


# --------------------------------------------------------------------------
# Flask endpoint — mirrors /news/sync
# --------------------------------------------------------------------------

weather_bp = Blueprint("weather", __name__)


@weather_bp.route("/weather/sync", methods=["POST"])
def weather_sync():
    """
    POST /weather/sync
    Body: {"locations": ["Chicago, IL", "Austin, TX"], "limit": 50}

    Fetches active alerts + forecasts for each location, normalizes them,
    and upserts into weather_documents. Returns a count of documents synced.
    """
    body = request.get_json(force=True, silent=True) or {}
    locations = body.get("locations")
    limit = body.get("limit", 50)

    if not locations or not isinstance(locations, list):
        return jsonify({"error": "'locations' must be a non-empty list of strings"}), 400

    try:
        count = sync_weather(locations, limit=limit)
    except Exception as exc:  # noqa: BLE001 - surface a clean error to the caller
        return jsonify({"error": str(exc)}), 500

    return jsonify({"synced": count})


@weather_bp.route("/weather/locations", methods=["GET"])
def get_weather_locations():
    """GET /weather/locations - Returns list of unique locations we have weather data for."""
    ensure_table()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT location FROM weather_documents ORDER BY location"
            )
            rows = cur.fetchall()
    return jsonify([{"location": row[0]} for row in rows])


@weather_bp.route("/weather/city/<path:location>", methods=["GET"])
def get_city_weather(location: str):
    """GET /weather/city/{location} - Returns alerts and forecasts for a specific location."""
    ensure_table()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT source_type, headline, narrative_text, issued_at, effective_at, payload
                FROM weather_documents
                WHERE location = %s
                ORDER BY effective_at DESC NULLS LAST, issued_at DESC NULLS LAST
                """,
                (location,),
            )
            rows = cur.fetchall()

    alerts = []
    forecasts = []
    for row in rows:
        source_type, headline, narrative, issued_at, effective_at, payload = row
        doc = {
            "source_type": source_type,
            "headline": headline,
            "narrative_text": narrative,
            "issued_at": issued_at.isoformat() if issued_at else None,
            "effective_at": effective_at.isoformat() if effective_at else None,
            "payload": payload,
        }
        if source_type == "alert":
            alerts.append(doc)
        elif source_type == "forecast":
            forecasts.append(doc)

    return jsonify({"location": location, "alerts": alerts, "forecasts": forecasts})


@weather_bp.route("/weather/search", methods=["GET"])
def search_weather():
    """
    GET /weather/search?query=<text>&limit=<N>
    
    Vector similarity search over weather document embeddings.
    Embeds the query and returns the top-K most relevant chunks.
    """
    query = request.args.get("query", "")
    limit = int(request.args.get("limit", 10))
    
    if not query or not query.strip():
        return jsonify({"error": "query parameter is required"}), 400
    
    try:
        # Embed the query using the same model as the embeddings pipeline
        model = get_embedding_model()
        query_embedding = model.encode(query).tolist()
        
        # Format embedding as a string for pgvector: "[0.1, 0.2, ...]"
        embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
        
        # Query the embeddings table using cosine similarity
        # pgvector's <=> operator is cosine distance (1 - cosine similarity)
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 
                        e.document_id,
                        e.chunk_index,
                        e.chunk_text,
                        d.location,
                        d.source_type,
                        d.headline,
                        d.effective_at,
                        1 - (e.embedding <=> %s::vector) AS similarity
                    FROM weather_embeddings e
                    JOIN weather_documents d ON e.document_id = d.id
                    ORDER BY e.embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (embedding_str, embedding_str, limit),
                )
                rows = cur.fetchall()
        
        results = []
        for row in rows:
            doc_id, chunk_idx, chunk_text, location, source_type, headline, effective_at, similarity = row
            results.append({
                "document_id": doc_id,
                "chunk_index": chunk_idx,
                "chunk_text": chunk_text,
                "location": location,
                "source_type": source_type,
                "headline": headline,
                "effective_at": effective_at.isoformat() if effective_at else None,
                "similarity": float(similarity),
            })
        
        return jsonify({"query": query, "results": results, "count": len(results)})
    
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
