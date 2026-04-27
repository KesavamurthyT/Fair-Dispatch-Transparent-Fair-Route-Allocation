"""
API Key management endpoints.
Generate, list, revoke, and check usage of API keys.
"""

import hashlib
import secrets
import uuid
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, func, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.api_key import ApiKey, ApiKeyUsageLog


router = APIRouter(prefix="/api-keys", tags=["API Keys"])


# ─── Pydantic Schemas ────────────────────────────────────────────────────────

class CreateApiKeyRequest(BaseModel):
    """Request to create a new API key."""
    name: str = Field(..., min_length=1, max_length=100, description="Human-readable label")
    expires_in_days: Optional[int] = Field(None, ge=1, le=365, description="Days until expiry")


class CreateApiKeyResponse(BaseModel):
    """Response after creating an API key (only time full key is shown)."""
    id: str
    name: str
    key: str  # The full API key — shown only once
    key_prefix: str
    created_at: str
    expires_at: Optional[str] = None
    message: str = "Store this key securely. It will not be shown again."


class ApiKeyInfo(BaseModel):
    """Public API key info (never includes the full key)."""
    id: str
    name: str
    key_prefix: str
    is_active: bool
    total_requests: int
    total_allocations: int
    created_at: str
    last_used_at: Optional[str] = None
    expires_at: Optional[str] = None


class ApiKeyListResponse(BaseModel):
    """List of all API keys."""
    keys: List[ApiKeyInfo]
    total: int
    active_count: int


class UsageLogEntry(BaseModel):
    """Single usage log entry."""
    endpoint: str
    method: str
    status_code: int
    response_time_ms: Optional[int] = None
    timestamp: str


class DailyUsage(BaseModel):
    """Usage count for a single day."""
    date: str
    count: int


class EndpointBreakdown(BaseModel):
    """Usage broken down by endpoint."""
    endpoint: str
    count: int
    avg_response_ms: Optional[float] = None


class ApiKeyUsageResponse(BaseModel):
    """Detailed usage stats for a single API key."""
    key_id: str
    key_name: str
    key_prefix: str
    total_requests: int
    total_allocations: int
    last_used_at: Optional[str] = None
    daily_usage: List[DailyUsage]
    endpoint_breakdown: List[EndpointBreakdown]
    recent_logs: List[UsageLogEntry]


class ApiKeyStatsResponse(BaseModel):
    """Global API key statistics."""
    total_keys: int
    active_keys: int
    total_requests_all: int
    requests_last_24h: int


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _generate_api_key() -> str:
    """Generate a cryptographically secure API key with fd_ prefix."""
    raw = secrets.token_hex(24)  # 48 hex chars
    return f"fd_{raw}"


def _hash_key(key: str) -> str:
    """SHA-256 hash of the full API key."""
    return hashlib.sha256(key.encode()).hexdigest()


def _dt_to_str(dt: Optional[datetime]) -> Optional[str]:
    """Convert datetime to ISO string or None."""
    return dt.isoformat() if dt else None


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/stats", response_model=ApiKeyStatsResponse)
async def get_api_key_stats(db: AsyncSession = Depends(get_db)):
    """Get global API key statistics for the dashboard header."""
    # Total & active keys
    total_result = await db.execute(select(func.count(ApiKey.id)))
    total_keys = total_result.scalar() or 0

    active_result = await db.execute(
        select(func.count(ApiKey.id)).where(ApiKey.is_active == True)
    )
    active_keys = active_result.scalar() or 0

    # Total requests across all keys
    total_req_result = await db.execute(select(func.coalesce(func.sum(ApiKey.total_requests), 0)))
    total_requests_all = total_req_result.scalar() or 0

    # Requests in last 24h
    cutoff = datetime.utcnow() - timedelta(hours=24)
    last_24h_result = await db.execute(
        select(func.count(ApiKeyUsageLog.id))
        .where(ApiKeyUsageLog.timestamp >= cutoff)
    )
    requests_last_24h = last_24h_result.scalar() or 0

    return ApiKeyStatsResponse(
        total_keys=total_keys,
        active_keys=active_keys,
        total_requests_all=total_requests_all,
        requests_last_24h=requests_last_24h,
    )


@router.post("", response_model=CreateApiKeyResponse, status_code=201)
async def create_api_key(
    body: CreateApiKeyRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Generate a new API key.
    The full key is returned **only once** in this response.
    """
    # Generate key
    full_key = _generate_api_key()
    key_hash = _hash_key(full_key)
    key_prefix = full_key[:11]  # "fd_" + 8 hex chars

    # Calculate expiry
    expires_at = None
    if body.expires_in_days:
        expires_at = datetime.utcnow() + timedelta(days=body.expires_in_days)

    # Persist
    api_key = ApiKey(
        name=body.name,
        key_prefix=key_prefix,
        key_hash=key_hash,
        is_active=True,
        expires_at=expires_at,
    )
    db.add(api_key)
    await db.flush()  # Get the ID

    return CreateApiKeyResponse(
        id=str(api_key.id),
        name=api_key.name,
        key=full_key,
        key_prefix=key_prefix,
        created_at=_dt_to_str(api_key.created_at),
        expires_at=_dt_to_str(expires_at),
    )


@router.get("", response_model=ApiKeyListResponse)
async def list_api_keys(db: AsyncSession = Depends(get_db)):
    """List all API keys (never returns the full key)."""
    result = await db.execute(
        select(ApiKey).order_by(desc(ApiKey.created_at))
    )
    keys = result.scalars().all()

    items = [
        ApiKeyInfo(
            id=str(k.id),
            name=k.name,
            key_prefix=k.key_prefix,
            is_active=k.is_active,
            total_requests=k.total_requests,
            total_allocations=k.total_allocations,
            created_at=_dt_to_str(k.created_at),
            last_used_at=_dt_to_str(k.last_used_at),
            expires_at=_dt_to_str(k.expires_at),
        )
        for k in keys
    ]

    active_count = sum(1 for k in keys if k.is_active)

    return ApiKeyListResponse(
        keys=items,
        total=len(items),
        active_count=active_count,
    )


@router.get("/{key_id}/usage", response_model=ApiKeyUsageResponse)
async def get_api_key_usage(
    key_id: str,
    days: int = Query(7, ge=1, le=90, description="Days of history"),
    db: AsyncSession = Depends(get_db),
):
    """Get detailed usage statistics for a specific API key."""
    try:
        key_uuid = uuid.UUID(key_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid key ID format")

    # Fetch the key
    result = await db.execute(select(ApiKey).where(ApiKey.id == key_uuid))
    api_key = result.scalar_one_or_none()
    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")

    cutoff = datetime.utcnow() - timedelta(days=days)

    # Daily usage (group by date)
    # For SQLite compatibility, cast to date string
    daily_result = await db.execute(
        select(
            func.date(ApiKeyUsageLog.timestamp).label("day"),
            func.count(ApiKeyUsageLog.id).label("cnt"),
        )
        .where(
            and_(
                ApiKeyUsageLog.api_key_id == key_uuid,
                ApiKeyUsageLog.timestamp >= cutoff,
            )
        )
        .group_by(func.date(ApiKeyUsageLog.timestamp))
        .order_by(func.date(ApiKeyUsageLog.timestamp))
    )
    daily_rows = daily_result.all()
    daily_usage = [
        DailyUsage(date=str(row.day), count=row.cnt)
        for row in daily_rows
    ]

    # Fill missing days with zero
    all_dates = set()
    for i in range(days):
        d = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
        all_dates.add(d)
    existing_dates = {du.date for du in daily_usage}
    for d in sorted(all_dates - existing_dates):
        daily_usage.append(DailyUsage(date=d, count=0))
    daily_usage.sort(key=lambda x: x.date)

    # Endpoint breakdown
    endpoint_result = await db.execute(
        select(
            ApiKeyUsageLog.endpoint,
            func.count(ApiKeyUsageLog.id).label("cnt"),
            func.avg(ApiKeyUsageLog.response_time_ms).label("avg_ms"),
        )
        .where(
            and_(
                ApiKeyUsageLog.api_key_id == key_uuid,
                ApiKeyUsageLog.timestamp >= cutoff,
            )
        )
        .group_by(ApiKeyUsageLog.endpoint)
        .order_by(desc("cnt"))
    )
    endpoint_rows = endpoint_result.all()
    endpoint_breakdown = [
        EndpointBreakdown(
            endpoint=row.endpoint,
            count=row.cnt,
            avg_response_ms=round(float(row.avg_ms), 1) if row.avg_ms else None,
        )
        for row in endpoint_rows
    ]

    # Recent logs (last 20)
    logs_result = await db.execute(
        select(ApiKeyUsageLog)
        .where(ApiKeyUsageLog.api_key_id == key_uuid)
        .order_by(desc(ApiKeyUsageLog.timestamp))
        .limit(20)
    )
    logs = logs_result.scalars().all()
    recent_logs = [
        UsageLogEntry(
            endpoint=log.endpoint,
            method=log.method,
            status_code=log.status_code,
            response_time_ms=log.response_time_ms,
            timestamp=_dt_to_str(log.timestamp),
        )
        for log in logs
    ]

    return ApiKeyUsageResponse(
        key_id=str(api_key.id),
        key_name=api_key.name,
        key_prefix=api_key.key_prefix,
        total_requests=api_key.total_requests,
        total_allocations=api_key.total_allocations,
        last_used_at=_dt_to_str(api_key.last_used_at),
        daily_usage=daily_usage,
        endpoint_breakdown=endpoint_breakdown,
        recent_logs=recent_logs,
    )


@router.delete("/{key_id}")
async def revoke_api_key(
    key_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Revoke (soft-delete) an API key."""
    try:
        key_uuid = uuid.UUID(key_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid key ID format")

    result = await db.execute(select(ApiKey).where(ApiKey.id == key_uuid))
    api_key = result.scalar_one_or_none()
    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")

    if not api_key.is_active:
        raise HTTPException(status_code=400, detail="Key is already revoked")

    api_key.is_active = False

    return {"status": "revoked", "key_id": str(api_key.id), "name": api_key.name}
