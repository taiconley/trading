"""
Helpers for tracking end-to-end data synchronization latency.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from .db import execute_with_retry
from .models import DataSyncStatus


def _lag_ms(start: Optional[datetime], end: Optional[datetime]) -> Optional[int]:
    if not start or not end:
        return None
    return int((end - start).total_seconds() * 1000)


def update_sync_status(
    category: str,
    *,
    source_ts: Optional[datetime] = None,
    db_ts: Optional[datetime] = None,
    frontend_ts: Optional[datetime] = None,
    note: Optional[str] = None
) -> DataSyncStatus:
    """
    Persist the latest timestamps for a given data category and compute lag metrics.
    """
    now = datetime.now(timezone.utc)

    def _upsert(session):
        status = session.query(DataSyncStatus).filter(
            DataSyncStatus.category == category
        ).first()
        if not status:
            status = DataSyncStatus(category=category)
            session.add(status)
        
        if source_ts:
            status.source_ts = source_ts
        if db_ts:
            status.db_ts = db_ts
        if frontend_ts:
            status.frontend_ts = frontend_ts
        if note is not None:
            status.note = note
        
        status.source_to_db_ms = _lag_ms(status.source_ts, status.db_ts)
        status.db_to_frontend_ms = _lag_ms(status.db_ts, status.frontend_ts)
        status.source_to_frontend_ms = _lag_ms(status.source_ts, status.frontend_ts)
        status.updated_at = now
        
        session.commit()
        session.refresh(status)
        return status
    
    return execute_with_retry(_upsert)


def get_sync_status(category: str) -> Optional[Dict[str, Any]]:
    """
    Fetch sync status for a category as a serializable dict.
    """
    def _fetch(session):
        status = session.query(DataSyncStatus).filter(
            DataSyncStatus.category == category
        ).first()
        if not status:
            return None
        return serialize_sync_status(status)
    
    return execute_with_retry(_fetch)


def list_sync_statuses() -> List[Dict[str, Any]]:
    """
    Return sync statuses for all categories.
    """
    def _fetch(session):
        statuses = session.query(DataSyncStatus).order_by(DataSyncStatus.category).all()
        return [serialize_sync_status(status) for status in statuses]
    
    return execute_with_retry(_fetch)


def serialize_sync_status(status: DataSyncStatus) -> Dict[str, Any]:
    """Serialize a DataSyncStatus ORM object."""
    return {
        "category": status.category,
        "source_ts": status.source_ts.isoformat() if status.source_ts else None,
        "db_ts": status.db_ts.isoformat() if status.db_ts else None,
        "frontend_ts": status.frontend_ts.isoformat() if status.frontend_ts else None,
        "source_to_db_ms": status.source_to_db_ms,
        "db_to_frontend_ms": status.db_to_frontend_ms,
        "source_to_frontend_ms": status.source_to_frontend_ms,
        "updated_at": status.updated_at.isoformat() if status.updated_at else None,
        "note": status.note,
    }
