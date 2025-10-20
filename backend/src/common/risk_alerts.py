"""
Risk alert notification system.

This module provides functions to send alerts for risk violations.
Currently supports logging to database and stdout. Can be extended to support:
- Email notifications
- Slack/Discord webhooks
- SMS alerts
- PagerDuty integration
"""

import asyncio
from typing import Optional, Dict, Any
from datetime import datetime, timezone
from common.logging import get_logger
from common.db import get_db_session

logger = get_logger(__name__)


class RiskAlertManager:
    """Manage risk alerts and notifications"""
    
    def __init__(self):
        self.alert_handlers = [
            self._log_alert,
            self._database_alert,
        ]
        # Future: Add email, slack, sms handlers here
    
    async def send_alert(self, 
                        alert_type: str,
                        severity: str,
                        message: str,
                        violation_id: Optional[int] = None,
                        metadata: Optional[Dict[str, Any]] = None):
        """
        Send a risk alert through all configured channels.
        
        Args:
            alert_type: Type of alert (violation, emergency_stop, limit_breach, etc)
            severity: Alert severity (info, warning, critical)
            message: Human-readable alert message
            violation_id: Associated risk violation ID
            metadata: Additional context data
        """
        alert_data = {
            'alert_type': alert_type,
            'severity': severity,
            'message': message,
            'violation_id': violation_id,
            'metadata': metadata,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Execute all alert handlers
        tasks = [handler(alert_data) for handler in self.alert_handlers]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _log_alert(self, alert_data: Dict[str, Any]):
        """Log alert to application logs"""
        severity = alert_data['severity']
        message = alert_data['message']
        alert_type = alert_data['alert_type']
        
        if severity == 'critical':
            logger.critical(f"🚨 RISK ALERT [{alert_type.upper()}]: {message}", extra=alert_data)
        elif severity == 'warning':
            logger.warning(f"⚠️ RISK ALERT [{alert_type.upper()}]: {message}", extra=alert_data)
        else:
            logger.info(f"ℹ️ RISK ALERT [{alert_type.upper()}]: {message}", extra=alert_data)
    
    async def _database_alert(self, alert_data: Dict[str, Any]):
        """Store alert in database logs table"""
        try:
            with get_db_session() as session:
                from common.models import LogEntry
                
                log_entry = LogEntry(
                    service='risk_manager',
                    level='CRITICAL' if alert_data['severity'] == 'critical' else 'WARNING',
                    msg=f"RISK ALERT: {alert_data['message']}",
                    ts=datetime.now(timezone.utc),
                    meta_json=alert_data
                )
                session.add(log_entry)
                session.commit()
        except Exception as e:
            logger.error(f"Failed to store risk alert in database: {e}")
    
    # Extension points for future notification channels
    async def _email_alert(self, alert_data: Dict[str, Any]):
        """Send email notification (placeholder for future implementation)"""
        # TODO: Implement SMTP email sending
        pass
    
    async def _slack_alert(self, alert_data: Dict[str, Any]):
        """Send Slack webhook notification (placeholder for future implementation)"""
        # TODO: Implement Slack webhook
        pass
    
    async def _sms_alert(self, alert_data: Dict[str, Any]):
        """Send SMS notification (placeholder for future implementation)"""
        # TODO: Implement Twilio SMS sending
        pass


# Singleton instance
_alert_manager = None

def get_alert_manager() -> RiskAlertManager:
    """Get or create the global alert manager instance"""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = RiskAlertManager()
    return _alert_manager


# Convenience functions
async def send_risk_alert(alert_type: str, severity: str, message: str, 
                         violation_id: Optional[int] = None, 
                         metadata: Optional[Dict[str, Any]] = None):
    """Send a risk alert (convenience function)"""
    manager = get_alert_manager()
    await manager.send_alert(alert_type, severity, message, violation_id, metadata)


async def send_emergency_alert(message: str, metadata: Optional[Dict[str, Any]] = None):
    """Send a critical emergency alert"""
    manager = get_alert_manager()
    await manager.send_alert(
        alert_type='emergency_stop',
        severity='critical',
        message=message,
        metadata=metadata
    )


async def send_violation_alert(violation_type: str, message: str, 
                               violation_id: int, severity: str = 'warning',
                               metadata: Optional[Dict[str, Any]] = None):
    """Send an alert for a risk violation"""
    manager = get_alert_manager()
    await manager.send_alert(
        alert_type=violation_type,
        severity=severity,
        message=message,
        violation_id=violation_id,
        metadata=metadata
    )

