"""
Resource monitoring for optimization runs.

Tracks CPU, memory, and performance metrics during optimization.
"""

import logging
import psutil
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    """Snapshot of resource usage at a point in time."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    num_threads: int
    num_processes: int


class ResourceMonitor:
    """
    Monitors system resource usage during optimization.
    
    Tracks CPU, memory, and other metrics to help identify
    performance bottlenecks and resource constraints.
    """
    
    def __init__(self):
        """Initialize resource monitor."""
        self.process = psutil.Process()
        self.start_time: Optional[float] = None
        self.snapshots: list = []
        self.enabled = True
    
    def start(self) -> None:
        """Start monitoring."""
        self.start_time = time.time()
        self.snapshots = []
        logger.info("Resource monitoring started")
    
    def capture_snapshot(self) -> ResourceSnapshot:
        """Capture current resource usage."""
        try:
            # Get process info
            with self.process.oneshot():
                cpu_percent = self.process.cpu_percent(interval=0.1)
                memory_info = self.process.memory_info()
                num_threads = self.process.num_threads()
            
            # Get system memory info
            memory = psutil.virtual_memory()
            
            snapshot = ResourceSnapshot(
                timestamp=datetime.now(timezone.utc),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_mb=memory_info.rss / (1024 * 1024),  # Convert to MB
                num_threads=num_threads,
                num_processes=len(self.process.children(recursive=True)) + 1
            )
            
            if self.enabled:
                self.snapshots.append(snapshot)
            
            return snapshot
            
        except Exception as e:
            logger.warning(f"Failed to capture resource snapshot: {e}")
            return ResourceSnapshot(
                timestamp=datetime.now(timezone.utc),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_mb=0.0,
                num_threads=0,
                num_processes=0
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get resource usage statistics.
        
        Returns:
            Dictionary with min, max, avg, current metrics
        """
        if not self.snapshots:
            return {
                'enabled': self.enabled,
                'snapshots_count': 0,
                'message': 'No snapshots captured yet'
            }
        
        cpu_values = [s.cpu_percent for s in self.snapshots]
        memory_values = [s.memory_mb for s in self.snapshots]
        
        current = self.snapshots[-1]
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        return {
            'enabled': self.enabled,
            'snapshots_count': len(self.snapshots),
            'elapsed_seconds': elapsed,
            'cpu': {
                'current': current.cpu_percent,
                'min': min(cpu_values),
                'max': max(cpu_values),
                'avg': sum(cpu_values) / len(cpu_values)
            },
            'memory_mb': {
                'current': current.memory_mb,
                'min': min(memory_values),
                'max': max(memory_values),
                'avg': sum(memory_values) / len(memory_values)
            },
            'system_memory_percent': current.memory_percent,
            'threads': current.num_threads,
            'processes': current.num_processes
        }
    
    def get_current_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        snapshot = self.capture_snapshot()
        return {
            'timestamp': snapshot.timestamp.isoformat(),
            'cpu_percent': snapshot.cpu_percent,
            'memory_mb': snapshot.memory_mb,
            'memory_percent': snapshot.memory_percent,
            'threads': snapshot.num_threads,
            'processes': snapshot.num_processes
        }
    
    def check_limits(
        self,
        max_memory_mb: Optional[float] = None,
        max_cpu_percent: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Check if resource usage exceeds limits.
        
        Args:
            max_memory_mb: Maximum memory in MB
            max_cpu_percent: Maximum CPU usage percent
        
        Returns:
            Dictionary with warnings if limits exceeded
        """
        current = self.capture_snapshot()
        warnings = []
        
        if max_memory_mb and current.memory_mb > max_memory_mb:
            warnings.append({
                'type': 'memory',
                'message': f'Memory usage ({current.memory_mb:.1f} MB) exceeds limit ({max_memory_mb} MB)',
                'current': current.memory_mb,
                'limit': max_memory_mb
            })
        
        if max_cpu_percent and current.cpu_percent > max_cpu_percent:
            warnings.append({
                'type': 'cpu',
                'message': f'CPU usage ({current.cpu_percent:.1f}%) exceeds limit ({max_cpu_percent}%)',
                'current': current.cpu_percent,
                'limit': max_cpu_percent
            })
        
        return {
            'within_limits': len(warnings) == 0,
            'warnings': warnings,
            'current_usage': asdict(current)
        }
    
    def enable(self) -> None:
        """Enable monitoring."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable monitoring."""
        self.enabled = False
    
    def reset(self) -> None:
        """Reset monitoring data."""
        self.snapshots = []
        self.start_time = None
        logger.info("Resource monitoring reset")


class OptimizationMonitor:
    """
    High-level monitor for optimization runs.
    
    Combines resource monitoring with progress tracking.
    """
    
    def __init__(self, run_id: int, total_combinations: Optional[int] = None):
        """
        Initialize optimization monitor.
        
        Args:
            run_id: Optimization run ID
            total_combinations: Total number of combinations to test
        """
        self.run_id = run_id
        self.total_combinations = total_combinations
        self.completed_combinations = 0
        self.resource_monitor = ResourceMonitor()
        self.start_time: Optional[datetime] = None
    
    def start(self) -> None:
        """Start monitoring."""
        self.start_time = datetime.now(timezone.utc)
        self.completed_combinations = 0
        self.resource_monitor.start()
        logger.info(f"Optimization monitoring started for run {self.run_id}")
    
    def update_progress(self, completed: int) -> None:
        """
        Update progress and capture resource snapshot.
        
        Args:
            completed: Number of completed combinations
        """
        self.completed_combinations = completed
        self.resource_monitor.capture_snapshot()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status including progress and resources."""
        elapsed = (
            (datetime.now(timezone.utc) - self.start_time).total_seconds()
            if self.start_time else 0
        )
        
        progress_pct = (
            (self.completed_combinations / self.total_combinations) * 100
            if self.total_combinations else None
        )
        
        # Estimate time remaining
        eta_seconds = None
        if self.total_combinations and self.completed_combinations > 0:
            rate = self.completed_combinations / elapsed if elapsed > 0 else 0
            remaining = self.total_combinations - self.completed_combinations
            eta_seconds = remaining / rate if rate > 0 else None
        
        return {
            'run_id': self.run_id,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'elapsed_seconds': elapsed,
            'elapsed_formatted': f"{int(elapsed // 60)}m {int(elapsed % 60)}s",
            'progress': {
                'completed': self.completed_combinations,
                'total': self.total_combinations,
                'percent': progress_pct,
                'remaining': self.total_combinations - self.completed_combinations if self.total_combinations else None
            },
            'eta': {
                'seconds': eta_seconds,
                'formatted': f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s" if eta_seconds else None
            },
            'resources': self.resource_monitor.get_current_usage(),
            'resource_stats': self.resource_monitor.get_stats()
        }


def get_system_info() -> Dict[str, Any]:
    """Get general system information."""
    return {
        'cpu': {
            'count': psutil.cpu_count(),
            'count_physical': psutil.cpu_count(logical=False),
            'percent': psutil.cpu_percent(interval=1, percpu=False),
            'percent_per_cpu': psutil.cpu_percent(interval=1, percpu=True)
        },
        'memory': {
            'total_mb': psutil.virtual_memory().total / (1024 * 1024),
            'available_mb': psutil.virtual_memory().available / (1024 * 1024),
            'used_mb': psutil.virtual_memory().used / (1024 * 1024),
            'percent': psutil.virtual_memory().percent
        },
        'disk': {
            'total_gb': psutil.disk_usage('/').total / (1024 ** 3),
            'used_gb': psutil.disk_usage('/').used / (1024 ** 3),
            'free_gb': psutil.disk_usage('/').free / (1024 ** 3),
            'percent': psutil.disk_usage('/').percent
        }
    }

