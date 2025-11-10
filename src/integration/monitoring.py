"""
Comprehensive monitoring and logging system for the occupation data reports application.
Provides performance metrics, error tracking, and system health monitoring.
"""

import logging
import logging.handlers
import time
import threading

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json


@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    category: str = "general"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ErrorEvent:
    """Error event data structure."""
    timestamp: datetime
    level: str
    message: str
    exception_type: str
    module: str
    function: str
    line_number: int
    stack_trace: str
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}


@dataclass
class SystemHealth:
    """System health metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    active_threads: int
    open_files: int


class PerformanceMonitor:
    """
    Performance monitoring system that tracks metrics, system resources,
    and provides real-time monitoring capabilities.
    """
    
    def __init__(self, max_metrics: int = 10000, collection_interval: float = 1.0):
        """
        Initialize the performance monitor.
        
        Args:
            max_metrics: Maximum number of metrics to keep in memory
            collection_interval: Interval for system metrics collection (seconds)
        """
        self.max_metrics = max_metrics
        self.collection_interval = collection_interval
        self.logger = logging.getLogger(__name__)
        
        # Metric storage
        self.metrics: deque = deque(maxlen=max_metrics)
        self.error_events: deque = deque(maxlen=1000)
        self.system_health: deque = deque(maxlen=1000)
        
        # Performance tracking
        self.operation_timers: Dict[str, float] = {}
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self.operation_durations: Dict[str, List[float]] = defaultdict(list)
        
        # System monitoring
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        
        # Callbacks for real-time monitoring
        self.metric_callbacks: List[Callable] = []
        self.error_callbacks: List[Callable] = []
        self.health_callbacks: List[Callable] = []
    
    def start_monitoring(self):
        """Start system resource monitoring."""
        if self.monitoring_active:
            self.logger.warning("Monitoring is already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop system resource monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop for system metrics collection."""
        while self.monitoring_active:
            try:
                # Collect system health metrics
                health = self._collect_system_health()
                
                with self.lock:
                    self.system_health.append(health)
                
                # Notify health callbacks
                for callback in self.health_callbacks:
                    try:
                        callback(health)
                    except Exception as e:
                        self.logger.warning(f"Health callback failed: {str(e)}")
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {str(e)}")
                time.sleep(self.collection_interval)
    
    def _collect_system_health(self) -> SystemHealth:
        """Collect current system health metrics."""
        try:
            if not PSUTIL_AVAILABLE:
                # Return default values when psutil is not available
                return SystemHealth(
                    timestamp=datetime.now(),
                    cpu_percent=0.0,
                    memory_percent=0.0,
                    memory_used_mb=0.0,
                    memory_available_mb=0.0,
                    disk_usage_percent=0.0,
                    disk_free_gb=0.0,
                    active_threads=1,
                    open_files=0
                )
            
            # CPU and memory metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Process metrics
            process = psutil.Process()
            active_threads = process.num_threads()
            
            try:
                open_files = len(process.open_files())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                open_files = 0
            
            return SystemHealth(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=disk.percent,
                disk_free_gb=disk.free / (1024 * 1024 * 1024),
                active_threads=active_threads,
                open_files=open_files
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect system health: {str(e)}")
            return SystemHealth(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                memory_available_mb=0.0,
                disk_usage_percent=0.0,
                disk_free_gb=0.0,
                active_threads=0,
                open_files=0
            )
    
    def record_metric(self, name: str, value: float, unit: str = "", 
                     category: str = "general", metadata: Optional[Dict[str, Any]] = None):
        """
        Record a performance metric.
        
        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
            category: Metric category
            metadata: Additional metadata
        """
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            category=category,
            metadata=metadata or {}
        )
        
        with self.lock:
            self.metrics.append(metric)
        
        # Notify metric callbacks
        for callback in self.metric_callbacks:
            try:
                callback(metric)
            except Exception as e:
                self.logger.warning(f"Metric callback failed: {str(e)}")
    
    def start_timer(self, operation_name: str) -> str:
        """
        Start timing an operation.
        
        Args:
            operation_name: Name of the operation to time
            
        Returns:
            Timer ID for stopping the timer
        """
        timer_id = f"{operation_name}_{int(time.time() * 1000000)}"
        self.operation_timers[timer_id] = time.time()
        return timer_id
    
    def stop_timer(self, timer_id: str, operation_name: Optional[str] = None) -> float:
        """
        Stop timing an operation and record the duration.
        
        Args:
            timer_id: Timer ID returned by start_timer
            operation_name: Optional operation name override
            
        Returns:
            Duration in seconds
        """
        if timer_id not in self.operation_timers:
            self.logger.warning(f"Timer {timer_id} not found")
            return 0.0
        
        start_time = self.operation_timers.pop(timer_id)
        duration = time.time() - start_time
        
        # Extract operation name from timer ID if not provided
        if operation_name is None:
            operation_name = timer_id.split('_')[0]
        
        # Record operation metrics
        with self.lock:
            self.operation_counts[operation_name] += 1
            self.operation_durations[operation_name].append(duration)
            
            # Keep only recent durations to prevent memory growth
            if len(self.operation_durations[operation_name]) > 1000:
                self.operation_durations[operation_name] = \
                    self.operation_durations[operation_name][-500:]
        
        # Record as metric
        self.record_metric(
            name=f"{operation_name}_duration",
            value=duration,
            unit="seconds",
            category="performance",
            metadata={"operation": operation_name}
        )
        
        return duration
    
    def record_error(self, error: Exception, context: Optional[Dict[str, Any]] = None):
        """
        Record an error event.
        
        Args:
            error: Exception that occurred
            context: Additional context information
        """
        import traceback
        import inspect
        
        # Get caller information
        frame = inspect.currentframe().f_back
        module = frame.f_globals.get('__name__', 'unknown')
        function = frame.f_code.co_name
        line_number = frame.f_lineno
        
        error_event = ErrorEvent(
            timestamp=datetime.now(),
            level="ERROR",
            message=str(error),
            exception_type=type(error).__name__,
            module=module,
            function=function,
            line_number=line_number,
            stack_trace=traceback.format_exc(),
            context=context or {}
        )
        
        with self.lock:
            self.error_events.append(error_event)
        
        # Notify error callbacks
        for callback in self.error_callbacks:
            try:
                callback(error_event)
            except Exception as e:
                self.logger.warning(f"Error callback failed: {str(e)}")
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            Dictionary containing operation statistics
        """
        with self.lock:
            durations = self.operation_durations.get(operation_name, [])
            count = self.operation_counts.get(operation_name, 0)
        
        if not durations:
            return {
                'operation': operation_name,
                'count': count,
                'total_duration': 0.0,
                'average_duration': 0.0,
                'min_duration': 0.0,
                'max_duration': 0.0
            }
        
        total_duration = sum(durations)
        average_duration = total_duration / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        
        return {
            'operation': operation_name,
            'count': count,
            'total_duration': total_duration,
            'average_duration': average_duration,
            'min_duration': min_duration,
            'max_duration': max_duration,
            'recent_samples': len(durations)
        }
    
    def get_all_operation_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all tracked operations."""
        stats = {}
        
        with self.lock:
            operation_names = set(self.operation_counts.keys()) | set(self.operation_durations.keys())
        
        for operation_name in operation_names:
            stats[operation_name] = self.get_operation_stats(operation_name)
        
        return stats
    
    def get_recent_metrics(self, category: Optional[str] = None, 
                          minutes: int = 60) -> List[PerformanceMetric]:
        """
        Get recent metrics within the specified time window.
        
        Args:
            category: Optional category filter
            minutes: Time window in minutes
            
        Returns:
            List of recent metrics
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self.lock:
            recent_metrics = [
                metric for metric in self.metrics
                if metric.timestamp >= cutoff_time
            ]
        
        if category:
            recent_metrics = [m for m in recent_metrics if m.category == category]
        
        return recent_metrics
    
    def get_recent_errors(self, minutes: int = 60) -> List[ErrorEvent]:
        """
        Get recent error events within the specified time window.
        
        Args:
            minutes: Time window in minutes
            
        Returns:
            List of recent error events
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self.lock:
            return [
                error for error in self.error_events
                if error.timestamp >= cutoff_time
            ]
    
    def get_system_health_summary(self, minutes: int = 60) -> Dict[str, Any]:
        """
        Get system health summary for the specified time window.
        
        Args:
            minutes: Time window in minutes
            
        Returns:
            Dictionary containing health summary
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        
        with self.lock:
            recent_health = [
                health for health in self.system_health
                if health.timestamp >= cutoff_time
            ]
        
        if not recent_health:
            return {
                'samples': 0,
                'time_window_minutes': minutes
            }
        
        # Calculate averages and extremes
        cpu_values = [h.cpu_percent for h in recent_health]
        memory_values = [h.memory_percent for h in recent_health]
        disk_values = [h.disk_usage_percent for h in recent_health]
        
        return {
            'samples': len(recent_health),
            'time_window_minutes': minutes,
            'cpu': {
                'average': sum(cpu_values) / len(cpu_values),
                'min': min(cpu_values),
                'max': max(cpu_values)
            },
            'memory': {
                'average': sum(memory_values) / len(memory_values),
                'min': min(memory_values),
                'max': max(memory_values)
            },
            'disk': {
                'average': sum(disk_values) / len(disk_values),
                'min': min(disk_values),
                'max': max(disk_values)
            },
            'latest': asdict(recent_health[-1])
        }
    
    def export_metrics(self, file_path: str, format_type: str = "json"):
        """
        Export collected metrics to file.
        
        Args:
            file_path: Output file path
            format_type: Export format ("json" or "csv")
        """
        try:
            output_path = Path(file_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with self.lock:
                metrics_data = [asdict(metric) for metric in self.metrics]
                errors_data = [asdict(error) for error in self.error_events]
                health_data = [asdict(health) for health in self.system_health]
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'metrics': metrics_data,
                'errors': errors_data,
                'system_health': health_data,
                'operation_stats': self.get_all_operation_stats()
            }
            
            if format_type.lower() == "json":
                with open(output_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported export format: {format_type}")
            
            self.logger.info(f"Metrics exported to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {str(e)}")
            raise
    
    def add_metric_callback(self, callback: Callable[[PerformanceMetric], None]):
        """Add a callback for real-time metric notifications."""
        self.metric_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[ErrorEvent], None]):
        """Add a callback for real-time error notifications."""
        self.error_callbacks.append(callback)
    
    def add_health_callback(self, callback: Callable[[SystemHealth], None]):
        """Add a callback for real-time health notifications."""
        self.health_callbacks.append(callback)
    
    def clear_metrics(self):
        """Clear all collected metrics and statistics."""
        with self.lock:
            self.metrics.clear()
            self.error_events.clear()
            self.system_health.clear()
            self.operation_counts.clear()
            self.operation_durations.clear()
        
        self.logger.info("All metrics cleared")


class LoggingManager:
    """
    Comprehensive logging manager with structured logging,
    log rotation, and multiple output destinations.
    """
    
    def __init__(self, log_dir: str = "logs", app_name: str = "occupation_reports"):
        """
        Initialize the logging manager.
        
        Args:
            log_dir: Directory for log files
            app_name: Application name for log files
        """
        self.log_dir = Path(log_dir)
        self.app_name = app_name
        self.log_dir.mkdir(exist_ok=True)
        
        # Performance monitor integration
        self.performance_monitor: Optional[PerformanceMonitor] = None
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up comprehensive logging configuration."""
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)
        
        # Main application log file (rotating)
        main_log_file = self.log_dir / f"{self.app_name}.log"
        main_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        main_handler.setLevel(logging.DEBUG)
        main_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(main_handler)
        
        # Error log file (rotating)
        error_log_file = self.log_dir / f"{self.app_name}_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_handler)
        
        # Performance log file (time-based rotation)
        performance_log_file = self.log_dir / f"{self.app_name}_performance.log"
        performance_handler = logging.handlers.TimedRotatingFileHandler(
            performance_log_file,
            when='midnight',
            interval=1,
            backupCount=7
        )
        performance_handler.setLevel(logging.INFO)
        performance_handler.setFormatter(detailed_formatter)
        
        # Create performance logger
        performance_logger = logging.getLogger('performance')
        performance_logger.addHandler(performance_handler)
        performance_logger.setLevel(logging.INFO)
        performance_logger.propagate = False
        
        logging.info("Logging system initialized")
    
    def set_performance_monitor(self, monitor: PerformanceMonitor):
        """
        Set the performance monitor for integration.
        
        Args:
            monitor: PerformanceMonitor instance
        """
        self.performance_monitor = monitor
        
        # Add callbacks to log performance events
        monitor.add_error_callback(self._log_error_event)
        monitor.add_health_callback(self._log_health_event)
    
    def _log_error_event(self, error_event: ErrorEvent):
        """Log error events from performance monitor."""
        error_logger = logging.getLogger('errors')
        error_logger.error(
            f"Error in {error_event.module}.{error_event.function}:{error_event.line_number} - "
            f"{error_event.message} ({error_event.exception_type})"
        )
    
    def _log_health_event(self, health: SystemHealth):
        """Log system health events (only if concerning)."""
        performance_logger = logging.getLogger('performance')
        
        # Log if resource usage is high
        if health.cpu_percent > 80:
            performance_logger.warning(f"High CPU usage: {health.cpu_percent:.1f}%")
        
        if health.memory_percent > 85:
            performance_logger.warning(f"High memory usage: {health.memory_percent:.1f}%")
        
        if health.disk_usage_percent > 90:
            performance_logger.warning(f"High disk usage: {health.disk_usage_percent:.1f}%")
    
    def get_log_files(self) -> List[str]:
        """Get list of current log files."""
        return [str(f) for f in self.log_dir.glob("*.log")]
    
    def archive_logs(self, archive_path: Optional[str] = None) -> str:
        """
        Archive current log files.
        
        Args:
            archive_path: Optional custom archive path
            
        Returns:
            Path to created archive
        """
        import shutil
        
        if archive_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            archive_path = self.log_dir / f"logs_archive_{timestamp}"
        
        archive_path = Path(archive_path)
        archive_path.mkdir(parents=True, exist_ok=True)
        
        # Copy log files to archive
        for log_file in self.log_dir.glob("*.log*"):
            shutil.copy2(log_file, archive_path)
        
        logging.info(f"Logs archived to {archive_path}")
        return str(archive_path)


# Global instances for easy access
_performance_monitor: Optional[PerformanceMonitor] = None
_logging_manager: Optional[LoggingManager] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def get_logging_manager() -> LoggingManager:
    """Get the global logging manager instance."""
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = LoggingManager()
    return _logging_manager


def initialize_monitoring(log_dir: str = "logs", 
                         app_name: str = "occupation_reports",
                         start_monitoring: bool = True) -> Tuple[PerformanceMonitor, LoggingManager]:
    """
    Initialize the complete monitoring system.
    
    Args:
        log_dir: Directory for log files
        app_name: Application name
        start_monitoring: Whether to start system monitoring
        
    Returns:
        Tuple of (PerformanceMonitor, LoggingManager)
    """
    global _performance_monitor, _logging_manager
    
    # Initialize logging manager
    _logging_manager = LoggingManager(log_dir, app_name)
    
    # Initialize performance monitor
    _performance_monitor = PerformanceMonitor()
    
    # Integrate them
    _logging_manager.set_performance_monitor(_performance_monitor)
    
    # Start monitoring if requested
    if start_monitoring:
        _performance_monitor.start_monitoring()
    
    logging.info("Monitoring system initialized")
    
    return _performance_monitor, _logging_manager