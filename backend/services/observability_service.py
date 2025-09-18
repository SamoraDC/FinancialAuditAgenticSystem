"""
OpenTelemetry + Logfire Observability Service
Comprehensive monitoring and observability for financial audit system
"""

import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Callable
from functools import wraps
from datetime import datetime
import json
import traceback
from contextvars import ContextVar

# OpenTelemetry imports
from opentelemetry import trace, metrics, baggage
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# Logfire imports (placeholder - would use actual logfire SDK)
try:
    import logfire
    HAS_LOGFIRE = True
except ImportError:
    HAS_LOGFIRE = False
    logging.warning("Logfire not available. Using standard logging.")

from ..core.config import settings

logger = logging.getLogger(__name__)

# Context variables for tracing
trace_context: ContextVar[Dict[str, Any]] = ContextVar('trace_context', default={})
audit_session_context: ContextVar[str] = ContextVar('audit_session_context', default='')


class ObservabilityService:
    """
    Comprehensive observability service with OpenTelemetry and Logfire integration
    """

    def __init__(self):
        self.resource = Resource.create({
            "service.name": "financial-audit-system",
            "service.version": "1.0.0",
            "environment": settings.ENVIRONMENT
        })

        # Initialize OpenTelemetry
        self._setup_tracing()
        self._setup_metrics()
        self._setup_logging()

        # Initialize Logfire if available
        if HAS_LOGFIRE:
            self._setup_logfire()

        # Instrumentation
        self._setup_auto_instrumentation()

        # Custom metrics
        self.tracer = trace.get_tracer(__name__)
        self.meter = metrics.get_meter(__name__)

        # Business metrics
        self.audit_counter = self.meter.create_counter(
            "audit_sessions_total",
            description="Total number of audit sessions",
            unit="1"
        )

        self.document_processing_histogram = self.meter.create_histogram(
            "document_processing_duration_seconds",
            description="Time spent processing documents",
            unit="s"
        )

        self.anomaly_detection_counter = self.meter.create_counter(
            "anomalies_detected_total",
            description="Total number of anomalies detected",
            unit="1"
        )

        self.llm_request_histogram = self.meter.create_histogram(
            "llm_request_duration_seconds",
            description="LLM request duration",
            unit="s"
        )

        self.security_violation_counter = self.meter.create_counter(
            "security_violations_total",
            description="Total security violations detected",
            unit="1"
        )

    def _setup_tracing(self):
        """Setup OpenTelemetry tracing"""
        try:
            # Create tracer provider
            trace.set_tracer_provider(TracerProvider(resource=self.resource))

            # Configure OTLP exporter
            otlp_exporter = OTLPSpanExporter(
                endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT,
                insecure=True  # Use secure=True in production with proper certificates
            )

            # Add span processor
            span_processor = BatchSpanProcessor(otlp_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)

            logger.info("OpenTelemetry tracing initialized successfully")

        except Exception as e:
            logger.error(f"Failed to setup tracing: {str(e)}")

    def _setup_metrics(self):
        """Setup OpenTelemetry metrics"""
        try:
            # Configure metric exporter
            metric_exporter = OTLPMetricExporter(
                endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT,
                insecure=True
            )

            # Create metric reader
            metric_reader = PeriodicExportingMetricReader(
                exporter=metric_exporter,
                export_interval_millis=5000  # Export every 5 seconds
            )

            # Set meter provider
            metrics.set_meter_provider(MeterProvider(
                resource=self.resource,
                metric_readers=[metric_reader]
            ))

            logger.info("OpenTelemetry metrics initialized successfully")

        except Exception as e:
            logger.error(f"Failed to setup metrics: {str(e)}")

    def _setup_logging(self):
        """Setup structured logging with OpenTelemetry context"""
        try:
            # Configure structured logging format
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - '
                'trace_id=%(otelTraceID)s span_id=%(otelSpanID)s - %(message)s'
            )

            # Get root logger
            root_logger = logging.getLogger()
            for handler in root_logger.handlers:
                handler.setFormatter(formatter)

            logger.info("Structured logging with OpenTelemetry context initialized")

        except Exception as e:
            logger.error(f"Failed to setup logging: {str(e)}")

    def _setup_logfire(self):
        """Setup Logfire integration"""
        try:
            if HAS_LOGFIRE:
                logfire.configure(
                    service_name="financial-audit-system",
                    environment=settings.ENVIRONMENT,
                    # Additional Logfire configuration
                )
                logger.info("Logfire integration initialized successfully")

        except Exception as e:
            logger.error(f"Failed to setup Logfire: {str(e)}")

    def _setup_auto_instrumentation(self):
        """Setup automatic instrumentation for common libraries"""
        try:
            # FastAPI instrumentation
            FastAPIInstrumentor.instrument()

            # Redis instrumentation
            RedisInstrumentor.instrument()

            # Requests instrumentation
            RequestsInstrumentor.instrument()

            logger.info("Auto-instrumentation setup completed")

        except Exception as e:
            logger.error(f"Failed to setup auto-instrumentation: {str(e)}")

    def trace_audit_workflow(self, workflow_name: str):
        """Decorator for tracing audit workflow steps"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(
                    f"audit_workflow.{workflow_name}",
                    attributes={
                        "workflow.name": workflow_name,
                        "workflow.step": func.__name__
                    }
                ) as span:
                    try:
                        # Add audit session context if available
                        session_id = audit_session_context.get('')
                        if session_id:
                            span.set_attribute("audit.session_id", session_id)

                        start_time = time.time()
                        result = await func(*args, **kwargs)
                        duration = time.time() - start_time

                        # Record success metrics
                        span.set_attribute("workflow.duration_seconds", duration)
                        span.set_attribute("workflow.status", "success")

                        return result

                    except Exception as e:
                        # Record error details
                        span.record_exception(e)
                        span.set_attribute("workflow.status", "error")
                        span.set_attribute("error.type", type(e).__name__)
                        span.set_attribute("error.message", str(e))

                        # Log to Logfire if available
                        if HAS_LOGFIRE:
                            logfire.error(
                                f"Workflow step {workflow_name}.{func.__name__} failed",
                                error=e,
                                workflow=workflow_name,
                                step=func.__name__,
                                session_id=session_id
                            )

                        raise

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(
                    f"audit_workflow.{workflow_name}",
                    attributes={
                        "workflow.name": workflow_name,
                        "workflow.step": func.__name__
                    }
                ) as span:
                    try:
                        session_id = audit_session_context.get('')
                        if session_id:
                            span.set_attribute("audit.session_id", session_id)

                        start_time = time.time()
                        result = func(*args, **kwargs)
                        duration = time.time() - start_time

                        span.set_attribute("workflow.duration_seconds", duration)
                        span.set_attribute("workflow.status", "success")

                        return result

                    except Exception as e:
                        span.record_exception(e)
                        span.set_attribute("workflow.status", "error")
                        span.set_attribute("error.type", type(e).__name__)
                        span.set_attribute("error.message", str(e))

                        if HAS_LOGFIRE:
                            logfire.error(
                                f"Workflow step {workflow_name}.{func.__name__} failed",
                                error=e,
                                workflow=workflow_name,
                                step=func.__name__,
                                session_id=session_id
                            )

                        raise

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator

    def trace_document_processing(self):
        """Decorator for tracing document processing operations"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(
                    "document_processing",
                    attributes={"operation": func.__name__}
                ) as span:
                    try:
                        start_time = time.time()
                        result = await func(*args, **kwargs)
                        duration = time.time() - start_time

                        # Record metrics
                        self.document_processing_histogram.record(duration, {
                            "operation": func.__name__,
                            "status": "success"
                        })

                        # Add document-specific attributes
                        if hasattr(result, 'confidence_score'):
                            span.set_attribute("document.confidence_score", result.confidence_score)

                        if hasattr(result, 'text') and result.text:
                            span.set_attribute("document.text_length", len(result.text))

                        return result

                    except Exception as e:
                        # Record error metrics
                        self.document_processing_histogram.record(
                            time.time() - start_time,
                            {"operation": func.__name__, "status": "error"}
                        )

                        span.record_exception(e)
                        raise

            return wrapper

        return decorator

    def trace_statistical_analysis(self):
        """Decorator for tracing statistical analysis operations"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(
                    "statistical_analysis",
                    attributes={"analysis_type": func.__name__}
                ) as span:
                    try:
                        start_time = time.time()
                        result = func(*args, **kwargs)
                        duration = time.time() - start_time

                        # Add analysis-specific attributes
                        if hasattr(result, 'anomaly_detected'):
                            span.set_attribute("analysis.anomaly_detected", result.anomaly_detected)

                            if result.anomaly_detected:
                                self.anomaly_detection_counter.add(1, {
                                    "analysis_type": func.__name__,
                                    "severity": getattr(result, 'severity', 'unknown')
                                })

                        if hasattr(result, 'p_value'):
                            span.set_attribute("analysis.p_value", result.p_value)

                        if hasattr(result, 'sample_size'):
                            span.set_attribute("analysis.sample_size", result.sample_size)

                        span.set_attribute("analysis.duration_seconds", duration)

                        return result

                    except Exception as e:
                        span.record_exception(e)
                        raise

            return wrapper

        return decorator

    def trace_llm_interaction(self):
        """Decorator for tracing LLM interactions"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(
                    "llm_interaction",
                    attributes={"operation": func.__name__}
                ) as span:
                    try:
                        start_time = time.time()
                        result = await func(*args, **kwargs)
                        duration = time.time() - start_time

                        # Record LLM metrics
                        self.llm_request_histogram.record(duration, {
                            "model": getattr(result, 'model', 'unknown'),
                            "operation": func.__name__,
                            "status": "success"
                        })

                        # Add LLM-specific attributes
                        if hasattr(result, 'tokens_used'):
                            span.set_attribute("llm.tokens_used", result.tokens_used)

                        if hasattr(result, 'model'):
                            span.set_attribute("llm.model", result.model)

                        if hasattr(result, 'safety_check'):
                            safety = result.safety_check
                            if safety:
                                span.set_attribute("llm.safety_check.passed", safety.get('output_safe', True))

                        return result

                    except Exception as e:
                        # Record error metrics
                        self.llm_request_histogram.record(
                            time.time() - start_time,
                            {"operation": func.__name__, "status": "error"}
                        )

                        span.record_exception(e)
                        raise

            return wrapper

        return decorator

    def trace_security_check(self):
        """Decorator for tracing security checks"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(
                    "security_check",
                    attributes={"operation": func.__name__}
                ) as span:
                    try:
                        result = await func(*args, **kwargs)

                        # Record security metrics
                        if hasattr(result, 'violations'):
                            violation_count = len(result.violations)
                            span.set_attribute("security.violation_count", violation_count)

                            for violation in result.violations:
                                self.security_violation_counter.add(1, {
                                    "violation_type": violation.violation_type,
                                    "severity": violation.severity
                                })

                        if hasattr(result, 'risk_score'):
                            span.set_attribute("security.risk_score", result.risk_score)

                        if hasattr(result, 'is_safe'):
                            span.set_attribute("security.is_safe", result.is_safe)

                        return result

                    except Exception as e:
                        span.record_exception(e)
                        raise

            return wrapper

        return decorator

    def log_audit_event(self, event_type: str, details: Dict[str, Any],
                       session_id: Optional[str] = None, severity: str = "info"):
        """Log structured audit events"""
        try:
            # Create structured log entry
            log_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "event_type": event_type,
                "session_id": session_id or audit_session_context.get(''),
                "details": details,
                "service": "financial-audit-system",
                "environment": settings.ENVIRONMENT
            }

            # Add trace context if available
            current_span = trace.get_current_span()
            if current_span.is_recording():
                span_context = current_span.get_span_context()
                log_data["trace_id"] = format(span_context.trace_id, "032x")
                log_data["span_id"] = format(span_context.span_id, "016x")

            # Log with appropriate level
            log_message = f"Audit Event: {event_type}"

            if severity == "error":
                logger.error(log_message, extra=log_data)
            elif severity == "warning":
                logger.warning(log_message, extra=log_data)
            else:
                logger.info(log_message, extra=log_data)

            # Log to Logfire if available
            if HAS_LOGFIRE:
                if severity == "error":
                    logfire.error(log_message, **log_data)
                elif severity == "warning":
                    logfire.warn(log_message, **log_data)
                else:
                    logfire.info(log_message, **log_data)

        except Exception as e:
            logger.error(f"Failed to log audit event: {str(e)}")

    def record_audit_session_start(self, session_id: str, client_name: str, auditor_id: str):
        """Record audit session start event"""
        audit_session_context.set(session_id)

        self.audit_counter.add(1, {
            "event": "session_start",
            "client": client_name
        })

        self.log_audit_event(
            "audit_session_started",
            {
                "session_id": session_id,
                "client_name": client_name,
                "auditor_id": auditor_id
            },
            session_id=session_id
        )

    def record_audit_session_complete(self, session_id: str, findings_count: int,
                                    anomalies_count: int, duration_seconds: float):
        """Record audit session completion"""
        self.audit_counter.add(1, {
            "event": "session_complete",
            "findings_count": str(findings_count),
            "anomalies_count": str(anomalies_count)
        })

        self.log_audit_event(
            "audit_session_completed",
            {
                "session_id": session_id,
                "findings_count": findings_count,
                "anomalies_count": anomalies_count,
                "duration_seconds": duration_seconds
            },
            session_id=session_id
        )

    def record_anomaly_detection(self, session_id: str, anomaly_type: str,
                                severity: str, confidence: float):
        """Record anomaly detection event"""
        self.anomaly_detection_counter.add(1, {
            "type": anomaly_type,
            "severity": severity
        })

        self.log_audit_event(
            "anomaly_detected",
            {
                "session_id": session_id,
                "anomaly_type": anomaly_type,
                "severity": severity,
                "confidence": confidence
            },
            session_id=session_id,
            severity="warning" if severity in ["high", "critical"] else "info"
        )

    def record_security_incident(self, incident_type: str, severity: str,
                                details: Dict[str, Any], session_id: Optional[str] = None):
        """Record security incident"""
        self.security_violation_counter.add(1, {
            "incident_type": incident_type,
            "severity": severity
        })

        self.log_audit_event(
            "security_incident",
            {
                "incident_type": incident_type,
                "severity": severity,
                **details
            },
            session_id=session_id,
            severity="error" if severity == "critical" else "warning"
        )

    def create_dashboard_metrics(self) -> Dict[str, Any]:
        """Create real-time dashboard metrics"""
        current_span = trace.get_current_span()

        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "service_status": "healthy",
            "active_sessions": audit_session_context.get('session_count', 0),
            "trace_id": format(current_span.get_span_context().trace_id, "032x") if current_span.is_recording() else None
        }

        return metrics

    def setup_health_checks(self):
        """Setup health check monitoring"""
        try:
            # Create health check metrics
            health_gauge = self.meter.create_gauge(
                "service_health_status",
                description="Service health status (1=healthy, 0=unhealthy)",
                unit="1"
            )

            # Record initial healthy status
            health_gauge.set(1)

            logger.info("Health check monitoring setup completed")

        except Exception as e:
            logger.error(f"Failed to setup health checks: {str(e)}")


# Global observability service instance
observability = ObservabilityService()

# Export decorators for easy use
trace_audit_workflow = observability.trace_audit_workflow
trace_document_processing = observability.trace_document_processing
trace_statistical_analysis = observability.trace_statistical_analysis
trace_llm_interaction = observability.trace_llm_interaction
trace_security_check = observability.trace_security_check