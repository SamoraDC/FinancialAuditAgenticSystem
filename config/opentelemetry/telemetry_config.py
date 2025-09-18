"""
OpenTelemetry Configuration for Financial Audit System
"""

import logging
from typing import Dict, Any, Optional
from opentelemetry import trace, metrics
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
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.semantic_conventions.resource import ResourceAttributes

from backend.core.config import settings


class TelemetryManager:
    """OpenTelemetry configuration and management"""

    def __init__(self):
        self.tracer_provider = None
        self.meter_provider = None
        self.tracer = None
        self.meter = None
        self.logger = logging.getLogger(__name__)

    def setup_telemetry(self) -> None:
        """Initialize OpenTelemetry tracing and metrics"""

        # Create resource
        resource = Resource.create({
            ResourceAttributes.SERVICE_NAME: settings.OTEL_SERVICE_NAME,
            ResourceAttributes.SERVICE_VERSION: "1.0.0",
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: settings.ENVIRONMENT,
        })

        # Setup tracing
        self._setup_tracing(resource)

        # Setup metrics
        self._setup_metrics(resource)

        # Setup auto-instrumentation
        self._setup_auto_instrumentation()

        self.logger.info("OpenTelemetry setup completed")

    def _setup_tracing(self, resource: Resource) -> None:
        """Setup distributed tracing"""

        # Create tracer provider
        self.tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(self.tracer_provider)

        # Create OTLP span exporter
        otlp_exporter = OTLPSpanExporter(
            endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT,
            insecure=True  # Use False in production with proper certificates
        )

        # Create span processor
        span_processor = BatchSpanProcessor(otlp_exporter)
        self.tracer_provider.add_span_processor(span_processor)

        # Get tracer
        self.tracer = trace.get_tracer(__name__)

    def _setup_metrics(self, resource: Resource) -> None:
        """Setup metrics collection"""

        # Create OTLP metric exporter
        metric_exporter = OTLPMetricExporter(
            endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT,
            insecure=True  # Use False in production with proper certificates
        )

        # Create metric reader
        metric_reader = PeriodicExportingMetricReader(
            exporter=metric_exporter,
            export_interval_millis=30000  # Export every 30 seconds
        )

        # Create meter provider
        self.meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[metric_reader]
        )
        metrics.set_meter_provider(self.meter_provider)

        # Get meter
        self.meter = metrics.get_meter(__name__)

    def _setup_auto_instrumentation(self) -> None:
        """Setup automatic instrumentation for common libraries"""

        # Instrument FastAPI
        FastAPIInstrumentor.instrument()

        # Instrument Redis
        RedisInstrumentor.instrument()

        # Instrument HTTP requests
        RequestsInstrumentor.instrument()

        # Instrument SQLAlchemy (if used)
        SQLAlchemyInstrumentor.instrument()

    def create_custom_metrics(self) -> Dict[str, Any]:
        """Create custom metrics for audit operations"""

        if not self.meter:
            raise RuntimeError("Meter not initialized. Call setup_telemetry() first.")

        # Counters
        audit_counter = self.meter.create_counter(
            name="audit_operations_total",
            description="Total number of audit operations",
            unit="1"
        )

        fraud_detection_counter = self.meter.create_counter(
            name="fraud_detections_total",
            description="Total number of fraud detections",
            unit="1"
        )

        # Histograms
        audit_duration = self.meter.create_histogram(
            name="audit_duration_seconds",
            description="Duration of audit operations",
            unit="s"
        )

        risk_score_histogram = self.meter.create_histogram(
            name="risk_score_distribution",
            description="Distribution of risk scores",
            unit="1"
        )

        # Gauges (using observable gauge)
        active_audits_gauge = self.meter.create_observable_gauge(
            name="active_audits_count",
            description="Number of currently active audits",
            unit="1"
        )

        return {
            "audit_counter": audit_counter,
            "fraud_detection_counter": fraud_detection_counter,
            "audit_duration": audit_duration,
            "risk_score_histogram": risk_score_histogram,
            "active_audits_gauge": active_audits_gauge
        }

    def trace_audit_operation(self, operation_name: str, audit_id: str):
        """Context manager for tracing audit operations"""

        if not self.tracer:
            raise RuntimeError("Tracer not initialized. Call setup_telemetry() first.")

        return self.tracer.start_as_current_span(
            operation_name,
            attributes={
                "audit.id": audit_id,
                "audit.operation": operation_name,
                "service.name": settings.OTEL_SERVICE_NAME
            }
        )

    def add_audit_attributes(self, span, audit_data: Dict[str, Any]) -> None:
        """Add audit-specific attributes to a span"""

        if not span:
            return

        # Add common audit attributes
        span.set_attribute("audit.company_name", audit_data.get("company_name", ""))
        span.set_attribute("audit.scope", str(audit_data.get("audit_scope", [])))
        span.set_attribute("audit.risk_threshold", audit_data.get("risk_threshold", 0.0))

        # Add custom attributes if present
        if "findings_count" in audit_data:
            span.set_attribute("audit.findings.count", audit_data["findings_count"])

        if "overall_risk_score" in audit_data:
            span.set_attribute("audit.risk_score", audit_data["overall_risk_score"])

    def record_audit_metrics(self, metrics: Dict[str, Any], operation_type: str, audit_id: str) -> None:
        """Record audit operation metrics"""

        custom_metrics = self.create_custom_metrics()

        # Record audit operation
        custom_metrics["audit_counter"].add(
            1,
            attributes={
                "operation_type": operation_type,
                "audit_id": audit_id
            }
        )

        # Record duration if provided
        if "duration" in metrics:
            custom_metrics["audit_duration"].record(
                metrics["duration"],
                attributes={
                    "operation_type": operation_type,
                    "audit_id": audit_id
                }
            )

        # Record risk score if provided
        if "risk_score" in metrics:
            custom_metrics["risk_score_histogram"].record(
                metrics["risk_score"],
                attributes={
                    "operation_type": operation_type,
                    "audit_id": audit_id
                }
            )

        # Record fraud detections if provided
        if "fraud_detected" in metrics and metrics["fraud_detected"]:
            custom_metrics["fraud_detection_counter"].add(
                1,
                attributes={
                    "audit_id": audit_id,
                    "detection_type": metrics.get("detection_type", "unknown")
                }
            )


# Global telemetry manager instance
telemetry_manager = TelemetryManager()


def init_telemetry():
    """Initialize telemetry - call this at application startup"""
    telemetry_manager.setup_telemetry()
    return telemetry_manager


def get_tracer():
    """Get the configured tracer"""
    return telemetry_manager.tracer


def get_meter():
    """Get the configured meter"""
    return telemetry_manager.meter