"""
Logging configuration with OpenTelemetry integration
"""

import logging
import sys
import os

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

from backend.core.config import settings


def setup_logging():
    """Setup logging and OpenTelemetry tracing"""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if not settings.DEBUG else logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout
    )

    # Setup OpenTelemetry tracing if available
    if OTEL_AVAILABLE and not os.getenv('TEST_MODE', '').lower() == 'true':
        trace.set_tracer_provider(TracerProvider())
        tracer = trace.get_tracer(__name__)

        # Setup OTLP exporter
        otlp_exporter = OTLPSpanExporter(
            endpoint=settings.OTEL_EXPORTER_OTLP_ENDPOINT,
            insecure=True
        )

        span_processor = BatchSpanProcessor(otlp_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)

        return tracer
    else:
        # Return None when OpenTelemetry is not available or in test mode
        return None


def get_logger(name: str = __name__) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name)