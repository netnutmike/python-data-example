# Occupation Data Reports - Docker Configuration
# Multi-stage build for optimized production image

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION=1.0.0
ARG VCS_REF

# Add metadata labels
LABEL maintainer="Occupation Data Reports Team <team@occupation-reports.com>" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="occupation-data-reports" \
      org.label-schema.description="Comprehensive analysis of occupational requirements survey data" \
      org.label-schema.url="https://github.com/your-org/occupation-data-reports" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/your-org/occupation-data-reports" \
      org.label-schema.vendor="Occupation Data Reports Team" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual environment and install Python dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Set build arguments for production stage
ARG BUILD_DATE
ARG VERSION=1.0.0
ARG VCS_REF

# Add metadata labels
LABEL maintainer="Occupation Data Reports Team <team@occupation-reports.com>" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.name="occupation-data-reports" \
      org.label-schema.description="Comprehensive analysis of occupational requirements survey data" \
      org.label-schema.url="https://github.com/your-org/occupation-data-reports" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/your-org/occupation-data-reports" \
      org.label-schema.vendor="Occupation Data Reports Team" \
      org.label-schema.version=$VERSION \
      org.label-schema.schema-version="1.0"

# Install system dependencies for runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    # For WeasyPrint PDF generation
    libpango-1.0-0 \
    libharfbuzz0b \
    libpangoft2-1.0-0 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    shared-mime-info \
    # For system monitoring
    procps \
    # For file operations
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY . .

# Install the application
RUN pip install --no-cache-dir -e .

# Create necessary directories
RUN mkdir -p /app/data /app/reports /app/logs /app/config /app/temp && \
    chown -R appuser:appuser /app

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    ORS_CONFIG_DIR=/app/config \
    ORS_DATA_DIR=/app/data \
    ORS_OUTPUT_DIR=/app/reports \
    ORS_LOG_DIR=/app/logs

# Switch to non-root user
USER appuser

# Create default configuration
RUN python -m src.main create-config --config-dir /app/config

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -m src.main status || exit 1

# Expose port for potential web interface (future feature)
EXPOSE 8080

# Default command
CMD ["python", "-m", "src.main", "--help"]

# Volume mounts for data persistence
VOLUME ["/app/data", "/app/reports", "/app/logs", "/app/config"]

# Development stage (optional)
FROM production as development

# Switch back to root for development tools installation
USER root

# Install development dependencies
RUN pip install --no-cache-dir -e ".[dev,docs,all]"

# Install additional development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    less \
    && rm -rf /var/lib/apt/lists/*

# Switch back to appuser
USER appuser

# Override default command for development
CMD ["bash"]