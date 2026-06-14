# Customer churn analysis app — runnable locally with one command.
#
# On this Linux image (unlike a bare macOS host) the optional native deps are
# available, so inside the container you get the *full* experience:
#   * libgomp1            -> XGBoost loads natively (else the app falls back to
#                            sklearn HistGradientBoosting)
#   * pango/cairo libs    -> weasyprint PDF export works (else markdown/HTML)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# System libraries: OpenMP for XGBoost + Pango/Cairo stack for weasyprint PDFs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libpango-1.0-0 libpangocairo-1.0-0 libcairo2 \
        libgdk-pixbuf-2.0-0 libffi8 shared-mime-info fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install "weasyprint>=63"   # enabled here because system libs are present

# Note: plotly static image export (kaleido) needs a headless Chrome matching the
# image architecture, which isn't reliably available on arm64. The SHAP/driver
# visuals are therefore embedded in the in-app Report view and the HTML export
# (interactive); the PDF export carries the full narrative + data tables.

# App source.
COPY . .

# Pre-generate the bundled synthetic demo dataset so the in-app demo button works.
RUN python -m sample_data.make_synthetic

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
