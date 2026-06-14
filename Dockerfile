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

# System libraries:
#  - libgomp1                      -> XGBoost (OpenMP)
#  - pango/cairo/gdk-pixbuf/...    -> weasyprint PDF rendering
#  - libnss3/libatk/libxkbcommon/… -> headless Chrome that kaleido uses to render
#                                     plotly figures to PNG for the PDF report
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libpango-1.0-0 libpangocairo-1.0-0 libcairo2 \
        libgdk-pixbuf-2.0-0 libffi8 shared-mime-info fonts-dejavu-core \
        libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
        libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
        libgbm1 libasound2 libxshmfence1 libatspi2.0-0 libx11-6 libxcb1 \
        libxext6 libxi6 fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install "weasyprint>=63"   # enabled here because system libs are present

# Fetch the headless Chrome that kaleido uses to render plotly figures to PNG,
# so the SHAP/driver visuals embed in the exported PDF report. Best-effort: the
# app degrades to interactive-HTML figures if this is unavailable.
RUN kaleido_get_chrome || echo "kaleido chrome fetch skipped (PDF figures degrade to HTML)"

# App source.
COPY . .

# Pre-generate the bundled synthetic demo dataset so the in-app demo button works.
RUN python -m sample_data.make_synthetic

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
