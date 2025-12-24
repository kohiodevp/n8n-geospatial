# ============================================
# n8n avec QGIS (base officielle QGIS)
# ============================================
FROM qgis/qgis:latest

USER root

# ============================================
# Variables d'environnement
# ============================================
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    N8N_RUNNERS_MODE=external \
    QT_QPA_PLATFORM=offscreen \
    XDG_RUNTIME_DIR=/tmp/runtime-node \
    GDAL_CACHEMAX=1024 \
    GDAL_NUM_THREADS=ALL_CPUS \
    PROJ_NETWORK=ON

# ============================================
# Dépendances système + Node.js
# ============================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    gnupg \
    bash \
    git \
    jq \
    zip \
    unzip \
    postgresql-client \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Installer Node.js LTS (requis par n8n runner)
# ============================================
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get update && apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# ============================================
# Installer n8n (runner)
# ============================================
RUN npm install -g n8n@2.1.1

# ============================================
# Python packages
# ============================================
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install -r /tmp/requirements.txt && \
    rm -rf /tmp/* ~/.cache/pip

# Make sure scripts use the virtual environment Python
ENV PATH="/opt/venv/bin:$PATH"

# ============================================
# Scripts géospatiaux
# ============================================
RUN mkdir -p /opt/geoscripts /tmp/runtime-node && chmod 777 /tmp/runtime-node

COPY scripts/qgis_processing.py /opt/geoscripts/
COPY scripts/postgis_utils.py /opt/geoscripts/
COPY scripts/grass_utils.py /opt/geoscripts/
COPY scripts/health_check.py /opt/geoscripts/

RUN chmod +x /opt/geoscripts/*.py
ENV PYTHONPATH="/opt/geoscripts:${PYTHONPATH}"

# ============================================
# Dossiers de travail
# ============================================
RUN mkdir -p /files /geodata /qgis-output /tmp/geodata-cache && \
    chmod 777 /files /geodata /qgis-output /tmp/geodata-cache

# ============================================
# Script de démarrage
# ============================================
COPY startup.sh /startup.sh
RUN chmod +x /startup.sh

EXPOSE 5678

CMD ["/startup.sh"]
