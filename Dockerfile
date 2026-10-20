# SPDX-FileCopyrightText: 2025 Nextcloud GmbH and Nextcloud contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
# main.py imports `ex_app.lib.ocs`, so the repo root must be importable
# even though the entrypoint runs from WORKDIR /ex_app/lib.
ENV PYTHONPATH=/

COPY requirements.txt /

RUN \
   apt-get update -y && \
   apt-get install -y software-properties-common && \
   add-apt-repository -y ppa:deadsnakes/ppa && \
   apt-get update -y && \
   apt-get install -y --no-install-recommends \
     python3.11 python3.11-venv python3.11-dev \
     git curl ca-certificates pciutils \
     poppler-utils libgl1 libglib2.0-0 && \
   update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
   rm -rf /var/lib/apt/lists/*

# Download and install FRP client with checksum verification
# FRP version and checksums - update these when upgrading
ARG FRP_VERSION=0.71.1
ARG FRP_AMD64_SHA256=1d9875210972a5d1a62b2dc4c48df45ab0af05d3ca1b5e494a014428259df803
ARG FRP_ARM64_SHA256=8edf590b0c396fa42f9c147121a76af83cfc2002b08f5aef0ea1008c43482019

RUN set -ex; \
    ARCH=$(uname -m); \
    if [ "$ARCH" = "aarch64" ]; then \
        FRP_ARCH="arm64"; \
        FRP_SHA256="${FRP_ARM64_SHA256}"; \
    else \
        FRP_ARCH="amd64"; \
        FRP_SHA256="${FRP_AMD64_SHA256}"; \
    fi; \
    FRP_URL="https://github.com/fatedier/frp/releases/download/v${FRP_VERSION}/frp_${FRP_VERSION}_linux_${FRP_ARCH}.tar.gz"; \
    echo "Downloading FRP v${FRP_VERSION} for ${FRP_ARCH}..."; \
    curl -fsSL "${FRP_URL}" -o /tmp/frp.tar.gz; \
    ACTUAL_SHA256=$(sha256sum /tmp/frp.tar.gz | cut -d' ' -f1); \
    if [ "$ACTUAL_SHA256" != "$FRP_SHA256" ]; then \
        echo "Checksum verification failed for FRP v${FRP_VERSION} (${FRP_ARCH})"; \
        echo "Expected: ${FRP_SHA256}"; \
        echo "Got:      ${ACTUAL_SHA256}"; \
        exit 1; \
    fi; \
    tar -C /tmp -xzf /tmp/frp.tar.gz; \
    cp /tmp/frp_${FRP_VERSION}_linux_${FRP_ARCH}/frpc /usr/local/bin/frpc; \
    chmod +x /usr/local/bin/frpc; \
    rm -rf /tmp/frp_${FRP_VERSION}_linux_${FRP_ARCH} /tmp/frp.tar.gz; \
    echo "FRP client installed successfully"

# Bootstrap pip for python3.11 explicitly so it does not resolve to the
# distro pip that ships with the system python.
RUN \
  python3 -m ensurepip --upgrade && \
  python3 -m pip install --upgrade pip && \
  python3 -m pip install -r requirements.txt && \
  rm -rf ~/.cache requirements.txt

ADD /ex_app/cs[s] /ex_app/css
ADD /ex_app/im[g] /ex_app/img
ADD /ex_app/j[s] /ex_app/js
ADD /ex_app/l10[n] /ex_app/l10n
ADD /ex_app/li[b] /ex_app/lib

COPY --chmod=775 healthcheck.sh /
COPY --chmod=775 start.sh /

WORKDIR /ex_app/lib
ENTRYPOINT ["/start.sh", "python3", "main.py"]

LABEL org.opencontainers.image.source=https://github.com/nextcloud/ocr_paddle
HEALTHCHECK --interval=2s --timeout=2s --retries=300 CMD /healthcheck.sh
