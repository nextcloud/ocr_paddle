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

# Download and install FRP client into /usr/local/bin.
RUN set -ex; \
    ARCH=$(uname -m); \
    if [ "$ARCH" = "aarch64" ]; then \
      FRP_URL="https://raw.githubusercontent.com/nextcloud/HaRP/main/exapps_dev/frp_0.61.1_linux_arm64.tar.gz"; \
    else \
      FRP_URL="https://raw.githubusercontent.com/nextcloud/HaRP/main/exapps_dev/frp_0.61.1_linux_amd64.tar.gz"; \
    fi; \
    echo "Downloading FRP client from $FRP_URL"; \
    curl -L "$FRP_URL" -o /tmp/frp.tar.gz; \
    tar -C /tmp -xzf /tmp/frp.tar.gz; \
    mv /tmp/frp_0.61.1_linux_* /tmp/frp; \
    cp /tmp/frp/frpc /usr/local/bin/frpc; \
    chmod +x /usr/local/bin/frpc; \
    rm -rf /tmp/frp /tmp/frp.tar.gz

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
