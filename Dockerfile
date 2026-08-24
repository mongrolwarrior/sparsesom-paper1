# Multi-stage build: compile all CUDA binaries then copy into a slim runtime image.
# Pin CUDA 12.8 — the host contributes only the NVIDIA driver.
# Build:  docker build -t sparsesom-paper1 .
# Run:    docker run --gpus all -v ./experiments:/workspace/experiments sparsesom-paper1 all

# ── Stage 1: build ──────────────────────────────────────────────
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ninja-build g++ git ca-certificates wget \
    && wget -qO /tmp/cmake.sh https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-linux-x86_64.sh \
    && sh /tmp/cmake.sh --skip-license --prefix=/usr/local \
    && rm /tmp/cmake.sh \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Copy submodule sources
COPY external/SparseBinarySOM external/SparseBinarySOM
COPY external/StandardSparseSOM external/StandardSparseSOM
COPY external/SparseBinEval external/SparseBinEval
COPY external/MedSOM external/MedSOM
COPY external/somoclu external/somoclu

# Build SparseBinarySOM (sparsesom)
RUN cd external/SparseBinarySOM \
    && cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="86;89;90" \
    && cmake --build build --parallel

# Build StandardSparseSOM (standardsparsesom)
RUN cd external/StandardSparseSOM \
    && cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="86;89;90" \
    && cmake --build build --parallel

# Build MedSOM (MedSOM_Naive)
RUN cd external/MedSOM \
    && cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="86;89;90" \
    && cmake --build build --parallel

# Build SparseBinEval metrics tool
RUN cd external/SparseBinEval \
    && g++ -O3 -fopenmp tools/metrics.cpp -o tools/metrics

# Build somoclu (CPU-only sparse SOM — autotools)
RUN apt-get update && apt-get install -y --no-install-recommends autoconf automake libtool \
    && rm -rf /var/lib/apt/lists/* \
    && cd external/somoclu \
    && ./autogen.sh \
    && ./configure --without-mpi --without-cuda \
    && for f in src/*.cpp; do sed -i '0,/^#include/{s/^#include/using namespace std;\n#include/}' "$f"; done \
    && make -j$(nproc)

# ── Stage 2: runtime ───────────────────────────────────────────
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Optional: Nsight Compute for experiment 10 (roofline). Adds ~1 GB, so it is
# off by default; build with --build-arg WITH_NCU=1 to include it. At run time
# the container also needs GPU performance-counter access (see README).
ARG WITH_NCU=0
RUN if [ "$WITH_NCU" = "1" ]; then \
        apt-get update && apt-get install -y --no-install-recommends \
            cuda-nsight-compute-12-8 \
        && rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /workspace

# Copy built binaries
COPY --from=builder /src/external/SparseBinarySOM/build/sparsesom /usr/local/bin/sparsesom
COPY --from=builder /src/external/StandardSparseSOM/build/standardsparsesom /usr/local/bin/standardsparsesom
COPY --from=builder /src/external/MedSOM/build/medsom /usr/local/bin/medsom
COPY --from=builder /src/external/SparseBinEval/tools/metrics /usr/local/bin/som-metrics
COPY --from=builder /src/external/somoclu/src/somoclu /usr/local/bin/somoclu

# Install pinned Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Cache-bust: on WSL2/Docker Desktop, COPY may not detect content changes after
# git pull.  The Makefile passes --build-arg CONFIGS_HASH=<sha256> so this layer
# invalidates whenever any config file changes.
ARG CONFIGS_HASH=0
COPY configs/ configs/
COPY pipeline/ pipeline/
COPY figures/ figures/
COPY scripts/ scripts/
COPY Makefile .
COPY repro .

ENV SPARSESOM=/usr/local/bin/sparsesom
ENV STANDARDSPARSESOM=/usr/local/bin/standardsparsesom
ENV MEDSOM=/usr/local/bin/medsom
ENV SOM_METRICS=/usr/local/bin/som-metrics
ENV SOMOCLU=/usr/local/bin/somoclu

ENTRYPOINT ["python3", "repro"]
