# FAVOR Phase 0 — official diffusion_policy env, containerized.
#
# Design:
#   1. conda_environment.robodiff.yaml is a verbatim vendor of upstream's
#      conda_environment.yaml — nothing hand-picked here.
#   2. Base image matches cudatoolkit=11.6 in that file.
#   3. The diffusion_policy repo itself is NOT copied into the image —
#      it lives on the host (cloned in Step 1) and is bind-mounted at
#      runtime via docker-compose, so every host edit is reflected
#      immediately with no rebuild.
#   4. data/ and results/ are also host directories, bind-mounted —
#      checkpoints and outputs live outside the container.
#   5. Headless rendering: MUJOCO_GL=osmesa (matches upstream Ubuntu 20.04
#      instructions, no GPU/X needed for eval rendering).
#
# Build : docker compose -f docker/docker-compose.yml build
# Verify: docker compose -f docker/docker-compose.yml run favor verify
# Shell : docker compose -f docker/docker-compose.yml run favor

FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MUJOCO_GL=osmesa
ENV PATH=/opt/conda/bin:$PATH

# apt packages per upstream README (Ubuntu 20.04 mujoco_py requirements)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget curl ca-certificates \
    libosmesa6-dev libgl1-mesa-glx libglfw3 libglew-dev libegl1 patchelf \
    && rm -rf /var/lib/apt/lists/*

# Mambaforge (fast conda solver, as upstream README recommends)
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh

COPY docker/conda_environment.robodiff.yaml /tmp/conda_environment.robodiff.yaml
RUN conda env create -f /tmp/conda_environment.robodiff.yaml && conda clean -afy

SHELL ["conda", "run", "--no-capture-output", "-n", "robodiff", "/bin/bash", "-c"]

# Resolve-and-record actual installed versions at build time (source of truth
# for "what did we actually get", independent of the yaml's intent).
RUN python -c "import sys; print('python', sys.version)" > /opt/PINS.txt \
    && conda list >> /opt/PINS.txt \
    && python -m pip freeze > /opt/PINS.pip-freeze.txt

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY docker/verify_env.py /opt/verify_env.py
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]
