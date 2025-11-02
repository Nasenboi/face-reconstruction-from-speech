FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04
WORKDIR /app

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    wget \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    pkg-config \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    cmake \
    curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV PYOPENGL_PLATFORM=egl

COPY 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    elif [ "$arch" = "aarch64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
    echo "Unsupported architecture: $arch"; \
    exit 1; \
    fi && \
    wget $MINICONDA_URL -O miniconda.sh && \
    mkdir -p /root/.conda && \
    bash miniconda.sh -b -p /root/miniconda3 && \
    rm -f miniconda.sh

ENV PATH="/root/miniconda3/bin:$PATH"
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && conda init

RUN git clone https://github.com/sicxu/Deep3DFaceRecon_pytorch.git
RUN cd Deep3DFaceRecon_pytorch && \ 
    git clone https://github.com/NVlabs/nvdiffrast.git
RUN cd Deep3DFaceRecon_pytorch && \ 
    git clone https://github.com/deepinsight/insightface.git && \
    cp -r ./insightface/recognition/arcface_torch ./models/

COPY environment.yml .

RUN conda env create -f environment.yml

RUN /root/miniconda3/envs/deep3d_pytorch/bin/python -m pip install /app/Deep3DFaceRecon_pytorch/nvdiffrast Ninja

COPY docker-entrypoint.sh .
ENTRYPOINT [ "/bin/bash", "docker-entrypoint.sh" ]
CMD [ "" ]

#CMD [ "/bin/bash" ]