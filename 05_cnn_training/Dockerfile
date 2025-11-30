FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04
WORKDIR /app

# include but make it not interactive
RUN apt-get update && \
    apt-get install -y wget ffmpeg git libsndfile1-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda on x86 or ARM platforms
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

ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_XLA_FLAGS="--tf_xla_auto_jit=2"
ENV XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"

COPY environment.yml .

RUN conda env create -f environment.yml

RUN /root/miniconda3/envs/cnn-training/bin/python -m ipykernel install --user --name=cnn-training

CMD [ "/root/miniconda3/envs/cnn-training/bin/jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--notebook-dir=/app/notebooks" ]

#CMD ["/bin/bash"]