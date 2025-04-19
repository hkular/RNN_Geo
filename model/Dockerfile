
# Base image from tf
FROM tensorflow/tensorflow:1.10.0-gpu-py3

LABEL maintainer="Holly Kular <hkular@ucsd.edu>"
LABEL description="Reproduction of 2022 Probrnn environment with TensorFlow 1.10.0"

# Install essential libraries

RUN apt-get update && apt-get install -y --no-install-recommends \
	wget \
	curl \
	bzip2 \
	ca-certificates \
	libglib2.0-0 \
	libxext6 \
	libsm6 \
	libxrender1 \
	git \
	&& rm -rf /ver/lib/apt/lists/*


# Install anaconda
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda/3-5.2.0-Linux-x86_64.sh -O ~/anaconda3.sh && \
	/bin/bash ~/anaconda3.sh -b -p /opt/conda && \
	rm ~/anaconda3.sh

# Add conda to PATH
ENV PATH="/opt/conda/bin:${PATH}"

# Project code copy for shipping
COPY . /workspace

# Set working directory
WORKDIR /workspace

# Default command
CMD [ "python" ]

