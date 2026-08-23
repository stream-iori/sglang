# Build for linux/amd64 on Apple Silicon; see docs/triton-docker-apple-silicon.md.
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/python:3.12-slim-bookworm

WORKDIR /workspace

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir torch==2.13.0+cpu \
       --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir triton==3.7.1 numpy==2.5.2 uv

# The Triton interpreter executes a real @triton.jit kernel on CPU.  It is for
# learning and correctness testing, not performance measurement.
ENV TRITON_INTERPRET=1

# The source tree is mounted at /workspace when the container starts.
CMD ["bash"]
