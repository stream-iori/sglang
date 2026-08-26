FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/python:3.12-slim-bookworm

WORKDIR /workspace

RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir torch==2.13.0+cpu \
       --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir triton==3.7.1 numpy==2.5.2 uv

# Must be set before Python imports Triton. Kernels execute op-by-op on CPU.
ENV TRITON_INTERPRET=1 \
    PYTHONUNBUFFERED=1

COPY examples/triton /workspace/examples/triton
COPY docker/run-triton-lessons.sh /workspace/docker/run-triton-lessons.sh

RUN python -c "import torch, triton; assert not torch.cuda.is_available(); print('torch', torch.__version__, 'triton', triton.__version__)"

CMD ["bash", "/workspace/docker/run-triton-lessons.sh"]
