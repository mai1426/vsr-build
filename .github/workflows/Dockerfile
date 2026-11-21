# PPIO Serverless Endpoint 用的 VSR 镜像
#
# 思路：不自建环境，直接复用官方已构建好的 VSR 镜像（Python 3.12 + PaddleGPU 3.0.0 +
# Torch 2.7.0 都已装好、CUDNN/CUDA 版本也匹配），只在上面叠一层很薄的 HTTP 包装。
# 这样镜像增量只有几十 MB，构建 1~2 分钟，PPIO 拉镜像的费用也很低。
#
# 构建（在能访问 Docker Hub 的机器上）：
#   docker build -t <仓库地址>:1.4.0-ppio .
#   例：docker build -t registry.cn-hangzhou.aliyuncs.com/<命名空间>/vsr:1.4.0-ppio .
#
# 40 系显卡用 cuda12.6；30 系也可用 cuda12.6；如遇驱动过老改 cuda11.8。
FROM eritpchy/video-subtitle-remover:1.4.0-cuda12.6

ENV PYTHONUNBUFFERED=1 \
    VSR_DIR=/vsr \
    OUTPUT_DIR=/tmp/vsr-out \
    JOB_TIMEOUT=1500 \
    FILE_TTL=3600

COPY server /opt/server

# 官方镜像的 python 里应该自带 pip；若为 conda 环境同样是 python 指到该环境
RUN python -m pip install --no-cache-dir -r /opt/server/requirements.txt

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--app-dir", "/opt/server", "--host", "0.0.0.0", "--port", "8000"]
