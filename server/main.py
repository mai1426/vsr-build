"""VSR HTTP 包装服务

把 VSR 的 CLI（backend/main.py）包成 HTTP 接口，供 PPIO Serverless Endpoint 调用。

接口：
  GET  /health          健康检查（PPIO 探活）
  POST /dedub           同步处理一条视频，返回结果下载地址
  GET  /files/{name}    取回处理结果（1 小时后自动清理）

请求示例：
  curl -X POST http://<endpoint>/dedub \
       -H "Content-Type: application/json" \
       -d '{"video_url":"https://.../in.mp4","mode":"lama",
            "regions":[{"ymin":1050,"ymax":1180,"xmin":0,"xmax":720}]}'
"""

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

VSR_DIR = Path(os.environ.get("VSR_DIR", "/vsr"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/tmp/vsr-out"))
JOB_TIMEOUT = int(os.environ.get("JOB_TIMEOUT", "1500"))
FILE_TTL = int(os.environ.get("FILE_TTL", "3600"))
ALLOWED_MODES = {"sttn-auto", "sttn-det", "lama", "propainter", "opencv"}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="vsr-server")

# 单卡只能同时跑一个任务；PPIO 侧并发上限配 1 时这里是冗余保护，
# 配错成 >1 时靠这把锁兜底，避免两个任务抢显存 OOM。
_gpu_lock = threading.Lock()


class Region(BaseModel):
    ymin: int = Field(ge=0)
    ymax: int = Field(ge=0)
    xmin: int = Field(ge=0)
    xmax: int = Field(ge=0)


class DedubRequest(BaseModel):
    video_url: str
    mode: str = "lama"
    # 不传则让 VSR 自动检测全片文字
    regions: Optional[List[Region]] = None


def _purge_expired() -> None:
    now = time.time()
    for p in OUTPUT_DIR.glob("*.mp4"):
        try:
            if now - p.stat().st_mtime > FILE_TTL:
                p.unlink()
        except OSError:
            pass


def _download(url: str, dst: Path) -> None:
    with requests.get(url, stream=True, timeout=(10, 120)) as resp:
        resp.raise_for_status()
        with open(dst, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)


def _run_vsr(src: Path, dst: Path, mode: str, regions: Optional[List[Region]]) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(VSR_DIR / "backend" / "main.py"),
        "-i", str(src),
        "-o", str(dst),
        "--inpaint-mode", mode,
    ]
    for r in regions or []:
        cmd += ["-c", str(r.ymin), str(r.ymax), str(r.xmin), str(r.xmax)]

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    return subprocess.run(
        cmd,
        cwd=str(VSR_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=JOB_TIMEOUT,
    )


@app.get("/health")
def health():
    return {"ok": True, "vsr_ready": (VSR_DIR / "backend" / "main.py").is_file()}


@app.post("/dedub")
def dedub(req: DedubRequest, request: Request):
    if req.mode not in ALLOWED_MODES:
        raise HTTPException(status_code=400, detail=f"不支持的 mode：{req.mode}")
    if req.regions:
        for r in req.regions:
            if r.ymax <= r.ymin or r.xmax <= r.xmin:
                raise HTTPException(status_code=400, detail="字幕区域坐标非法：需满足 ymax>ymin 且 xmax>xmin")

    _purge_expired()
    job_id = uuid.uuid4().hex[:12]
    work = Path(tempfile.mkdtemp(prefix=f"vsr-{job_id}-"))
    src = work / "input.mp4"
    dst = work / "output.mp4"

    try:
        try:
            _download(req.video_url, src)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"下载源视频失败：{exc}")

        started = time.time()
        with _gpu_lock:
            try:
                proc = _run_vsr(src, dst, req.mode, req.regions)
            except subprocess.TimeoutExpired:
                raise HTTPException(status_code=504, detail=f"处理超时（>{JOB_TIMEOUT}s）")

        if proc.returncode != 0 or not dst.is_file():
            tail = (proc.stderr or proc.stdout or "")[-1500:]
            raise HTTPException(status_code=500, detail=f"VSR 执行失败：{tail}")

        final = OUTPUT_DIR / f"{job_id}.mp4"
        shutil.move(str(dst), str(final))
        return {
            "job_id": job_id,
            "elapsed_sec": round(time.time() - started, 1),
            "output_url": str(request.base_url).rstrip("/") + f"/files/{final.name}",
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


@app.get("/files/{name}")
def get_file(name: str):
    if "/" in name or "\\" in name or not name.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="非法文件名")
    path = OUTPUT_DIR / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="文件不存在或已过期")
    return FileResponse(path, media_type="video/mp4", filename=name)
