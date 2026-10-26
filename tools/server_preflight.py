"""Record server topology and test real NCCL collectives without training."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import torch
import torch.distributed as dist


def command(arguments):
    try:
        result = subprocess.run(arguments, text=True, capture_output=True, timeout=30)
        return dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(error=str(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-gpus", type=int, default=7)
    args = parser.parse_args()
    visible = torch.cuda.device_count()
    if visible != args.expect_gpus:
        raise RuntimeError(f"Expected {args.expect_gpus} visible GPUs, found {visible}")
    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    device = torch.device("cuda", local)
    if rank == 0:
        if args.output.exists():
            raise FileExistsError(args.output)
        report = dict(host=platform.node(), python=platform.python_version(), torch=torch.__version__,
                      cuda=torch.version.cuda, visible_gpus=visible, world_size=world,
                      devices=[dict(name=torch.cuda.get_device_name(i),
                                    total_gib=torch.cuda.get_device_properties(i).total_memory / 2**30)
                               for i in range(visible)],
                      peer_access=[[bool(torch.cuda.can_device_access_peer(i, j)) if i != j else True
                                    for j in range(visible)] for i in range(visible)],
                      topology=command(["nvidia-smi", "topo", "-m"]),
                      gpu_status=command(["nvidia-smi"]), cpu=command(["lscpu"]),
                      memory=command(["free", "-h"]), storage=command(["df", "-h", "."]))
    measurements = []
    if world > 1:
        for size in (1 << 20, 32 << 20, 128 << 20):
            tensor = torch.ones(size // 4, device=device)
            dist.all_reduce(tensor)
            if not torch.all(tensor == world):
                raise AssertionError("NCCL all-reduce produced incorrect values")
            for _ in range(5):
                dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
            torch.cuda.synchronize()
            tick = time.monotonic()
            for _ in range(20):
                dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
            torch.cuda.synchronize()
            seconds = torch.tensor((time.monotonic() - tick) / 20, device=device)
            dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
            measurements.append(dict(bytes=size, seconds=float(seconds),
                                     algorithm_gbps=size / float(seconds) / 1e9,
                                     estimated_bus_gbps=size / float(seconds) / 1e9 * 2 * (world - 1) / world))
    if rank == 0:
        report.update(collectives=measurements, collective_correctness=True if world > 1 else "not tested: one rank")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
