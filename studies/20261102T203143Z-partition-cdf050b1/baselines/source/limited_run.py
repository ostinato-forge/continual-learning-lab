"""Run one local job with low priority, one CPU, and memory guards."""

import argparse
import ctypes
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parent
THREAD_VARIABLES = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
)


def group_usage(group):
    """Read memory and CPU use for the process group started by this launcher."""
    rss = virtual = ticks = members = 0
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) != group:
                continue
            members += 1
            ticks += int(fields[11]) + int(fields[12])
            virtual += int(fields[20])
            rss += int(fields[21]) * os.sysconf("SC_PAGE_SIZE")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return {
        "processes": members, "resident_mib": rss / 2**20,
        "virtual_mib": virtual / 2**20,
        "cpu_seconds": ticks / os.sysconf("SC_CLK_TCK"),
    }


def available_mib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError("Available memory could not be read")


def signal_group(group, sig):
    try:
        os.killpg(group, sig)
    except ProcessLookupError:
        pass


def run_limited(command, output_root, cpu_percent=50, address_mib=2048,
                resident_mib=1024, reserve_mib=4096, wall_seconds=10800):
    if not command or not 1 <= cpu_percent <= 100:
        raise ValueError("Provide a command and a CPU percentage from 1 to 100")
    if not 0 < resident_mib <= address_mib or reserve_mib < 0 or wall_seconds <= 0:
        raise ValueError("Invalid memory or time limit")
    lock = (ROOT / ".resource-run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("Another limited lab job is running") from None
    if available_mib() < reserve_mib + resident_mib:
        lock.close()
        raise RuntimeError("There is not enough free memory to start safely")
    cpus = sorted(os.sched_getaffinity(0))
    cpu = cpus[-1]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = output_root / (stamp + "-limited-" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    settings = {
        "command": command, "cwd": str(Path.cwd()), "cpu": cpu,
        "cpu_percent_of_one_core": cpu_percent, "nice": 15,
        "address_space_limit_mib_per_process": address_mib,
        "resident_watchdog_mib_per_group": resident_mib,
        "minimum_available_memory_mib": reserve_mib,
        "wall_seconds_limit": wall_seconds, "library_threads": 1,
        "memory_poll_seconds": 0.2,
    }
    (output / "limits.json").write_text(json.dumps(settings, indent=2) + "\n")
    shutil.copy2(Path(__file__), output / "launcher_source.py")
    env = {**os.environ, **{name: "1" for name in THREAD_VARIABLES}}
    env["MALLOC_ARENA_MAX"] = "2"
    env["CLL_RESOURCE_LIMITS"] = str(output / "limits.json")
    parent = os.getpid()

    def prepare_child():
        # Kill the child if its supervising launcher exits unexpectedly.
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "Could not set the parent-death signal")
        if os.getppid() != parent:
            os._exit(125)
        os.sched_setaffinity(0, {cpu})
        os.nice(15)
        cap = address_mib * 2**20
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    interrupted = False

    def interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True

    old_handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    process = None
    started = time.monotonic()
    peak = 0.0
    last_cpu = 0.0
    reason = None
    period = 0.2
    active = period * cpu_percent / 100
    next_sample = 0.0
    print("Limited job: " + str(output), flush=True)
    try:
        with (output / "output.log").open("x") as log, (output / "resources.jsonl").open("x") as samples:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=env, preexec_fn=prepare_child, start_new_session=True)
            while process.poll() is None:
                usage = group_usage(process.pid)
                peak = max(peak, usage["resident_mib"])
                last_cpu = max(last_cpu, usage["cpu_seconds"])
                elapsed = time.monotonic() - started
                free = available_mib()
                if elapsed >= next_sample:
                    samples.write(json.dumps({"elapsed_seconds": elapsed, "available_mib": free, **usage}) + "\n")
                    samples.flush()
                    next_sample = elapsed + 10
                if interrupted:
                    reason = "interrupted"
                elif usage["resident_mib"] > resident_mib:
                    reason = "resident_memory_limit"
                elif free < reserve_mib:
                    reason = "low_available_memory"
                elif elapsed >= wall_seconds:
                    reason = "wall_time_limit"
                if reason:
                    signal_group(process.pid, signal.SIGCONT)
                    signal_group(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        signal_group(process.pid, signal.SIGKILL)
                        process.wait()
                    break
                signal_group(process.pid, signal.SIGCONT)
                time.sleep(active)
                signal_group(process.pid, signal.SIGSTOP)
                time.sleep(period - active)
            process.wait()
        result = {
            "status": "completed" if process.returncode == 0 and reason is None else "stopped" if reason else "failed",
            "exit_code": process.returncode, "stop_reason": reason,
            "wall_seconds": time.monotonic() - started,
            "sampled_peak_resident_mib": peak, "sampled_cpu_seconds": last_cpu,
        }
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return process.returncode if process.returncode != 0 else 124 if reason else 0
    finally:
        if process is not None:
            signal_group(process.pid, signal.SIGKILL)
            process.wait()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "checks/limited")
    parser.add_argument("--cpu-percent", type=int, default=50)
    parser.add_argument("--address-mib", type=int, default=2048)
    parser.add_argument("--resident-mib", type=int, default=1024)
    parser.add_argument("--reserve-mib", type=int, default=4096)
    parser.add_argument("--wall-seconds", type=int, default=10800)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    sys.exit(run_limited(command, args.output, args.cpu_percent, args.address_mib,
                         args.resident_mib, args.reserve_mib, args.wall_seconds))


if __name__ == "__main__":
    main()
