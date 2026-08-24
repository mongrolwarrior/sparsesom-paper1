"""Run harness: dispatches to SOM implementations, captures output, builds provenance."""

import os
import subprocess
import time
from pathlib import Path

from . import parsers


SPARSESOM = os.environ.get("SPARSESOM", "sparsesom")
STANDARDSPARSESOM = os.environ.get("STANDARDSPARSESOM", "standardsparsesom")
MEDSOM = os.environ.get("MEDSOM", "medsom")
SOMOCLU = os.environ.get("SOMOCLU", "somoclu")
SOM_METRICS = os.environ.get("SOM_METRICS", "som-metrics")


class RunResult:
    def __init__(self, stdout: str, stderr: str, returncode: int,
                 wall_s: float, cmd: list[str]):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.wall_s = wall_s
        self.cmd = cmd


def _run(cmd: list[str], timeout: int = 86400, env: dict | None = None,
         stdin_data: str | None = None, cwd: str | None = None) -> RunResult:
    """Run a command, streaming stdout to the terminal while capturing it for parsing."""
    import io
    import selectors
    import sys

    full_env = dict(os.environ)
    if env:
        full_env.update(env)

    t0 = time.monotonic()
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        stdin=subprocess.PIPE if stdin_data else subprocess.DEVNULL,
        env=full_env, cwd=cwd,
    )

    if stdin_data:
        proc.stdin.write(stdin_data.encode())
        proc.stdin.close()

    stdout_parts = []
    stderr_parts = []
    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ)
    sel.register(proc.stderr, selectors.EVENT_READ)

    open_streams = 2
    while open_streams > 0:
        for key, _ in sel.select(timeout=timeout):
            data = key.fileobj.read1(8192) if hasattr(key.fileobj, 'read1') else key.fileobj.read(8192)
            if not data:
                sel.unregister(key.fileobj)
                open_streams -= 1
                continue
            if key.fileobj is proc.stdout:
                text = data.decode("utf-8", errors="replace")
                stdout_parts.append(text)
                sys.stdout.write(text)
                sys.stdout.flush()
            else:
                stderr_parts.append(data.decode("utf-8", errors="replace"))

    proc.wait(timeout=timeout)
    wall = time.monotonic() - t0

    return RunResult(
        stdout="".join(stdout_parts),
        stderr="".join(stderr_parts),
        returncode=proc.returncode,
        wall_s=wall,
        cmd=cmd,
    )


def run_sparsesom(corpus: str, edge: int, seed: int, *,
                  binary: bool = True,
                  eval_corpus: str | None = None,
                  pca_init: str | None = None,
                  save_weights: str | None = None,
                  sigma_init: float | None = None,
                  extra_flags: list[str] | None = None,
                  timeout: int = 86400) -> dict:
    """Run sparsesom (SparseBinarySOM) and parse output."""
    cmd = [SPARSESOM, corpus, "--rows", str(edge), "--cols", str(edge),
           "--seed", str(seed)]

    if binary:
        cmd.append("--bin")

    if sigma_init is not None:
        cmd.extend(["--sigma-init", str(sigma_init)])
    else:
        cmd.extend(["--sigma-init", str(0.5 * edge)])

    # Deterministic σ schedule (manuscript §3.4): σ = σ_min + (σ₀−σ_min)·exp(−0.3·epoch).
    cmd.extend(["--sigma-sched", "det-exp"])
    # Disable the restart-slower watchdog; KL plateau stop still fires correctly.
    cmd.extend(["--wd-path-frac", "1.0"])

    if pca_init:
        cmd.extend(["--init-pca", pca_init])
    if eval_corpus:
        cmd.extend(["--eval-corpus", eval_corpus])
    if save_weights:
        cmd.extend(["--save-weights", save_weights])
    if extra_flags:
        cmd.extend(extra_flags)

    result = _run(cmd, timeout=timeout)

    if result.returncode != 0:
        raise RuntimeError(
            f"sparsesom failed (exit {result.returncode}):\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stderr: {result.stderr[:500]}"
        )

    parsed = parsers.parse_sparsesom_stdout(result.stdout)
    parsed["raw_stdout"] = result.stdout
    parsed["wall_s_outer"] = result.wall_s
    parsed["cmd"] = cmd
    return parsed


def run_standardsparsesom(corpus: str, edge: int, seed: int, *,
                          layout: str = "feature",
                          neighbourhood: str = "box",
                          stop: str = "kl",
                          precision: str = "fp32",
                          extra_flags: list[str] | None = None,
                          timeout: int = 86400) -> dict:
    """Run standardsparsesom (cuSPARSE baseline) and parse output."""
    cmd = [STANDARDSPARSESOM, corpus,
           "--map", str(edge),
           "--seed", str(seed),
           "--layout", layout,
           "--neighbourhood", neighbourhood,
           "--stop", stop,
           "--precision", precision,
           "--sigma-init", str(0.5 * edge)]

    if extra_flags:
        cmd.extend(extra_flags)

    result = _run(cmd, timeout=timeout)

    if result.returncode != 0:
        if "out of memory" in result.stderr.lower() or "OOM" in result.stderr:
            return {"oom": True, "cmd": cmd, "stderr": result.stderr[:500]}
        raise RuntimeError(
            f"standardsparsesom failed (exit {result.returncode}):\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stderr: {result.stderr[:500]}"
        )

    parsed = parsers.parse_standardsparsesom_stdout(result.stdout)
    parsed["oom"] = False
    parsed["raw_stdout"] = result.stdout
    parsed["wall_s_outer"] = result.wall_s
    parsed["cmd"] = cmd
    return parsed


def ensure_medsom_data(data_dir: str) -> str:
    """Ensure MedSOM-format data exists, converting from .sbcsr if needed."""
    data = Path(data_dir)
    medsom_dir = data / "medsom_input"
    if (medsom_dir / "bin" / "mesh_offsets.bin").exists():
        return str(medsom_dir)

    train_sbcsr = data / "corpus.train.sbcsr"
    if not train_sbcsr.exists():
        raise FileNotFoundError(
            f"No MedSOM data and no .sbcsr to convert at {train_sbcsr}")

    import subprocess, sys
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    subprocess.run(
        [sys.executable, str(scripts_dir / "sbcsr_to_medsom.py"),
         str(train_sbcsr), str(medsom_dir)],
        check=True,
    )
    return str(medsom_dir)


def run_medsom(corpus: str, edge: int, *,
               epochs: int = 20,
               scratch_dir: str | None = None,
               timeout: int = 86400) -> dict:
    """Run MedSOM_Naive and parse output.

    MedSOM takes positional args: <corpus.sbcsr> [epochs] [dimX] [dimY].
    It has a hardcoded seed and no --seed flag, and dumps codebook .dat
    files into its working directory whenever TE < 0.4 — pass scratch_dir
    to keep those out of the repo (caller cleans up).
    """
    cmd = [MEDSOM, str(Path(corpus).resolve()), str(epochs), str(edge)]

    result = _run(cmd, timeout=timeout, cwd=scratch_dir)

    if result.returncode != 0:
        stderr_l = result.stderr.lower()
        if ("out of memory" in stderr_l or "bad_alloc" in stderr_l
                or "cudaerrormemoryallocation" in stderr_l):
            return {"oom": True, "cmd": cmd, "stderr": result.stderr[:500]}
        raise RuntimeError(
            f"medsom failed (exit {result.returncode}):\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stderr: {result.stderr[:500]}"
        )

    epochs_data = parsers.parse_medsom_epoch(result.stdout)
    conv = parsers.parse_medsom_convergence(result.stdout)

    return {
        "epochs": epochs_data,
        "convergence": conv,
        "oom": False,
        "raw_stdout": result.stdout,
        "wall_s_outer": result.wall_s,
        "cmd": cmd,
    }


def run_somoclu(libsvm_corpus: str, edge: int, *,
                epochs: int = 3,
                scratch_dir: str | None = None,
                timeout: int = 172800) -> dict:
    """Run somoclu's sparse CPU kernel (-k 2) and parse per-epoch times.

    libsvm_corpus is a libsvm-format text file (see scripts/sbcsr_to_libsvm.py).
    somoclu prints "Time for epoch N: T" to stderr and dumps codebook files
    into its cwd, so pass scratch_dir to keep those out of the tree.
    Returns {oom, epoch_times, per_epoch_s} (per_epoch_s = mean of steady
    epochs, epoch 0 dropped as warm-up when >1 epoch).
    """
    import re
    out_prefix = str(Path(scratch_dir or ".") / f"somoclu_{edge}")
    cmd = [SOMOCLU, "-k", "2", "-x", str(edge), "-y", str(edge),
           "-e", str(epochs), "-v", "1", libsvm_corpus, out_prefix]
    # epoch times + progress go to stderr; merge so we can parse them
    result = _run(cmd, timeout=timeout)
    combined = result.stdout + result.stderr
    if result.returncode != 0:
        low = combined.lower()
        if ("bad_alloc" in low or "out of memory" in low
                or result.returncode in (-9, 137)):
            return {"oom": True, "cmd": cmd}
        raise RuntimeError(f"somoclu failed (exit {result.returncode}):\n"
                           f"  cmd: {' '.join(cmd)}\n  {combined[-500:]}")
    for suf in (".wts", ".bm", ".umx"):
        Path(out_prefix + suf).unlink(missing_ok=True)
    ep = [float(m) for m in re.findall(r"Time for epoch \d+:\s*([\d.eE+-]+)", combined)]
    steady = ep[1:] if len(ep) > 1 else ep
    return {"oom": False, "epoch_times": ep,
            "per_epoch_s": (sum(steady) / len(steady)) if steady else None,
            "cmd": cmd}


def run_metrics(corpus: str, codebook: str, *,
                codebook_format: str = "somw",
                rows: int = 0, cols: int = 0) -> dict:
    """Run the SparseBinEval metrics evaluator."""
    cmd = [SOM_METRICS,
           "--corpus", corpus,
           "--codebook", codebook,
           "--format", codebook_format]
    if rows:
        cmd.extend(["--rows", str(rows)])
    if cols:
        cmd.extend(["--cols", str(cols)])

    result = _run(cmd, timeout=3600)
    if result.returncode != 0:
        raise RuntimeError(
            f"metrics failed (exit {result.returncode}):\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stderr: {result.stderr[:500]}"
        )
    return parsers.parse_metrics_json(result.stdout)
