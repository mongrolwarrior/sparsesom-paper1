"""Output parsers for the four SOM implementations."""

import json
import re


def parse_sparsesom_epoch(line: str) -> dict | None:
    """Parse a per-epoch line from sparsesom stdout."""
    m = re.match(
        r"epoch\s+(\d+)\s+sigma=\s*([\d.]+)\s+QE=([\d.eE+-]+)\s+KL=([\d.eE+-]+)"
        r"\s+path=([\d.eE+-]+)\s+stab=([\d.eE+-]+)\s+TE=([\d.eE+-]+)"
        r"\s+dead=([\d.eE+-]+)%",
        line.strip(),
    )
    if not m:
        return None
    return {
        "epoch": int(m.group(1)),
        "sigma": float(m.group(2)),
        "qe": float(m.group(3)),
        "kl": float(m.group(4)),
        "path": float(m.group(5)),
        "stability": float(m.group(6)),
        "te": float(m.group(7)),
        "dead_frac": float(m.group(8)) / 100.0,
    }


def parse_sparsesom_done(line: str) -> dict | None:
    """Parse the 'done:' summary line from sparsesom."""
    m = re.match(
        r"done:\s+(\d+)\s+epochs?\s+in\s+([\d.]+)s\s+"
        r"\(BMU\s+([\d.]+)s,\s+update\s+([\d.]+)s\)",
        line.strip(),
    )
    if not m:
        return None
    converged = "(converged)" in line
    return {
        "epochs": int(m.group(1)),
        "wall_s": float(m.group(2)),
        "bmu_s": float(m.group(3)),
        "update_s": float(m.group(4)),
        "converged": converged,
    }


def parse_sparsesom_metrics(line: str) -> dict | None:
    """Parse the metrics:{...} JSON line from --eval-corpus."""
    line = line.strip()
    if not line.startswith("metrics:"):
        return None
    return json.loads(line[len("metrics:"):])


def parse_sparsesom_stdout(stdout: str) -> dict:
    """Parse full sparsesom stdout into structured data."""
    epochs = []
    done = None
    metrics = None
    for line in stdout.splitlines():
        ep = parse_sparsesom_epoch(line)
        if ep:
            epochs.append(ep)
        d = parse_sparsesom_done(line)
        if d:
            done = d
        m = parse_sparsesom_metrics(line)
        if m:
            metrics = m
    return {"epochs": epochs, "done": done, "metrics": metrics}


def parse_standardsparsesom_done(line: str) -> dict | None:
    """Parse the 'done:' summary from standardsparsesom."""
    m = re.match(
        r"done:\s+(\d+)\s+epochs?\s+in\s+([\d.]+)s\s+"
        r"\(BMU\s+([\d.]+)s,\s+update\s+([\d.]+)s\)",
        line.strip(),
    )
    if not m:
        return None
    converged = "converged" in line.lower()
    return {
        "epochs": int(m.group(1)),
        "wall_s": float(m.group(2)),
        "bmu_s": float(m.group(3)),
        "update_s": float(m.group(4)),
        "converged": converged,
    }


def parse_standardsparsesom_epoch(line: str) -> dict | None:
    """Parse per-epoch line from standardsparsesom."""
    m = re.match(
        r"epoch\s+(\d+)\s+sigma=([\d.]+)\s+QE=([\d.eE+-]+)",
        line.strip(),
    )
    if not m:
        return None
    result = {
        "epoch": int(m.group(1)),
        "sigma": float(m.group(2)),
        "qe": float(m.group(3)),
    }
    kl_m = re.search(r"KL=([\d.eE+-]+)", line)
    if kl_m:
        result["kl"] = float(kl_m.group(1))
    te_m = re.search(r"TE=([\d.eE+-]+)", line)
    if te_m:
        result["te"] = float(te_m.group(1))
    return result


def parse_standardsparsesom_stdout(stdout: str) -> dict:
    """Parse full standardsparsesom stdout."""
    epochs = []
    done = None
    for line in stdout.splitlines():
        ep = parse_standardsparsesom_epoch(line)
        if ep:
            epochs.append(ep)
        d = parse_standardsparsesom_done(line)
        if d:
            done = d
    return {"epochs": epochs, "done": done}


def parse_medsom_epoch(stdout: str) -> list[dict]:
    """Parse MedSOM per-epoch output (multi-line per epoch).

    Format (sbcsr build):
        Epoch 0/20  radius=16
          QE: 2.99  TE: 0.42  Ave topo dist: 1.9
          Epoch 0 time: 3.412 min
    """
    epochs = []
    current = None
    for line in stdout.splitlines():
        m = re.match(r"Epoch\s+(\d+)/\d+\s+radius=([\d.]+)", line.strip())
        if m:
            current = {"epoch": int(m.group(1)), "radius": float(m.group(2))}
            continue
        if current is not None:
            qe_m = re.search(r"QE:\s*([\d.eE+-]+)", line)
            if qe_m:
                current["qe"] = float(qe_m.group(1))
            te_m = re.search(r"TE:\s*([\d.eE+-]+)", line)
            if te_m:
                current["te"] = float(te_m.group(1))
            time_m = re.search(r"Epoch\s+\d+\s+time:\s*([\d.]+)\s*min", line)
            if time_m:
                current["epoch_time_s"] = float(time_m.group(1)) * 60.0
                epochs.append(current)
                current = None
    return epochs


def parse_medsom_convergence(stdout: str) -> dict | None:
    """Parse MedSOM convergence line."""
    m = re.search(r"Converged after epoch (\d+)", stdout)
    if m:
        return {"converged": True, "converge_epoch": int(m.group(1))}
    return {"converged": False, "converge_epoch": None}


def parse_metrics_json(stdout: str) -> dict:
    """Parse the JSON output from the SparseBinEval metrics tool."""
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise ValueError("No JSON found in metrics output")
