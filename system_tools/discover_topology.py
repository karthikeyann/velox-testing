#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""
Discover system interconnect topology (CPU, GPU, NIC, PCIe, NVLink, C2C)
and render a block diagram.

Supports:
  - x86 + discrete GPU systems (DGX/HGX B200, H100, etc.)
  - Grace Hopper (GH200) with NVLink-C2C
  - Grace Blackwell (GB200) with NVLink-C2C

Usage:
  python discover_topology.py [--output-dir DIR] [--format {png,svg,dot,json,mermaid,all}]
                               [--no-render] [--verbose]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class DeviceType(str, Enum):
    CPU = "CPU"
    GPU = "GPU"
    NIC = "NIC"
    NVME = "NVMe"
    ETHERNET = "Ethernet"
    PCIE_SWITCH = "PCIe Switch"
    PCIE_BRIDGE = "PCIe Bridge"
    MEMORY = "Memory"


class SystemType(str, Enum):
    X86_DISCRETE = "x86 + Discrete GPU"
    GRACE_HOPPER = "Grace Hopper (GH200)"
    GRACE_BLACKWELL = "Grace Blackwell (GB200)"


TOPO_CONNECTION_TYPES = {
    "C2C": "NVLink-C2C",
    "PIX": "PCIe Switch",
    "PXB": "PCIe Bridge",
    "PHB": "PCIe Host Bridge",
    "NODE": "Same NUMA node",
    "SYS": "Cross-socket",
}

PCIE_SPEED_TABLE = {
    "2.5 GT/s": 0.250,
    "5.0 GT/s": 0.500,
    "5 GT/s": 0.500,
    "8.0 GT/s": 0.985,
    "8 GT/s": 0.985,
    "16.0 GT/s": 1.969,
    "16 GT/s": 1.969,
    "32.0 GT/s": 3.938,
    "32 GT/s": 3.938,
    "64.0 GT/s": 7.563,
    "64 GT/s": 7.563,
}

C2C_BW_GBPS = 900.0


@dataclass
class Device:
    name: str
    device_type: DeviceType
    pci_bdf: str = ""
    numa_node: int = -1
    details: dict = field(default_factory=dict)


@dataclass
class Link:
    src: str
    dst: str
    link_type: str
    bw_gbps: float = 0.0
    bidirectional: bool = True


@dataclass
class Topology:
    system_type: SystemType = SystemType.X86_DISCRETE
    cpu_arch: str = "x86_64"
    devices: list[Device] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)
    raw: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CLI runners
# ---------------------------------------------------------------------------

def _run(cmd: str, timeout: int = 60) -> str:
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return r.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------

def collect_lscpu() -> dict:
    raw = _run("lscpu")
    info: dict = {"raw": raw}
    for line in raw.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            info[k.strip()] = v.strip()
    return info


def collect_numactl() -> dict:
    raw = _run("numactl -H")
    info: dict = {"raw": raw, "nodes": {}}
    current_node = None
    for line in raw.splitlines():
        m = re.match(r"node (\d+) cpus: (.*)", line)
        if m:
            current_node = int(m.group(1))
            info["nodes"][current_node] = {"cpus": m.group(2).strip()}
        m = re.match(r"node (\d+) size: (\d+) MB", line)
        if m:
            nid = int(m.group(1))
            info["nodes"].setdefault(nid, {})["size_mb"] = int(m.group(2))
    return info


def collect_gpu_info() -> list[dict]:
    raw = _run(
        "nvidia-smi --query-gpu=index,name,pci.bus_id "
        "--format=csv,noheader"
    )
    gpus = []
    for line in raw.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"index": int(parts[0]), "name": parts[1], "bdf": parts[2]})
    return gpus


def collect_topo_matrix() -> str:
    return _run("nvidia-smi topo -m")


def collect_nvlink_status() -> dict[int, list[float]]:
    """Returns {gpu_index: [link_bw_gbps, ...]}."""
    raw = _run("nvidia-smi nvlink -s")
    result: dict[int, list[float]] = {}
    current_gpu = -1
    for line in raw.splitlines():
        m = re.match(r"GPU (\d+):", line)
        if m:
            current_gpu = int(m.group(1))
            result[current_gpu] = []
            continue
        m = re.search(r"Link \d+:\s+([\d.]+)\s+GB/s", line)
        if m and current_gpu >= 0:
            result[current_gpu].append(float(m.group(1)))
    return result


def collect_pcie_link(bdf: str) -> dict:
    """Read PCIe link speed/width from sysfs (no sudo needed)."""
    bdf_sysfs = bdf.lower().replace("00000000:", "0000:")
    base = f"/sys/bus/pci/devices/{bdf_sysfs}"
    speed_raw = ""
    width_raw = ""
    try:
        with open(f"{base}/current_link_speed") as f:
            speed_raw = f.read().strip()
        with open(f"{base}/current_link_width") as f:
            width_raw = f.read().strip()
    except (OSError, FileNotFoundError):
        pass

    speed_key = speed_raw.replace("PCIe", "").strip()
    per_lane = PCIE_SPEED_TABLE.get(speed_key, 0.0)
    try:
        width = int(width_raw)
    except ValueError:
        width = 0
    bw_per_dir = per_lane * width
    gen = ""
    for k, v in [("64", "5"), ("32", "5"), ("16", "4"), ("8", "3"), ("5", "2"), ("2.5", "1")]:
        if speed_key.startswith(k):
            gen = v
            break
    return {
        "speed_raw": speed_raw,
        "width": width,
        "gen": gen,
        "bw_per_dir_gbps": round(bw_per_dir, 2),
        "bw_bidi_gbps": round(bw_per_dir * 2, 2),
        "label": f"PCIe Gen{gen} x{width}" if gen else speed_raw,
    }


def collect_ibstat() -> dict[str, dict]:
    """Returns {ca_name: {rate, state, link_layer, ...}}."""
    raw = _run("ibstat")
    result: dict[str, dict] = {}
    current_ca: Optional[str] = None
    for line in raw.splitlines():
        m = re.match(r"CA '(\S+)'", line)
        if m:
            current_ca = m.group(1)
            result[current_ca] = {}
            continue
        if current_ca and ":" in line:
            k, v = line.strip().split(":", 1)
            k, v = k.strip(), v.strip()
            if k == "Rate":
                try:
                    result[current_ca]["rate_gbps"] = float(v)
                except ValueError:
                    result[current_ca]["rate_raw"] = v
            elif k == "State":
                result[current_ca]["state"] = v
            elif k == "Link layer":
                result[current_ca]["link_layer"] = v
    return result


def collect_rdma_bdf_map() -> dict[str, str]:
    """Map RDMA/IB device names (mlx5_X) to PCIe BDFs via sysfs."""
    result: dict[str, str] = {}
    ib_dir = Path("/sys/class/infiniband")
    if not ib_dir.exists():
        return result
    for dev_path in ib_dir.iterdir():
        name = dev_path.name
        device_link = dev_path / "device"
        if device_link.exists():
            try:
                bdf = device_link.resolve().name
                result[name] = bdf
            except OSError:
                pass
    return result


def collect_lspci_names() -> dict[str, str]:
    """Map PCIe BDFs to human-readable device names from lspci.

    Returns {bdf_with_domain: short_product_name}, e.g.
    {"0000:05:00.0": "ConnectX-7"}.
    """
    raw = _run("lspci")
    result: dict[str, str] = {}
    for line in raw.splitlines():
        m = re.match(r"(\S+)\s+(.+)", line)
        if not m:
            continue
        short_bdf = m.group(1)
        description = m.group(2)
        full_bdf = f"0000:{short_bdf}"

        product = _extract_product_name(description)
        if product:
            result[full_bdf] = product
    return result


INTERESTING_CLASSES = {
    "0108": DeviceType.NVME,
    "0200": DeviceType.ETHERNET,
    "0207": DeviceType.NIC,
    "0302": DeviceType.GPU,
    "0300": DeviceType.GPU,
}

SKIP_VENDORS = {"1a03"}  # ASPEED BMC graphics


@dataclass
class PcieTreeNode:
    """A node in the PCIe device tree."""
    bdf: str
    name: str
    product: str
    device_type: Optional[DeviceType]
    numa_node: int
    pcie_link: dict
    children: list[PcieTreeNode] = field(default_factory=list)
    is_bridge: bool = False
    is_interesting: bool = False


def collect_pcie_tree() -> dict[str, list[PcieTreeNode]]:
    """Build PCIe device tree from sysfs, pruned to interesting devices.

    Returns {root_complex_bdf: [top-level PcieTreeNode children]}.
    Each root complex represents a CPU root port.
    """
    lspci_raw = _run("lspci -n")
    bdf_class: dict[str, str] = {}
    bdf_vendor: dict[str, str] = {}
    for line in lspci_raw.splitlines():
        m = re.match(r"(\S+)\s+(\S+):\s+(\S+)", line)
        if m:
            full_bdf = f"0000:{m.group(1)}"
            bdf_class[full_bdf] = m.group(2)
            vendor = m.group(3).split(":")[0] if ":" in m.group(3) else ""
            bdf_vendor[full_bdf] = vendor

    lspci_names = collect_lspci_names()

    interesting_bdfs: set[str] = set()
    seen_base: set[str] = set()
    for bdf in sorted(bdf_class.keys()):
        cls = bdf_class[bdf]
        cls_short = cls[:4]
        if cls_short not in INTERESTING_CLASSES:
            continue
        vendor = bdf_vendor.get(bdf, "")
        if vendor in SKIP_VENDORS:
            continue
        base_bdf = bdf.rsplit(".", 1)[0]
        if base_bdf in seen_base:
            continue
        seen_base.add(base_bdf)
        interesting_bdfs.add(bdf)

    chains: list[list[str]] = []
    for bdf in sorted(interesting_bdfs):
        sysfs_path = Path(f"/sys/bus/pci/devices/{bdf}")
        if not sysfs_path.exists():
            continue
        try:
            real_path = sysfs_path.resolve()
        except OSError:
            continue
        parts = str(real_path).split("/")
        chain = [p for p in parts if re.match(r"(pci)?[0-9a-f]{4}:[0-9a-f]{2}", p)]
        normalized: list[str] = []
        for p in chain:
            if p.startswith("pci"):
                domain_bus = p[3:]
                normalized.append(f"{domain_bus}:00.0")
            elif re.match(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]", p):
                normalized.append(p)
        if normalized:
            chains.append(normalized)

    tree: dict[str, dict] = {}
    for chain in chains:
        root = chain[0] if chain else "unknown"
        node = tree.setdefault(root, {})
        for bdf in chain[1:]:
            node = node.setdefault(bdf, {})

    def _build_nodes(subtree: dict, parent_bdf: str) -> list[PcieTreeNode]:
        nodes: list[PcieTreeNode] = []
        for bdf, children_dict in subtree.items():
            cls = bdf_class.get(bdf, "")
            cls_short = cls[:4] if cls else ""
            dtype = INTERESTING_CLASSES.get(cls_short)
            is_bridge = cls_short == "0604"
            is_interesting = bdf in interesting_bdfs

            raw_name = lspci_names.get(bdf, "")
            product = raw_name if raw_name else ""
            lspci_desc = ""
            if not product:
                short_bdf = bdf.replace("0000:", "")
                lspci_desc = _run(f"lspci -s {short_bdf}").strip()
                if lspci_desc and " " in lspci_desc:
                    lspci_desc = lspci_desc.split(" ", 1)[1]

            display_name = product or lspci_desc or bdf

            numa = -1
            try:
                with open(f"/sys/bus/pci/devices/{bdf}/numa_node") as f:
                    val = int(f.read().strip())
                    if val >= 0:
                        numa = val
            except (OSError, ValueError):
                pass

            pcie_link = collect_pcie_link(bdf) if (bdf in interesting_bdfs or is_bridge) else {}

            child_nodes = _build_nodes(children_dict, bdf)
            interesting_below = is_interesting or any(
                c.is_interesting or c.children for c in child_nodes
            )

            if not interesting_below and not is_interesting:
                continue

            if not is_interesting and len(child_nodes) == 1 and not is_bridge:
                nodes.extend(child_nodes)
                continue

            node = PcieTreeNode(
                bdf=bdf,
                name=display_name,
                product=product,
                device_type=dtype if is_interesting else DeviceType.PCIE_BRIDGE,
                numa_node=numa,
                pcie_link=pcie_link,
                children=child_nodes,
                is_bridge=not is_interesting,
                is_interesting=is_interesting,
            )
            nodes.append(node)
        return nodes

    result: dict[str, list[PcieTreeNode]] = {}
    for root_bdf, subtree in tree.items():
        nodes = _build_nodes(subtree, root_bdf)
        if nodes:
            result[root_bdf] = nodes

    return result


def _extract_product_name(lspci_desc: str) -> str:
    """Extract a concise product name from an lspci description line.

    Examples:
      "Infiniband controller: Mellanox Technologies MT2910 Family [ConnectX-7]"
        -> "ConnectX-7"
      "Mellanox Technologies MT43244 BlueField-3 integrated ConnectX-7 network controller"
        -> "BlueField-3 ConnectX-7"
      "3D controller: NVIDIA Corporation Device 2901 (rev a1)"
        -> "NVIDIA Device 2901"
    """
    bracket = re.search(r"\[([^\]]+)\]", lspci_desc)
    if bracket:
        return bracket.group(1)

    if "BlueField" in lspci_desc:
        m = re.search(r"(BlueField-\d+)\s+.*?(ConnectX-\d+)", lspci_desc)
        if m:
            return f"{m.group(1)} {m.group(2)}"
        m = re.search(r"(BlueField-\d+)", lspci_desc)
        if m:
            return m.group(1)

    if "ConnectX" in lspci_desc:
        m = re.search(r"(ConnectX-\d+)", lspci_desc)
        if m:
            return m.group(1)

    after_colon = lspci_desc.split(":", 1)[-1].strip() if ":" in lspci_desc else lspci_desc
    after_colon = re.sub(r"\(rev [0-9a-f]+\)", "", after_colon).strip()

    for kw in ("NVMe SSD", "NVMe", "SSD"):
        if kw in after_colon:
            cleaned = re.sub(r"\s+", " ", after_colon).strip()
            tokens = cleaned.split()
            if len(tokens) > 5:
                return " ".join(tokens[:5])
            return cleaned

    if "Ethernet Controller" in after_colon or "Ethernet" in after_colon:
        cleaned = re.sub(r"\s+", " ", after_colon).strip()
        return cleaned

    tokens = after_colon.split()
    if len(tokens) > 4:
        return " ".join(tokens[:4])
    return after_colon


# ---------------------------------------------------------------------------
# Parser: build Topology from raw data
# ---------------------------------------------------------------------------

def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def parse_topo_matrix(
    raw_topo: str,
) -> tuple[list[str], dict[tuple[str, str], str], dict[str, int]]:
    """Parse nvidia-smi topo -m matrix.

    Returns (device_header_names,
             {(row_name, col_name): connection_type},
             {device_name: numa_node}).
    """
    raw_topo = _strip_ansi(raw_topo)
    lines = [l for l in raw_topo.splitlines() if l.strip()]

    device_headers: list[str] = []
    connections: dict[tuple[str, str], str] = {}
    numa_map: dict[str, int] = {}

    header_line = None
    for line in lines:
        if line.startswith("\t") or re.match(r"\s+(GPU|NIC|CPU)", line):
            header_line = line
            break
    if not header_line:
        for line in lines:
            if "GPU0" in line or "NIC0" in line:
                header_line = line
                break
    if not header_line:
        return device_headers, connections, numa_map

    all_cols = [p.strip() for p in header_line.split("\t")]
    numa_col_idx = -1
    for i, col in enumerate(all_cols):
        if col == "NUMA Affinity":
            numa_col_idx = i
            break

    meta_labels = {"CPU Affinity", "NUMA Affinity", "GPU NUMA ID"}
    device_headers = [c for c in all_cols if c and c not in meta_labels]

    data_lines = []
    for line in lines:
        if re.match(r"(GPU|NIC)\d+\s", line):
            data_lines.append(line)

    for line in data_lines:
        parts = [p.strip() for p in line.split("\t")]
        row_name = parts[0]
        for i, val in enumerate(parts[1:], start=1):
            if i >= len(all_cols):
                break
            col = all_cols[i]
            if col in meta_labels or not col:
                continue
            if val and val != "X":
                connections[(row_name, col)] = val

        if numa_col_idx >= 0 and numa_col_idx < len(parts):
            numa_val = parts[numa_col_idx].strip()
            if numa_val.isdigit():
                numa_map[row_name] = int(numa_val)

    return device_headers, connections, numa_map


def _parse_nic_legend(raw_topo: str) -> dict[str, str]:
    """Extract NIC Legend mapping from topo output, e.g. NIC0 -> mlx5_0."""
    result: dict[str, str] = {}
    in_legend = False
    for line in raw_topo.splitlines():
        if "NIC Legend" in line:
            in_legend = True
            continue
        if in_legend:
            m = re.match(r"\s*(NIC\d+):\s*(\S+)", line)
            if m:
                result[m.group(1)] = m.group(2)
            elif line.strip() == "":
                if result:
                    break
    return result


def detect_system_type(cpu_arch: str, connections: dict, gpu_names: list[str]) -> SystemType:
    has_c2c = any(v == "C2C" for v in connections.values())
    if cpu_arch == "aarch64":
        if has_c2c:
            cpu_names = set()
            for (a, b), v in connections.items():
                if v == "C2C":
                    if a.startswith("CPU"):
                        cpu_names.add(a)
                    if b.startswith("CPU"):
                        cpu_names.add(b)
            if cpu_names:
                ratio = len(gpu_names) / len(cpu_names) if cpu_names else 1
                if ratio >= 2:
                    return SystemType.GRACE_BLACKWELL
            return SystemType.GRACE_HOPPER
        return SystemType.GRACE_HOPPER
    if has_c2c:
        return SystemType.GRACE_BLACKWELL
    return SystemType.X86_DISCRETE


def build_topology(verbose: bool = False) -> Topology:
    topo = Topology()

    lscpu_info = collect_lscpu()
    topo.cpu_arch = lscpu_info.get("Architecture", "x86_64")
    if verbose:
        topo.raw["lscpu"] = lscpu_info

    numa_info = collect_numactl()
    if verbose:
        topo.raw["numactl"] = numa_info

    gpu_info_list = collect_gpu_info()
    if verbose:
        topo.raw["gpu_info"] = gpu_info_list

    raw_topo = collect_topo_matrix()
    if verbose:
        topo.raw["topo_matrix"] = raw_topo

    nvlink_status = collect_nvlink_status()
    if verbose:
        topo.raw["nvlink_status"] = {str(k): v for k, v in nvlink_status.items()}

    ibstat_info = collect_ibstat()
    if verbose:
        topo.raw["ibstat"] = ibstat_info

    rdma_bdf_map = collect_rdma_bdf_map()
    lspci_names = collect_lspci_names()
    if verbose:
        topo.raw["rdma_bdf_map"] = rdma_bdf_map
        topo.raw["lspci_names"] = lspci_names

    header_names, connections, numa_map = parse_topo_matrix(raw_topo)
    nic_legend = _parse_nic_legend(raw_topo)
    gpu_names_in_topo = [h for h in header_names if h.startswith("GPU")]

    topo.system_type = detect_system_type(topo.cpu_arch, connections, gpu_names_in_topo)

    num_sockets = int(lscpu_info.get("Socket(s)", "1"))
    cores_per_socket = lscpu_info.get("Core(s) per socket", "?")
    cpu_model = lscpu_info.get("Model name", "Unknown CPU")
    for s in range(num_sockets):
        topo.devices.append(Device(
            name=f"CPU{s}",
            device_type=DeviceType.CPU,
            numa_node=s,
            details={
                "model": cpu_model,
                "cores": cores_per_socket,
                "architecture": topo.cpu_arch,
            },
        ))

    # For NICs without NUMA from the topo matrix, infer from topo connections:
    # if a NIC has NODE/PXB/PHB to GPUs on a known NUMA, inherit that NUMA.
    def _infer_nic_numa(nic_name: str) -> int:
        """Infer NUMA for a NIC from its relationship to GPUs with known NUMA."""
        for (a, b), conn in connections.items():
            if conn in ("NODE", "PXB", "PHB", "PIX"):
                partner = b if a == nic_name else (a if b == nic_name else None)
                if partner and partner.startswith("GPU") and partner in numa_map:
                    return numa_map[partner]
        return -1

    gpu_bdf_map: dict[str, str] = {}
    for g in gpu_info_list:
        idx = g["index"]
        name = f"GPU{idx}"
        bdf = g["bdf"]
        gpu_bdf_map[name] = bdf
        numa = numa_map.get(name, -1)
        pcie = collect_pcie_link(bdf)
        topo.devices.append(Device(
            name=name,
            device_type=DeviceType.GPU,
            pci_bdf=bdf,
            numa_node=numa,
            details={
                "model": g["name"],
                "pcie": pcie,
                "nvlink_links": len(nvlink_status.get(idx, [])),
                "nvlink_per_link_gbps": (
                    nvlink_status[idx][0] if nvlink_status.get(idx) else 0
                ),
            },
        ))

    for nic_label in header_names:
        if not nic_label.startswith("NIC"):
            continue
        mlx_name = nic_legend.get(nic_label, nic_label)
        numa = numa_map.get(nic_label, -1)
        if numa < 0:
            numa = _infer_nic_numa(nic_label)
        ib_info = ibstat_info.get(mlx_name, {})
        nic_bdf = rdma_bdf_map.get(mlx_name, "")
        nic_product = lspci_names.get(nic_bdf, "")
        nic_pcie = collect_pcie_link(nic_bdf) if nic_bdf else {}
        topo.devices.append(Device(
            name=nic_label,
            device_type=DeviceType.NIC,
            pci_bdf=nic_bdf,
            numa_node=numa,
            details={
                "mlx_name": mlx_name,
                "product": nic_product,
                "rate_gbps": ib_info.get("rate_gbps", 0),
                "state": ib_info.get("state", "Unknown"),
                "link_layer": ib_info.get("link_layer", "Unknown"),
                "pcie": nic_pcie,
            },
        ))

    # NVLink connections from nvidia-smi topo matrix
    seen_links: set[tuple[str, str]] = set()
    for (row, col), conn in connections.items():
        if row == col or conn == "X":
            continue
        key = tuple(sorted([row, col]))
        if key in seen_links:
            continue

        if re.match(r"NV\d+", conn):
            seen_links.add(key)
            num_links = int(conn[2:])
            src_idx = int(row.replace("GPU", "")) if row.startswith("GPU") else -1
            per_link = 0.0
            if src_idx >= 0 and nvlink_status.get(src_idx):
                per_link = nvlink_status[src_idx][0]
            bw = per_link * num_links * 2 if per_link else 0
            topo.links.append(Link(
                src=row, dst=col,
                link_type=f"NVLink x{num_links}",
                bw_gbps=round(bw, 1),
            ))

        elif conn == "C2C":
            seen_links.add(key)
            topo.links.append(Link(
                src=row, dst=col,
                link_type="NVLink-C2C",
                bw_gbps=C2C_BW_GBPS,
            ))

    # PCIe tree from sysfs: NVMe, bridges, physical hierarchy
    pcie_tree = collect_pcie_tree()
    if verbose:
        topo.raw["pcie_tree_roots"] = list(pcie_tree.keys())

    def _norm_bdf(bdf: str) -> str:
        """Normalize BDF to 0000:xx:xx.x lowercase format."""
        bdf = bdf.lower().strip()
        if re.match(r"[0-9a-f]{8}:", bdf):
            bdf = "0000:" + bdf[9:]
        if not bdf.startswith("0000:"):
            bdf = "0000:" + bdf
        return bdf

    gpu_bdf_to_name = {_norm_bdf(d.pci_bdf): d.name for d in topo.devices if d.device_type == DeviceType.GPU and d.pci_bdf}
    nic_bdf_to_name = {_norm_bdf(d.pci_bdf): d.name for d in topo.devices if d.device_type == DeviceType.NIC and d.pci_bdf}
    existing_bdfs = gpu_bdf_to_name | nic_bdf_to_name

    bdf_to_topo_name: dict[str, str] = dict(existing_bdfs)
    nvme_counter = 0
    eth_counter = 0
    bridge_counter = 0

    def _flatten_tree(nodes: list[PcieTreeNode], parent_name: str) -> None:
        nonlocal nvme_counter, eth_counter, bridge_counter
        for node in nodes:
            if node.bdf in bdf_to_topo_name:
                this_name = bdf_to_topo_name[node.bdf]
            elif node.is_interesting:
                if node.device_type == DeviceType.NVME:
                    this_name = f"NVMe{nvme_counter}"
                    nvme_counter += 1
                    topo.devices.append(Device(
                        name=this_name,
                        device_type=DeviceType.NVME,
                        pci_bdf=node.bdf,
                        numa_node=node.numa_node,
                        details={
                            "product": node.product or node.name,
                            "pcie": node.pcie_link,
                        },
                    ))
                elif node.device_type == DeviceType.ETHERNET:
                    this_name = f"ETH{eth_counter}"
                    eth_counter += 1
                    topo.devices.append(Device(
                        name=this_name,
                        device_type=DeviceType.ETHERNET,
                        pci_bdf=node.bdf,
                        numa_node=node.numa_node,
                        details={
                            "product": node.product or node.name,
                            "pcie": node.pcie_link,
                        },
                    ))
                else:
                    this_name = node.bdf
                    topo.devices.append(Device(
                        name=this_name,
                        device_type=node.device_type or DeviceType.PCIE_BRIDGE,
                        pci_bdf=node.bdf,
                        numa_node=node.numa_node,
                        details={"product": node.product or node.name},
                    ))
                bdf_to_topo_name[node.bdf] = this_name
            elif node.is_bridge and len(node.children) >= 2:
                this_name = f"Bridge{bridge_counter}"
                bridge_counter += 1
                short_name = node.product or node.name
                if len(short_name) > 40:
                    short_name = short_name[:37] + "..."
                topo.devices.append(Device(
                    name=this_name,
                    device_type=DeviceType.PCIE_BRIDGE,
                    pci_bdf=node.bdf,
                    numa_node=node.numa_node,
                    details={"product": short_name},
                ))
                bdf_to_topo_name[node.bdf] = this_name
            else:
                _flatten_tree(node.children, parent_name)
                continue

            this_name = bdf_to_topo_name[node.bdf]
            pcie_info = node.pcie_link
            bw = pcie_info.get("bw_bidi_gbps", 0) if pcie_info else 0
            label = pcie_info.get("label", "PCIe") if pcie_info else "PCIe"
            if parent_name:
                topo.links.append(Link(
                    src=parent_name, dst=this_name,
                    link_type=label,
                    bw_gbps=bw,
                ))

            _flatten_tree(node.children, this_name)

    numa_to_root: dict[int, set[str]] = defaultdict(set)
    for root_bdf, nodes in pcie_tree.items():
        for node in nodes:
            numa = node.numa_node
            if numa < 0:
                def _find_numa(n: PcieTreeNode) -> int:
                    if n.numa_node >= 0:
                        return n.numa_node
                    for c in n.children:
                        r = _find_numa(c)
                        if r >= 0:
                            return r
                    return -1
                numa = _find_numa(node)
            if numa >= 0:
                numa_to_root[numa].add(root_bdf)

    for root_bdf, nodes in pcie_tree.items():
        cpu_name = None
        for numa_id, roots in numa_to_root.items():
            if root_bdf in roots:
                cpu_name = f"CPU{numa_id}"
                break
        _flatten_tree(nodes, cpu_name or "")

    # Memory devices
    for nid, info in numa_info.get("nodes", {}).items():
        size_mb = info.get("size_mb", 0)
        size_gb = round(size_mb / 1024, 1)
        if size_gb > 0:
            mem_name = f"MEM{nid}"
            topo.devices.append(Device(
                name=mem_name,
                device_type=DeviceType.MEMORY,
                numa_node=nid,
                details={"size_gb": size_gb},
            ))
            topo.links.append(Link(
                src=f"CPU{nid}", dst=mem_name,
                link_type="DDR5",
                bw_gbps=0,
            ))

    # Inter-socket link
    if topo.system_type == SystemType.X86_DISCRETE and num_sockets >= 2:
        topo.links.append(Link(
            src="CPU0", dst="CPU1",
            link_type="UPI",
            bw_gbps=82.0,
        ))

    return topo


def _find_device(topo: Topology, name: str) -> Optional[Device]:
    for d in topo.devices:
        if d.name == name:
            return d
    return None


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------

def export_json(topo: Topology, path: Path) -> None:
    data = {
        "system_type": topo.system_type.value,
        "cpu_arch": topo.cpu_arch,
        "devices": [
            {
                "name": d.name,
                "type": d.device_type.value,
                "pci_bdf": d.pci_bdf,
                "numa_node": d.numa_node,
                "details": d.details,
            }
            for d in topo.devices
        ],
        "links": [
            {
                "src": l.src,
                "dst": l.dst,
                "link_type": l.link_type,
                "bw_gbps": l.bw_gbps,
                "bidirectional": l.bidirectional,
            }
            for l in topo.links
        ],
    }
    if topo.raw:
        data["raw_collector_output"] = topo.raw
    path.write_text(json.dumps(data, indent=2, default=str))
    log.info("JSON exported to %s", path)


# ---------------------------------------------------------------------------
# DOT renderer
# ---------------------------------------------------------------------------

def _nic_detail_label(d: Device, sep: str = "\n") -> str:
    """Build a multi-line detail string for a NIC device."""
    product = d.details.get("product", "")
    mlx = d.details.get("mlx_name", "")
    parts = []
    if product:
        parts.append(product)
    if mlx:
        parts.append(mlx)
    return (sep + sep.join(parts)) if parts else ""


DOT_COLORS = {
    DeviceType.CPU: "#3B7DD8",
    DeviceType.GPU: "#5DA845",
    DeviceType.NIC: "#D98C21",
    DeviceType.NVME: "#E67E22",
    DeviceType.ETHERNET: "#1ABC9C",
    DeviceType.PCIE_SWITCH: "#8E44AD",
    DeviceType.PCIE_BRIDGE: "#8E44AD",
    DeviceType.MEMORY: "#2980B9",
}

LINK_COLORS = {
    "NVLink-C2C": "#E74C3C",
    "NVLink": "#2ECC71",
    "PCIe": "#3498DB",
    "UPI": "#E74C3C",
    "DDR": "#2980B9",
}


def _link_color(link_type: str) -> str:
    for prefix, color in LINK_COLORS.items():
        if link_type.startswith(prefix):
            return color
    return "#7F8C8D"


def _link_penwidth(link_type: str) -> str:
    if "NVLink" in link_type:
        return "3.0"
    if "C2C" in link_type:
        return "3.5"
    if "UPI" in link_type:
        return "2.0"
    return "1.5"


def _dot_node_id(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def export_dot(topo: Topology, path: Path) -> None:
    lines = [
        "digraph SystemTopology {",
        '  rankdir=TB;',
        '  bgcolor="white";',
        '  node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=11];',
        '  edge [fontname="Helvetica", fontsize=9];',
        "",
    ]

    by_numa: dict[int, list[Device]] = defaultdict(list)
    for d in topo.devices:
        by_numa[d.numa_node].append(d)

    for numa_id in sorted(by_numa):
        if numa_id < 0:
            continue
        label = f"NUMA Node {numa_id}"
        lines.append(f'  subgraph cluster_numa{numa_id} {{')
        lines.append(f'    label="{label}";')
        lines.append('    style=dashed; color="#95A5A6";')
        for d in by_numa[numa_id]:
            nid = _dot_node_id(d.name)
            color = DOT_COLORS.get(d.device_type, "#BDC3C7")
            detail = ""
            if d.device_type == DeviceType.GPU:
                detail = f"\\n{d.details.get('model', '')}"
            elif d.device_type == DeviceType.CPU:
                detail = f"\\n{d.details.get('model', '')}"
            elif d.device_type == DeviceType.NIC:
                detail = _nic_detail_label(d, sep="\\n")
            elif d.device_type == DeviceType.NVME:
                detail = f"\\n{d.details.get('product', '')}"
            elif d.device_type == DeviceType.ETHERNET:
                detail = f"\\n{d.details.get('product', '')}"
            elif d.device_type == DeviceType.PCIE_BRIDGE:
                detail = f"\\n{d.details.get('product', '')}"
            elif d.device_type == DeviceType.MEMORY:
                detail = f"\\n{d.details.get('size_gb', '')} GB"
            lines.append(
                f'    {nid} [label="{d.name}{detail}", '
                f'fillcolor="{color}", fontcolor="white"];'
            )
        lines.append("  }")

    for d in by_numa.get(-1, []):
        nid = _dot_node_id(d.name)
        color = DOT_COLORS.get(d.device_type, "#BDC3C7")
        lines.append(
            f'  {nid} [label="{d.name}", fillcolor="{color}", fontcolor="white"];'
        )

    lines.append("")

    seen_edges: set[tuple[str, str]] = set()
    for link in topo.links:
        key = tuple(sorted([link.src, link.dst]))
        if key in seen_edges:
            continue
        seen_edges.add(key)

        src_id = _dot_node_id(link.src)
        dst_id = _dot_node_id(link.dst)
        color = _link_color(link.link_type)
        pw = _link_penwidth(link.link_type)
        bw_label = f"{link.bw_gbps:.0f} GB/s" if link.bw_gbps else ""
        label = f"{link.link_type}"
        if bw_label:
            label += f"\\n{bw_label}"
        style = 'style=dashed, ' if link.link_type == "UPI" else ""
        direction = "dir=both, " if link.bidirectional else ""
        lines.append(
            f'  {src_id} -> {dst_id} [{direction}{style}'
            f'label="{label}", color="{color}", penwidth={pw}];'
        )

    lines.append("}")
    path.write_text("\n".join(lines))
    log.info("DOT file exported to %s", path)


# ---------------------------------------------------------------------------
# Mermaid renderer
# ---------------------------------------------------------------------------

MERMAID_LINK_STYLES = {
    "NVLink-C2C": "stroke:#E74C3C,stroke-width:3px",
    "NVLink": "stroke:#2ECC71,stroke-width:3px",
    "PCIe": "stroke:#3498DB,stroke-width:2px",
    "UPI": "stroke:#E74C3C,stroke-width:2px,stroke-dasharray:5 5",
    "DDR": "stroke:#2980B9,stroke-width:2px,stroke-dasharray:3 3",
}


def _mermaid_node_id(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def _mermaid_link_style(link_type: str) -> str:
    for prefix, style in MERMAID_LINK_STYLES.items():
        if link_type.startswith(prefix):
            return style
    return "stroke:#7F8C8D,stroke-width:1px"


def export_mermaid(topo: Topology, path: Path) -> None:
    lines: list[str] = []
    lines.append(f"---")
    lines.append(f"title: System Topology - {topo.system_type.value} ({topo.cpu_arch})")
    lines.append(f"---")
    lines.append("graph TB")

    by_numa: dict[int, list[Device]] = defaultdict(list)
    for d in topo.devices:
        by_numa[d.numa_node].append(d)

    for numa_id in sorted(by_numa):
        if numa_id < 0:
            continue
        lines.append(f"    subgraph NUMA{numa_id} [NUMA Node {numa_id}]")
        for d in by_numa[numa_id]:
            nid = _mermaid_node_id(d.name)
            label = _mermaid_node_label(d)
            shape = _mermaid_node_shape(d)
            lines.append(f"        {nid}{shape[0]}\"{label}\"{shape[1]}")
        lines.append("    end")

    for d in by_numa.get(-1, []):
        nid = _mermaid_node_id(d.name)
        label = _mermaid_node_label(d)
        shape = _mermaid_node_shape(d)
        lines.append(f"    {nid}{shape[0]}\"{label}\"{shape[1]}")

    lines.append("")

    link_idx = 0
    style_directives: list[str] = []
    seen_edges: set[tuple[str, str]] = set()
    for link in topo.links:
        key = tuple(sorted([link.src, link.dst]))
        if key in seen_edges:
            continue
        seen_edges.add(key)

        src_id = _mermaid_node_id(link.src)
        dst_id = _mermaid_node_id(link.dst)
        bw = f" {link.bw_gbps:.0f} GB/s" if link.bw_gbps else ""
        label = f"{link.link_type}{bw}"
        arrow = " <-->" if link.bidirectional else " -->"

        lines.append(f"    {src_id}{arrow}|\"{label}\"| {dst_id}")
        style_directives.append(
            f"    linkStyle {link_idx} {_mermaid_link_style(link.link_type)}"
        )
        link_idx += 1

    if style_directives:
        lines.append("")
        lines.extend(style_directives)

    lines.append("")
    lines.append("    classDef cpuNode fill:#3B7DD8,stroke:#2C5F9E,color:#fff")
    lines.append("    classDef gpuNode fill:#5DA845,stroke:#468033,color:#fff")
    lines.append("    classDef nicNode fill:#D98C21,stroke:#B07019,color:#fff")
    lines.append("    classDef nvmeNode fill:#E67E22,stroke:#C0651E,color:#fff")
    lines.append("    classDef ethNode fill:#1ABC9C,stroke:#16A085,color:#fff")
    lines.append("    classDef bridgeNode fill:#8E44AD,stroke:#6C3483,color:#fff")
    lines.append("    classDef memNode fill:#2980B9,stroke:#1F6391,color:#fff")

    cls_map = {
        DeviceType.CPU: "cpuNode",
        DeviceType.GPU: "gpuNode",
        DeviceType.NIC: "nicNode",
        DeviceType.NVME: "nvmeNode",
        DeviceType.ETHERNET: "ethNode",
        DeviceType.PCIE_SWITCH: "bridgeNode",
        DeviceType.PCIE_BRIDGE: "bridgeNode",
        DeviceType.MEMORY: "memNode",
    }
    for d in topo.devices:
        nid = _mermaid_node_id(d.name)
        cls = cls_map.get(d.device_type, "")
        if cls:
            lines.append(f"    class {nid} {cls}")

    lines.append("")
    path.write_text("\n".join(lines))
    log.info("Mermaid diagram exported to %s", path)


def _mermaid_node_label(d: Device) -> str:
    if d.device_type == DeviceType.GPU:
        return f"{d.name}<br/>{d.details.get('model', '')}"
    if d.device_type == DeviceType.CPU:
        model = d.details.get("model", "")
        if len(model) > 30:
            model = model[:27] + "..."
        return f"{d.name}<br/>{model}"
    if d.device_type == DeviceType.NIC:
        return f"{d.name}{_nic_detail_label(d, sep='<br/>')}"
    if d.device_type == DeviceType.NVME:
        product = d.details.get("product", "")
        if len(product) > 35:
            product = product[:32] + "..."
        return f"{d.name}<br/>{product}"
    if d.device_type == DeviceType.ETHERNET:
        product = d.details.get("product", "")
        if len(product) > 35:
            product = product[:32] + "..."
        return f"{d.name}<br/>{product}"
    if d.device_type == DeviceType.PCIE_BRIDGE:
        product = d.details.get("product", "")
        if len(product) > 35:
            product = product[:32] + "..."
        return f"{d.name}<br/>{product}"
    if d.device_type == DeviceType.MEMORY:
        return f"{d.name}<br/>{d.details.get('size_gb', '')} GB"
    if d.device_type == DeviceType.PCIE_SWITCH:
        members = d.details.get("members", [])
        return f"{d.name}<br/>({', '.join(members)})"
    return d.name


def _mermaid_node_shape(d: Device) -> tuple[str, str]:
    """Return (open_bracket, close_bracket) for the mermaid node shape."""
    if d.device_type in (DeviceType.CPU, DeviceType.GPU):
        return ("[", "]")
    if d.device_type in (DeviceType.NIC, DeviceType.ETHERNET):
        return ("([", "])")
    if d.device_type in (DeviceType.PCIE_SWITCH, DeviceType.PCIE_BRIDGE):
        return ("{{", "}}")
    if d.device_type == DeviceType.NVME:
        return ("[(", ")]")
    if d.device_type == DeviceType.MEMORY:
        return ("[(", ")]")
    return ("[", "]")


# ---------------------------------------------------------------------------
# Matplotlib + networkx renderer
# ---------------------------------------------------------------------------

def render_matplotlib(topo: Topology, path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import networkx as nx
    except ImportError:
        log.error("matplotlib/networkx not installed — skipping PNG render")
        return

    G = nx.Graph()

    node_colors = {}
    node_labels = {}
    for d in topo.devices:
        G.add_node(d.name)
        color_map = {
            DeviceType.CPU: "#3B7DD8",
            DeviceType.GPU: "#5DA845",
            DeviceType.NIC: "#D98C21",
            DeviceType.NVME: "#E67E22",
            DeviceType.ETHERNET: "#1ABC9C",
            DeviceType.PCIE_SWITCH: "#8E44AD",
            DeviceType.PCIE_BRIDGE: "#8E44AD",
            DeviceType.MEMORY: "#2980B9",
        }
        node_colors[d.name] = color_map.get(d.device_type, "#BDC3C7")
        if d.device_type == DeviceType.GPU:
            node_labels[d.name] = f"{d.name}\n{d.details.get('model', '')}"
        elif d.device_type == DeviceType.CPU:
            model = d.details.get("model", "")
            short_model = model.split("(")[0].strip() if "(" in model else model
            if len(short_model) > 25:
                short_model = short_model[:22] + "..."
            node_labels[d.name] = f"{d.name}\n{short_model}"
        elif d.device_type == DeviceType.NIC:
            node_labels[d.name] = f"{d.name}{_nic_detail_label(d)}"
        elif d.device_type == DeviceType.NVME:
            product = d.details.get("product", "")
            if len(product) > 25:
                product = product[:22] + "..."
            node_labels[d.name] = f"{d.name}\n{product}"
        elif d.device_type == DeviceType.ETHERNET:
            product = d.details.get("product", "")
            if len(product) > 25:
                product = product[:22] + "..."
            node_labels[d.name] = f"{d.name}\n{product}"
        elif d.device_type == DeviceType.PCIE_BRIDGE:
            product = d.details.get("product", "")
            if len(product) > 25:
                product = product[:22] + "..."
            node_labels[d.name] = f"{d.name}\n{product}"
        elif d.device_type == DeviceType.MEMORY:
            node_labels[d.name] = f"{d.name}\n{d.details.get('size_gb', '')} GB"
        else:
            node_labels[d.name] = d.name

    edge_labels = {}
    edge_colors = []
    edge_widths = []
    seen: set[tuple[str, str]] = set()
    for link in topo.links:
        key = tuple(sorted([link.src, link.dst]))
        if key in seen:
            continue
        if link.src not in G.nodes or link.dst not in G.nodes:
            continue
        seen.add(key)
        G.add_edge(link.src, link.dst)
        bw = f"\n{link.bw_gbps:.0f} GB/s" if link.bw_gbps else ""
        edge_labels[(link.src, link.dst)] = f"{link.link_type}{bw}"
        edge_colors.append(_link_color(link.link_type))
        if "NVLink" in link.link_type and "C2C" not in link.link_type:
            edge_widths.append(2.5)
        elif "C2C" in link.link_type:
            edge_widths.append(3.0)
        elif "UPI" in link.link_type:
            edge_widths.append(2.0)
        else:
            edge_widths.append(1.2)

    pos = _compute_layout(topo, G)

    num_nodes = len(G.nodes)
    fig_w = max(20, num_nodes * 1.2)
    fig_h = max(14, num_nodes * 0.8)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.set_title(
        f"System Topology — {topo.system_type.value}\n"
        f"({topo.cpu_arch})",
        fontsize=16, fontweight="bold", pad=20,
    )

    by_numa: dict[int, list[Device]] = defaultdict(list)
    for d in topo.devices:
        if d.numa_node >= 0:
            by_numa[d.numa_node].append(d)

    for numa_id, devs in by_numa.items():
        dev_names = [d.name for d in devs if d.name in pos]
        if not dev_names:
            continue
        xs = [pos[n][0] for n in dev_names]
        ys = [pos[n][1] for n in dev_names]
        margin = 0.8
        rect = mpatches.FancyBboxPatch(
            (min(xs) - margin, min(ys) - margin),
            max(xs) - min(xs) + 2 * margin,
            max(ys) - min(ys) + 2 * margin,
            boxstyle="round,pad=0.3",
            facecolor="#F0F0F0",
            edgecolor="#95A5A6",
            linestyle="--",
            linewidth=1.5,
            zorder=0,
        )
        ax.add_patch(rect)
        ax.text(
            (min(xs) + max(xs)) / 2,
            max(ys) + margin + 0.3,
            f"NUMA {numa_id}",
            ha="center", fontsize=11, color="#7F8C8D", fontweight="bold",
        )

    ordered_nodes = list(G.nodes)
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        nodelist=ordered_nodes,
        node_color=[node_colors.get(n, "#BDC3C7") for n in ordered_nodes],
        node_size=2800,
        node_shape="s",
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        labels={n: node_labels.get(n, n) for n in ordered_nodes},
        font_size=7, font_color="white", font_weight="bold",
    )

    if edge_colors:
        edges_in_order = list(G.edges)
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=edges_in_order,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.8,
            arrows=True,
            connectionstyle="arc3,rad=0.05",
        )
        nx.draw_networkx_edge_labels(
            G, pos, ax=ax,
            edge_labels=edge_labels,
            font_size=6,
            font_color="#2C3E50",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8, edgecolor="none"),
        )

    legend_handles = [
        mpatches.Patch(color="#3B7DD8", label="CPU"),
        mpatches.Patch(color="#5DA845", label="GPU"),
        mpatches.Patch(color="#D98C21", label="NIC"),
        mpatches.Patch(color="#E67E22", label="NVMe"),
        mpatches.Patch(color="#1ABC9C", label="Ethernet"),
        mpatches.Patch(color="#8E44AD", label="PCIe Bridge"),
        mpatches.Patch(color="#2980B9", label="Memory"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=9, framealpha=0.9)

    ax.axis("off")
    fig.tight_layout()

    suffix = path.suffix.lower()
    fmt = "svg" if suffix == ".svg" else "png"
    fig.savefig(str(path), format=fmt, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    log.info("Diagram saved to %s", path)


def _compute_layout(topo: Topology, G: nx.Graph) -> dict[str, tuple[float, float]]:
    """Compute node positions grouped by NUMA and device type."""
    pos: dict[str, tuple[float, float]] = {}

    by_numa: dict[int, dict[str, list[Device]]] = defaultdict(lambda: defaultdict(list))
    unplaced: list[Device] = []
    for d in topo.devices:
        if d.numa_node >= 0:
            by_numa[d.numa_node][d.device_type.value].append(d)
        else:
            unplaced.append(d)

    numa_ids = sorted(by_numa.keys())
    numa_spacing = 8.0

    for col_idx, numa_id in enumerate(numa_ids):
        x_base = col_idx * numa_spacing
        groups = by_numa[numa_id]

        row = 0
        type_order = ["CPU", "Memory", "PCIe Bridge", "GPU", "NIC",
                      "NVMe", "Ethernet", "PCIe Switch"]
        for dtype_key in type_order:
            devs = groups.get(dtype_key, [])
            if not devs:
                continue
            for i, d in enumerate(devs):
                x = x_base + (i - (len(devs) - 1) / 2) * 2.0
                y = -row * 3.0
                pos[d.name] = (x, y)
            row += 1

    if unplaced:
        x_center = (len(numa_ids) - 1) * numa_spacing / 2
        for i, d in enumerate(unplaced):
            pos[d.name] = (x_center + (i - len(unplaced) / 2) * 2, 3.0)

    for node in G.nodes:
        if node not in pos:
            pos[node] = (0, 0)

    return pos


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover system interconnect topology and render a block diagram.",
    )
    parser.add_argument(
        "--output-dir", default="./topology_output",
        help="Directory for output files (default: ./topology_output)",
    )
    parser.add_argument(
        "--format", default="all",
        choices=["png", "svg", "dot", "json", "mermaid", "all"],
        help="Output format (default: all)",
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="Skip diagram rendering, only output JSON",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Include raw CLI outputs in JSON and print debug info",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    log.info("Collecting system topology...")
    topo = build_topology(verbose=args.verbose)
    log.info("Detected system type: %s (%s)", topo.system_type.value, topo.cpu_arch)
    log.info("Found %d devices and %d links", len(topo.devices), len(topo.links))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    formats = (
        ["png", "dot", "mermaid", "json"]
        if args.format == "all"
        else [args.format]
    )

    if args.no_render:
        formats = ["json"]

    for fmt in formats:
        if fmt == "json":
            export_json(topo, out_dir / "topology.json")
        elif fmt == "dot":
            export_dot(topo, out_dir / "topology.dot")
        elif fmt == "mermaid":
            export_mermaid(topo, out_dir / "topology.mmd")
        elif fmt in ("png", "svg"):
            render_matplotlib(topo, out_dir / f"topology.{fmt}")

    log.info("Done. Output in %s", out_dir.resolve())


if __name__ == "__main__":
    main()
