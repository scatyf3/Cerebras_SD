"""Per-stage time and GPU utilization from an nsys sqlite export of profile_nvtx.py.

Usage: python analyze_nsys.py run.sqlite run.json

For every NVTX range class: total time per step, and the average of the hardware GPU metrics
sampled inside its windows. If kernels were traced, also kernel count, kernel busy fraction
(union of kernel intervals / wall) and the top kernel categories.
"""
import json, re, sqlite3, sys
from collections import defaultdict
import numpy as np

db, out = sys.argv[1], sys.argv[2]
c = sqlite3.connect(db)
strings = dict(c.execute("select id, value from StringIds"))
ev = []
for s, e, text, tid in c.execute("select start, end, text, textId from NVTX_EVENTS where end is not null and end > start"):
    name = text if text is not None else strings.get(tid, "?")
    if not name.startswith("cub::"):
        ev.append((s, e, name))
steps = [x for x in ev if x[2].startswith("step ")]
n_steps = len(steps)


def cls(name):
    if name.startswith("step "):
        return "step"
    if name.startswith("exact/expand"):
        r = int(re.search(r" r(\d+)", name).group(1))
        return "exact/expand r1" if r == 1 else "exact/expand r>=2"
    return name


groups = defaultdict(list)
for s, e, name in ev:
    groups[cls(name)].append((s, e))

mnames = dict(c.execute("select metricId, metricName from TARGET_INFO_GPU_METRICS"))
want = {
    "GPU Active [Throughput %]": "gpu_active",
    "SMs Active [Throughput %]": "sm_active",
    "SM Issue [Throughput %]": "sm_issue",
    "Tensor Active [Throughput %]": "tensor_active",
    "Compute Warps in Flight [Throughput %]": "warps_in_flight",
    "DRAM Read Bandwidth [Throughput %]": "dram_read",
    "DRAM Write Bandwidth [Throughput %]": "dram_write",
    "PCIe RX Throughput [Throughput %]": "pcie_rx",
    "PCIe TX Throughput [Throughput %]": "pcie_tx",
    "GPC Clock Frequency [MHz]": "gpc_mhz",
}
ids = {mid: want[n] for mid, n in mnames.items() if n in want}
series = {}
for mid, key in ids.items():
    rows = np.array(c.execute("select timestamp, value from GPU_METRICS where metricId=? order by timestamp", (mid,)).fetchall(), dtype=np.float64)
    series[key] = rows


def metric_avg(key, wins):
    rows = series.get(key)
    if rows is None or len(rows) == 0:
        return None
    ts, vs = rows[:, 0], rows[:, 1]
    tot, n = 0.0, 0
    for s, e in wins:
        a, b = np.searchsorted(ts, s), np.searchsorted(ts, e)
        tot += vs[a:b].sum(); n += b - a
    return tot / n if n else None


have_k = c.execute("select count(*) from sqlite_master where name='CUPTI_ACTIVITY_KIND_KERNEL'").fetchone()[0] > 0
if have_k:
    K = np.array(c.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall(), dtype=object)
    ks = K[:, 0].astype(np.int64) if len(K) else np.zeros(0, np.int64)
    ke = K[:, 1].astype(np.int64) if len(K) else np.zeros(0, np.int64)
    kn = [strings.get(int(x), "?") for x in K[:, 2]] if len(K) else []


def kcat(name):
    n = name.lower()
    if "fmha" in n or "flash" in n or "attention" in n:
        return "attention"
    if "gemm" in n or "cutlass" in n or "gemv" in n or "matmul" in n:
        return "gemm"
    if "softmax" in n:
        return "softmax"
    if "topk" in n or "mbtopk" in n or "sort" in n or "radix" in n:
        return "topk/sort"
    if "reduce" in n:
        return "reduce"
    if "copy" in n or "cat" in n or "index" in n or "gather" in n or "scatter" in n:
        return "copy/index"
    if "elementwise" in n:
        return "elementwise"
    return "other"


def kstats(wins):
    cnt, busy, cats = 0, 0, defaultdict(int)
    for s, e in wins:
        a, b = np.searchsorted(ks, s), np.searchsorted(ks, e)
        cur_s = cur_e = None
        for i in range(a, b):
            k0, k1 = int(ks[i]), min(int(ke[i]), e)
            cnt += 1
            cats[kcat(kn[i])] += k1 - k0
            if cur_e is None or k0 > cur_e:
                if cur_e is not None:
                    busy += cur_e - cur_s
                cur_s, cur_e = k0, k1
            else:
                cur_e = max(cur_e, k1)
        if cur_e is not None:
            busy += cur_e - cur_s
    return cnt, busy, cats


order = ["step", "tgt_prefix", "drf_prefix", "exact", "exact/approx", "exact/expand r1", "exact/expand r>=2",
         "dense_loss", "rollout", "rollout/draft", "rollout/verify", "bwd_head", "bwd_prefix"]
res = {}
step_ms = sum(e - s for s, e in groups["step"]) / 1e6 / n_steps
for name in order + sorted(set(groups) - set(order)):
    wins = groups.get(name)
    if not wins:
        continue
    ms = sum(e - s for s, e in wins) / 1e6 / n_steps
    r = {"ms_per_step": ms, "share": ms / step_ms, "calls_per_step": len(wins) / n_steps}
    for key in want.values():
        v = metric_avg(key, wins)
        if v is not None:
            r[key] = v / 1e6 if key == "gpc_mhz" and v > 1e5 else v
    if have_k:
        cnt, busy, cats = kstats(wins)
        wall = sum(e - s for s, e in wins)
        r["kernels_per_step"] = cnt / n_steps
        r["kernel_busy"] = busy / wall if wall else 0
        tot = sum(cats.values()) or 1
        r["kernel_time_by_cat"] = {k: v / tot for k, v in sorted(cats.items(), key=lambda x: -x[1])}
    res[name] = r
# exact minus its children: u lm_head + approx logits + CPU walk between rounds
ch = sum(res.get(k, {}).get("ms_per_step", 0) for k in ("exact/approx", "exact/expand r1", "exact/expand r>=2"))
if "exact" in res:
    res["exact/other (u, approx logits, walk)"] = {"ms_per_step": res["exact"]["ms_per_step"] - ch,
                                                  "share": (res["exact"]["ms_per_step"] - ch) / step_ms}
json.dump({"n_steps": n_steps, "stages": res}, open(out, "w"), indent=1)

cols = ["ms_per_step", "share", "gpu_active", "sm_active", "sm_issue", "tensor_active", "warps_in_flight",
        "dram_read", "dram_write", "gpc_mhz", "kernels_per_step", "kernel_busy"]
print(f"{n_steps} steps")
print(f"{'stage':38s}" + "".join(f"{x[:11]:>13s}" for x in cols))
for name, r in res.items():
    row = f"{name:38s}"
    for x in cols:
        v = r.get(x)
        if v is None:
            row += f"{'':>13s}"
        elif x in ("share", "kernel_busy"):
            row += f"{v:>12.1%} "
        elif x == "ms_per_step":
            row += f"{v:>13.1f}"
        else:
            row += f"{v:>13.1f}"
    print(row)
if have_k:
    for name in ("exact/expand r1", "exact/expand r>=2", "rollout", "bwd_head"):
        if name in res:
            print(name, {k: f"{v:.0%}" for k, v in list(res[name]["kernel_time_by_cat"].items())[:6]})
