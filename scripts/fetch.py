"""Stream ICP + PLETH (+ ECG II) windows from MIMIC-III matched waveform records.

One subject at a time, one 10-minute chunk at a time (wfdb byte-range reads via
pn_dir), so memory stays small. Up to MAX_HOURS per subject, taken as chunks
spread evenly over all segments that contain ICP + PLETH + II. Each chunk is
cut into 60 s windows; per subject we save data/raw/<subject>.npz with
ppg[n,7500], ecg[n,7500], icp[n,7500] (float32) and window start times.
Usage: python scripts/fetch.py [max_subjects] [max_hours]
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb

PN = "mimic3wdb/1.0/matched"
FS = 125
WIN = 60 * FS
CHUNK = 10 * 60 * FS
NEED = ["PLETH", "II", "ICP"]
OUT = Path("data/raw")
MIN_WINDOWS = 10  # a subject counts toward max_subjects only with >= 10 windows


def eligible_segments(rec):
    d, name = rec.rsplit("/", 1)
    pn = f"{PN}/{d}"
    master = wfdb.rdheader(name, pn_dir=pn)
    segs = []
    for seg, ln in zip(master.seg_name, master.seg_len):
        if seg == "~" or seg.endswith("_layout") or ln < WIN:
            continue
        h = wfdb.rdheader(seg, pn_dir=pn)
        if all(s in h.sig_name for s in NEED) and h.fs == FS:
            segs.append((seg, ln))
    return pn, segs


def fetch_subject(subject, recs, max_hours):
    segs = []
    for rec in recs:  # a subject can have several matched records (same folder)
        pn, sg = eligible_segments(rec)
        segs += sg
    chunks = [(s, a) for s, ln in segs for a in range(0, ln - CHUNK + 1, CHUNK)]
    if not chunks:
        # allow short segments: one chunk of whole windows
        chunks = [(s, 0) for s, ln in segs if ln >= WIN]
    n_take = int(max_hours * 6)
    if len(chunks) > n_take:
        idx = np.linspace(0, len(chunks) - 1, n_take).round().astype(int)
        chunks = [chunks[i] for i in idx]
    seglen = dict(segs)
    ppg, ecg, icp, meta = [], [], [], []
    for seg, a in chunks:
        b = min(a + CHUNK, seglen[seg])
        try:
            r = wfdb.rdrecord(seg, pn_dir=pn, sampfrom=a, sampto=b, channel_names=NEED)
        except Exception as e:  # network or format error: skip chunk, log it
            print("  skip", seg, a, e, flush=True)
            continue
        x = r.p_signal.astype(np.float32)
        order = [r.sig_name.index(s) for s in NEED]
        x = x[:, order]
        for w in range(0, x.shape[0] - WIN + 1, WIN):
            win = x[w:w + WIN]
            ppg.append(win[:, 0]); ecg.append(win[:, 1]); icp.append(win[:, 2])
            meta.append((seg, a + w))
    if not ppg:
        return None
    np.savez_compressed(OUT / f"{subject}.npz", ppg=np.stack(ppg), ecg=np.stack(ecg),
                        icp=np.stack(icp), seg=np.array([m[0] for m in meta]),
                        start=np.array([m[1] for m in meta]))
    return len(ppg), len(segs)


def main():
    max_subj = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    max_hours = float(sys.argv[2]) if len(sys.argv) > 2 else 2
    OUT.mkdir(parents=True, exist_ok=True)
    cand = pd.read_csv("results/candidates.csv")
    cand = cand[cand.has_ecg_ii].groupby("subject", sort=True)["record"].apply(list)
    log = []
    for subject, rec in cand.items():
        if sum(1 for l in log if l["windows"] >= MIN_WINDOWS) >= max_subj:
            break
        t0 = time.time()
        if (OUT / f"{subject}.npz").exists():
            n = len(np.load(OUT / f"{subject}.npz")["start"]); res = (n, -1)
        else:
            try:
                res = fetch_subject(subject, rec, max_hours)
            except Exception as e:
                print(subject, "failed", e, flush=True); res = None
        n, nseg = res if res else (0, 0)
        log.append({"subject": subject, "records": " ".join(rec), "eligible_segments": nseg, "windows": n})
        print(f"{subject} {rec} windows={n} segs={nseg} {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(log).to_csv("results/fetch_log.csv", index=False)


if __name__ == "__main__":
    main()
