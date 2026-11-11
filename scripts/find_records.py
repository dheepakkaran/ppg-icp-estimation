"""Scan the MIMIC-III Waveform DB *matched* subset for records with ICP + PLETH.

The matched subset names records by subject (pXXXXXX), which is what
leave-one-patient-out needs. For each record: read the master header (line 2 =
layout name), then the small layout header listing all signals. Writes
results/candidates.csv.
Usage: python scripts/find_records.py [max_subjects] [max_minutes] [offset] [out_csv]
"""
import csv, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = "https://physionet.org/files/mimic3wdb/1.0/"
BASE = ROOT + "matched/"
MAX_HITS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
MAX_MIN = float(sys.argv[2]) if len(sys.argv) > 2 else 9
OFFSET = int(sys.argv[3]) if len(sys.argv) > 3 else 0  # skip records already scanned
OUTCSV = sys.argv[4] if len(sys.argv) > 4 else "results/candidates.csv"


def get(url, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                return r.read().decode()
        except Exception:
            time.sleep(1 + i)
    return None


def signals(rec):
    # rec like "p00/p000020/p000020-2183-04-28-17-47"
    d = rec.rsplit("/", 1)[0] + "/"
    master = get(f"{BASE}{rec}.hea")
    if not master or len(master.splitlines()) < 2:
        return rec, None
    layout = master.splitlines()[1].split()[0]
    if not layout.endswith("_layout"):
        return rec, None
    txt = get(f"{BASE}{d}{layout}.hea")
    if not txt:
        return rec, None
    sigs = [ln.split()[-1] for ln in txt.splitlines()[1:] if ln.strip() and not ln.startswith("#")]
    return rec, sigs


def main():
    recs = get(BASE + "RECORDS-waveforms").split()[OFFSET:]
    t0 = time.time()
    hits, scanned = [], 0
    with ThreadPoolExecutor(48) as ex:
        futs = [ex.submit(signals, r) for r in recs]
        for f in as_completed(futs):
            rec, sigs = f.result()
            scanned += 1
            if sigs and "ICP" in sigs and "PLETH" in sigs:
                hits.append((rec.split("/")[1], rec, "II" in sigs, "ABP" in sigs, " ".join(sigs)))
                print("HIT", rec, sigs, flush=True)
            if scanned % 2000 == 0:
                print(f"scanned {scanned} hits {len(hits)} {time.time()-t0:.0f}s", flush=True)
            if len({h[0] for h in hits if h[2]}) >= MAX_HITS or time.time() - t0 > MAX_MIN * 60:
                for g in futs:
                    g.cancel()
                break
    with open(OUTCSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "record", "has_ecg_ii", "has_abp", "signals"])
        w.writerows(sorted(hits))
    print(f"done scanned={scanned} of {len(recs)} hits={len(hits)} subjects={len({h[0] for h in hits})} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
