"""Replay a CUDA allocator snapshot's trace and report what is live at the peak.

python scripts/peak_from_trace.py memprof/<tag>_snapshot.pickle [--top 30]
"""
import collections
import pickle
import sys

GiB = 2**30


def site_of(frames):
    hits = [fr for fr in frames or [] if "antstorch" in fr.get("filename", "")]
    return " <- ".join(f"{fr['filename'].split('antstorch/')[-1]}:{fr.get('line')} {fr.get('name')}"
                       for fr in hits[:2]) or "(no antstorch frame: autograd engine / C++)"


def main():
    path = sys.argv[1]
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 30
    snap = pickle.load(open(path, "rb"))
    dev = int(sys.argv[sys.argv.index("--device") + 1]) if "--device" in sys.argv else 0
    trace = snap["device_traces"][dev]
    live, cur, best, best_i = {}, 0, 0, -1
    best_live = {}
    stop_at = sys.argv[sys.argv.index("--before") + 1] if "--before" in sys.argv else None
    for i, ev in enumerate(trace):
        act = ev["action"]
        if stop_at and act == "alloc" and any(stop_at in (fr.get("name") or "") for fr in ev.get("frames") or []):
            print(f"(stopping at first allocation inside {stop_at!r}, event {i})")
            break
        if act == "alloc":
            live[ev["addr"]] = (ev["size"], ev.get("frames"))
            cur += ev["size"]
            if cur > best:
                best, best_i = cur, i
                best_live = dict(live)
        elif act in ("free_requested", "free_completed"):
            if act == "free_requested" and ev["addr"] in live:
                cur -= live.pop(ev["addr"])[0]
    print(f"peak live {best / GiB:.1f} GiB at event {best_i}/{len(trace)} "
          f"(trace covers the last {len(trace)} events)")
    by_site = collections.Counter()
    count = collections.Counter()
    for size, frames in best_live.values():
        s = site_of(frames)
        by_site[s] += size
        count[s] += 1
    for s, size in by_site.most_common(top):
        print(f"  {size / GiB:7.2f} GiB  x{count[s]:<3d} {s}")
    # what was being allocated at the peak
    ev = trace[best_i]
    print("allocation that set the peak:", f"{ev['size'] / GiB:.2f} GiB", site_of(ev.get("frames")))


if __name__ == "__main__":
    main()
