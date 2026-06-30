#!/usr/bin/env python3
"""Check RunPod GPU availability in North America via the public GraphQL API.

Lists GPU types that are IN STOCK in any US/Canada data center at or below a
price cap and VRAM floor. No API key required (public pricing endpoint).

  python3 runpod_avail.py --max-price 0.30 --min-vram 24
"""
import argparse, json, sys, urllib.request

GQL = "https://api.runpod.io/graphql"
NA_DCS = [
    "CA-MTL-1", "CA-MTL-2", "CA-MTL-3", "CA-MTL-4",
    "US-CA-1", "US-CA-2", "US-DE-1", "US-GA-1", "US-GA-2", "US-IL-1",
    "US-KS-1", "US-KS-2", "US-KS-3", "US-MD-1", "US-MO-1", "US-MO-2",
    "US-NC-1", "US-NC-2", "US-NE-1", "US-OR-1", "US-OR-2", "US-PA-1",
    "US-TX-1", "US-TX-2", "US-TX-3", "US-TX-4", "US-TX-5", "US-TX-6", "US-WA-1",
]
STOCK_RANK = {"High": 3, "Medium": 2, "Low": 1}


def query(dcs):
    aliases = []
    for dc in dcs:
        a = "dc_" + dc.replace("-", "_")
        aliases.append(
            '%s: gpuTypes { id displayName memoryInGb communityCloud secureCloud '
            'lowestPrice(input:{gpuCount:1, dataCenterId:"%s"}) '
            '{ uninterruptablePrice minimumBidPrice stockStatus } }' % (a, dc)
        )
    q = "query {\n" + "\n".join(aliases) + "\n}"
    req = urllib.request.Request(
        GQL, data=json.dumps({"query": q}).encode(),
        headers={"Content-Type": "application/json",
                 "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                               "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["data"]


def scan(max_price, min_vram):
    data = query(NA_DCS)
    # gpu id -> {vram, cloud, dcs: [(dc, price, stock)]}
    found = {}
    for dc in NA_DCS:
        a = "dc_" + dc.replace("-", "_")
        for g in data.get(a) or []:
            lp = g.get("lowestPrice") or {}
            stock = lp.get("stockStatus")
            price = lp.get("uninterruptablePrice")
            vram = g.get("memoryInGb") or 0
            if not stock or price is None:
                continue
            if price > max_price or vram < min_vram:
                continue
            e = found.setdefault(g["id"], {
                "name": g["displayName"], "vram": vram,
                "community": g.get("communityCloud"), "dcs": []})
            e["dcs"].append((dc, price, stock))
    return found


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-price", type=float, default=0.30)
    p.add_argument("--min-vram", type=int, default=24)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    found = scan(args.max_price, args.min_vram)
    if args.json:
        print(json.dumps(found))
        return 0 if found else 2
    if not found:
        print("NONE available: no NA GPU >= %dGB VRAM at <= $%.2f/hr in stock."
              % (args.min_vram, args.max_price))
        return 2
    print("AVAILABLE in North America (>=%dGB, <=$%.2f/hr on-demand):"
          % (args.min_vram, args.max_price))
    for gid, e in sorted(found.items(), key=lambda kv: min(d[1] for d in kv[1]["dcs"])):
        best = sorted(e["dcs"], key=lambda d: (d[1], -STOCK_RANK.get(d[2], 0)))
        cloud = "community" if e["community"] else "secure"
        locs = ", ".join("%s $%.3f/%s" % (dc, pr, st) for dc, pr, st in best[:6])
        print("  %-16s %3dGB [%s]  %s" % (e["name"], e["vram"], cloud, locs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
