"""Build data/games.csv from TWIC PGN files and the FIDE player list.

Keeps classical over-the-board games where both players have a FIDE ID and a
published rating. Rapid, blitz and online events are dropped by name and by
TWIC's EventType tag, since they belong to separate FIDE lists.
"""
import csv
import glob
import io
import os
import re
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TWIC = os.path.join(ROOT, "data", "twic")
FIDE_LIST = os.path.join(ROOT, "data", "fide", "players_list_foa.txt")
OUT = os.path.join(ROOT, "data", "games.csv")

FAST = re.compile(
    r"rapid|blitz|bullet|armageddon|titled|online|960|freestyle|chess\.com|"
    r"lichess|speed|\d+ ?[-+|] ?\d+ (mon|tue|wed|thu|fri|sat|sun)|arena|"
    r"pro chess|champions chess tour|cct|esports|playoff|tiebreak|tie-break|"
    r"fischer random|bughouse|puzzle|\bGCL\b|global chess league|\bBl\b|\bRap\b|9LX",
    re.I,
)
TAG = re.compile(r'^\[(\w+) "(.*)"\]')
SCORES = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


def load_birth_years():
    years = {}
    with open(FIDE_LIST, encoding="latin-1") as f:
        header = f.readline()
        b0 = header.index("B-day")
        for line in f:
            fid = line[:15].strip()
            by = line[b0:b0 + 4].strip()
            if fid.isdigit() and by.isdigit() and int(by) > 1900:
                years[fid] = int(by)
    return years


def games():
    for path in sorted(glob.glob(os.path.join(TWIC, "twic*g.zip"))):
        tags = {}
        z = zipfile.ZipFile(path)
        name = next(n for n in z.namelist() if n.lower().endswith(".pgn"))
        with io.TextIOWrapper(z.open(name), encoding="latin-1") as f:
            for line in f:
                m = TAG.match(line)
                if m:
                    if m.group(1) == "Event" and tags:
                        yield tags
                        tags = {}
                    tags[m.group(1)] = m.group(2)
        if tags:
            yield tags


def main():
    years = load_birth_years()
    seen = set()
    kept = dropped_fast = dropped_missing = dupes = 0
    with open(OUT, "w", newline="") as out:
        w = csv.writer(out)
        w.writerow(["date", "white", "black", "white_elo", "black_elo",
                    "score", "white_by", "black_by", "event"])
        for g in games():
            ev = g.get("Event", "")
            if FAST.search(ev) or FAST.search(g.get("EventType", "")):
                dropped_fast += 1
                continue
            try:
                wid, bid = g["WhiteFideId"], g["BlackFideId"]
                we, be = int(g["WhiteElo"]), int(g["BlackElo"])
                s = SCORES[g["Result"]]
                d = g["Date"].replace(".", "-")
                int(d[:4]), int(d[5:7])
            except (KeyError, ValueError):
                dropped_missing += 1
                continue
            if not (wid.isdigit() and bid.isdigit()) or wid == bid or "?" in d[:7]:
                dropped_missing += 1
                continue
            key = (d, wid, bid, g.get("Round", ""), ev)
            if key in seen:  # TWIC sometimes repeats late-arriving games
                dupes += 1
                continue
            seen.add(key)
            d = d.replace("-??", "-15") if d[8:] == "??" else d
            w.writerow([d, wid, bid, we, be, s,
                        years.get(wid, ""), years.get(bid, ""), ev])
            kept += 1
    print(f"kept {kept}, fast {dropped_fast}, missing tags {dropped_missing}, "
          f"duplicates {dupes}")


if __name__ == "__main__":
    main()
