"""Download FIDE monthly standard lists and keep the rows the analysis needs.

For every player in data/games.csv, a row is kept when:
  - the player had rated classical games in that list (activity and game counts),
  - the next list has games, so this list's K-factor is the one those games used,
  - the player is rated 2700 or above (top-player tables and charts),
  - it is the latest list (current flags and ratings).

Writes data/fide/monthly.csv: list month, FIDE ID, rating, games rated in that
list, K-factor, flag ("i" = inactive). Zips are streamed and discarded.
"""
import csv
import io
import os
import re
import urllib.request
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "fide", "monthly.csv")
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
FIRST, LAST = (2012, 9), (2026, 9)  # monthly lists begin in 2012


def game_ids():
    ids = set()
    with open(os.path.join(ROOT, "data", "games.csv")) as f:
        for r in csv.DictReader(f):
            ids.add(r["white"])
            ids.add(r["black"])
    return ids


def months():
    y, m = FIRST
    while (y, m) <= LAST:
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def read_list(y, m, ids):
    url = f"https://ratings.fide.com/download/standard_{MONTHS[m - 1]}{y % 100:02d}frl.zip"
    raw = urllib.request.urlopen(url, timeout=180).read()
    z = zipfile.ZipFile(io.BytesIO(raw))
    rows = {}
    with z.open(z.namelist()[0]) as f:
        text = io.TextIOWrapper(f, encoding="latin-1")
        header = text.readline()
        col = re.search(r"[A-Z]{3}\d{2}", header[60:], re.I).start() + 60
        g0 = header.index("Gms", col)
        k0 = header.index(" K", g0) + 1
        b0 = header.index("B-day")
        f0 = header.index("Flag")
        for line in text:
            fid = line[:15].strip()
            if fid not in ids:
                continue
            rating = line[col:g0].strip()
            if rating.isdigit():
                games = line[g0:k0].strip()
                rows[fid] = (rating, int(games) if games.isdigit() else 0,
                             line[k0:b0].strip(), line[f0:f0 + 4].strip())
    return rows


def main():
    ids = game_ids()
    print(f"{len(ids)} players to track")
    tmp = OUT + ".tmp"
    prev, prev_tag = {}, None
    with open(tmp, "w", newline="") as out:
        w = csv.writer(out)
        w.writerow(["month", "id", "rating", "games", "k", "flag"])
        for y, m in months():
            tag = f"{y}-{m:02d}"
            try:
                rows = read_list(y, m, ids)
            except Exception as e:
                print(f"{tag}: failed ({e})")
                continue
            last = (y, m) == LAST
            kept = 0
            for fid, (rating, games, k, flag) in rows.items():
                if games > 0 and fid in prev and prev_tag:
                    pr = prev[fid]
                    if not (pr[1] > 0 or int(pr[0]) >= 2700):  # not already written
                        w.writerow([prev_tag, fid, *pr])
                if games > 0 or int(rating) >= 2700 or last:
                    w.writerow([tag, fid, rating, games, k, flag])
                    kept += 1
            out.flush()
            prev, prev_tag = rows, tag
            print(f"{tag}: {len(rows)} tracked, {kept} kept", flush=True)
    os.replace(tmp, OUT)


if __name__ == "__main__":
    main()
