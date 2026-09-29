"""Download The Week in Chess game files into data/twic (zips are kept, not unpacked).

Issue 1038 (September 2014) to 1664 (September 2026).
"""
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "twic")
FIRST, LAST = 1038, 1664
# the site answers 406 to Python's default user agent


def fetch(n):
    path = os.path.join(OUT, f"twic{n}g.zip")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return None
    try:
        req = urllib.request.Request(f"https://theweekinchess.com/zips/twic{n}g.zip",
                                     headers={"User-Agent": "fide-rating-research/1.0"})
        data = urllib.request.urlopen(req, timeout=120).read()
    except Exception:
        return n
    with open(path + ".part", "wb") as f:
        f.write(data)
    os.replace(path + ".part", path)
    return None


def main():
    os.makedirs(OUT, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:  # a few at a time, to stay polite
        missing = [n for n in pool.map(fetch, range(FIRST, LAST + 1)) if n]
    print(f"issues {FIRST}-{LAST}, missing: {missing or 'none'}")


if __name__ == "__main__":
    main()
