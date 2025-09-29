# -*- coding: utf-8 -*-
"""
Lee los .txt de search_results/ (producidos por get_imgs_geo_gps_search_python3.py)
y descarga las imágenes a data/images/, generando un images_manifest.csv.

- Sin dependencias externas (solo stdlib)
- Reintentos + fallback de URLs (original y tamaños estándar)
"""

import os, csv, time, re
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# --- Config por defecto (puedes cambiar rutas si quieres)
ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "search_results"          # donde están los .txt
IMAGES_DIR = ROOT / "data" / "images"      # donde guardaremos los .jpg
MANIFEST_CSV = ROOT / "data" / "images_manifest.csv"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)

# --- Helpers

def safe_write_header_if_needed(csv_path, fieldnames):
    exists = csv_path.exists()
    f = open(csv_path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=fieldnames)
    if not exists:
        w.writeheader()
    return f, w

def url_ok(url, tries=2, timeout=20):
    last_exc = None
    for _ in range(tries):
        try:
            req = Request(url, headers={"User-Agent": "img-downloader-py3"})
            with urlopen(req, timeout=timeout) as r:
                # Aceptamos 200 y devolvemos contenido
                return r.read()
        except Exception as e:
            last_exc = e
            time.sleep(0.3)
    raise last_exc

def build_url_candidates(rec):
    """
    Preferimos el 'original' si tenemos originalsecret+originalformat.
    Si no, usamos live.staticflickr.com/{server}/{id}_{secret}.jpg
    y probamos algunos sufijos de tamaño como fallback.
    """
    server = rec.get("server", "").strip()
    pid    = rec.get("id", "").strip()
    secret = rec.get("secret", "").strip()
    osec   = rec.get("originalsecret", "").strip()
    ofmt   = rec.get("originalformat", "").strip()

    candidates = []

    # Original (si disponible)
    if osec and ofmt:
        candidates.append(f"https://live.staticflickr.com/{server}/{pid}_{osec}_o.{ofmt}")

    # Tamaño “base” sin sufijo
    if server and pid and secret:
        base = f"https://live.staticflickr.com/{server}/{pid}_{secret}"
        candidates.append(base + ".jpg")
        # Fallbacks populares de tamaños
        for suf in ["_b", "_c", "_z", "_n", "_w", "_m"]:
            candidates.append(base + f"{suf}.jpg")

    # Devuelve lista sin duplicados preservando orden
    dedup = []
    seen = set()
    for u in candidates:
        if u not in seen:
            dedup.append(u)
            seen.add(u)
    return dedup

def download_image(urls, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    for u in urls:
        try:
            data = url_ok(u)
            with open(dst, "wb") as f:
                f.write(data)
            return u, "ok"
        except HTTPError as e:
            status = f"http_{e.code}"
        except URLError as e:
            status = "url_error"
        except Exception:
            status = "error"
        # intenta siguiente candidato
        last_status = status
        time.sleep(0.2)
    return "", last_status

def subdir_for(pid: str) -> Path:
    a = pid[:2] if len(pid) >= 2 else "xx"
    b = pid[2:4] if len(pid) >= 4 else "yy"
    return IMAGES_DIR / a / b

# --- Parser de los .txt (estilo legacy)

def parse_results_txt(path: Path):
    """
    Devuelve una lista de dicts (un dict por foto).
    Espera bloques con líneas como:
      photo: <id> <secret> <server>
      owner: ...
      title: ...
      originalsecret: ...
      originalformat: ...
      latitude: ...
      longitude: ...
      ...
    Separados por líneas en blanco o fin de archivo.
    """
    records = []
    current = None

    def push():
        nonlocal current
        if current and "id" in current:
            records.append(current)
        current = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                # separador de bloque
                push()
                continue

            if line.startswith("photo:"):
                # nuevo bloque
                push()
                # photo: id secret server
                parts = line.split()
                # parts[0] = 'photo:'
                if len(parts) >= 4:
                    current = {
                        "id": parts[1],
                        "secret": parts[2],
                        "server": parts[3],
                        "query": path.stem,
                    }
                else:
                    current = {"query": path.stem}
                continue

            if current is None:
                current = {"query": path.stem}

            # Clave: valor
            if ":" in line:
                k, v = line.split(":", 1)
                current[k.strip().lower()] = v.strip()

        # último bloque
        push()

    return records

# --- Main

def main():
    fieldnames = ["query","id","server","secret","originalsecret","originalformat",
                  "latitude","longitude","chosen_url","path","status"]
    manifest_f, manifest_w = safe_write_header_if_needed(MANIFEST_CSV, fieldnames)

    txt_files = sorted(SRC_DIR.glob("*.txt"))
    if not txt_files:
        print(f"No se encontraron .txt en {SRC_DIR}")
        return

    total = 0
    ok = 0
    fail = 0

    for txt in txt_files:
        print(f"\nProcesando: {txt.name}")
        records = parse_results_txt(txt)
        print(f"  {len(records)} fotos en metadatos")

        for rec in records:
            pid = rec.get("id")
            if not pid:
                continue
            dst = subdir_for(pid) / f"{pid}.jpg"

            # si ya existe, lo registramos como cacheado
            if dst.exists() and dst.stat().st_size > 0:
                manifest_w.writerow({
                    "query": rec.get("query",""),
                    "id": pid,
                    "server": rec.get("server",""),
                    "secret": rec.get("secret",""),
                    "originalsecret": rec.get("originalsecret",""),
                    "originalformat": rec.get("originalformat",""),
                    "latitude": rec.get("latitude",""),
                    "longitude": rec.get("longitude",""),
                    "chosen_url": "",  # ya existía
                    "path": str(dst),
                    "status": "ok_cached"
                })
                continue

            candidates = build_url_candidates(rec)
            url_used, status = download_image(candidates, dst)
            if status.startswith("ok"):
                ok += 1
            else:
                fail += 1
                # borra archivo parcial si quedó corrupto
                if dst.exists() and dst.stat().st_size == 0:
                    try: dst.unlink()
                    except: pass

            manifest_w.writerow({
                "query": rec.get("query",""),
                "id": pid,
                "server": rec.get("server",""),
                "secret": rec.get("secret",""),
                "originalsecret": rec.get("originalsecret",""),
                "originalformat": rec.get("originalformat",""),
                "latitude": rec.get("latitude",""),
                "longitude": rec.get("longitude",""),
                "chosen_url": url_used,
                "path": str(dst if status.startswith("ok") else ""),
                "status": status
            })

            total += 1
            # Respeta un poco el rate
            time.sleep(0.15)

    manifest_f.close()
    print(f"\nListo. Descargadas OK: {ok}, fallidas: {fail}, total procesadas: {total}")
    print(f"Imágenes en: {IMAGES_DIR}")
    print(f"Manifest CSV: {MANIFEST_CSV}")

if __name__ == "__main__":
    main()
