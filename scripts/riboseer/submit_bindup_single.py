#!/usr/bin/env python3
"""Submit single-chain PDBs to BindUP's single-protein mode (file upload).

The 22 PDBs that failed BindUP's batch (PDB-ID) mode are large/new mmCIF-only
structures that BindUP could not fetch+convert. The fix: upload our own
single-chain PDBs (``graphbind_inputs/<sample_id>.pdb`` — one chain renamed to
``A``, legacy-PDB compatible) through the single-protein form.

Form (discovered at https://bindup.technion.ac.il/, no CAPTCHA):
    POST multipart -> https://bindup.technion.ac.il/cgi-bin/BindUP/BindUP.cgi
    model_type=experimental  input_type=file  PDB_file=<upload>
    chain_type=selected_chain  chain_id=A
    is_pos_patch=yes  patch_num=3        (positive patches x3, no negative)
    [is_email=yes email=...]             (optional)

For each sample it POSTs the file, saves the response HTML, follows the result
link, and downloads ``*_patch_list.txt`` / ``*_patch.cif`` into
``<results-dir>/<sample_id>/``.

Header rewrite (so the existing parser works unchanged)
-------------------------------------------------------
A single-upload result is reported as ``Chain A`` with ``PDB ID:`` set to the
uploaded file's name — neither matches the sample's real ``(source_pdb, chain)``.
Because the uploaded PDB preserved the original author residue numbering, the
only thing wrong for ``bindup_parse.py`` is the header. With
``--rewrite-header`` (default on) we rewrite ``PDB ID: ...`` -> the real
``source_pdb`` and ``Chain A`` -> the real ``orig_chain`` (from manifest.csv), so
the downloaded ``<sample_id>_patch_list.txt`` parses drop-in alongside the batch
results (author->label_seq map then aligns it to GT).

Usage
-----
::

    # validate without sending
    python scripts/riboseer/submit_bindup_single.py --dry-run --limit 1

    # submit one sample (test)
    python scripts/riboseer/submit_bindup_single.py --sample 8k22_C_P

    # submit all 35 (5s apart)
    python scripts/riboseer/submit_bindup_single.py --sleep 5
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

try:
    import requests
except ImportError as _e:  # pragma: no cover
    requests = None  # type: ignore
    _REQUESTS_ERR = _e
else:
    _REQUESTS_ERR = None

logger = logging.getLogger("submit_bindup_single")

BASE_URL = "https://bindup.technion.ac.il/"
ACTION_URL = "https://bindup.technion.ac.il/cgi-bin/BindUP/BindUP.cgi"
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
_CAPTCHA_MARKERS = ("captcha", "recaptcha", "hcaptcha")


def make_session(no_proxy: bool = False, insecure: bool = False):
    if requests is None:
        raise RuntimeError("'requests' is required: {}".format(_REQUESTS_ERR))
    s = requests.Session()
    if no_proxy:
        s.trust_env = False
        s.proxies = {"http": None, "https": None}
    if insecure:
        # The Technion server omits an intermediate CA, so Python's certifi
        # can't build the chain (browsers/Windows fetch it via AIA). We only
        # exchange a public PDB upload + public results, so skipping verify is
        # acceptable here. Off by default; opt in with --insecure.
        s.verify = False
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:  # noqa: BLE001
            pass
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def load_manifest(path: Path) -> list:
    """Read manifest.csv -> [{sample_id, pdb_id, orig_chain, upload_file}]."""
    if not path.is_file():
        raise FileNotFoundError("manifest not found: {}".format(path))
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_fields(chain: str, patch_num: int, email: Optional[str]) -> dict:
    fields = {
        "model_type": "experimental",
        "input_type": "file",
        "PDB_ID": "",
        "chain_type": "selected_chain",
        "chain_id": chain,
        "is_pos_patch": "yes",
        "patch_num": str(patch_num),
    }
    if email:
        fields["is_email"] = "yes"
        fields["email"] = email
    return fields


def job_urls(resp_url: str, pdb_id: str, sample_id: str) -> dict:
    """From the post-redirect results URL ``.../<jobid>/results.html`` derive the
    result-file URLs. Single mode names the patch list ``<pdb_id>_patch_list.txt``
    and the bundle ``<sample_id>_BindUP-Alpha_Results.zip``."""
    base = resp_url.rsplit("/", 1)[0] if resp_url.endswith((".html", ".cgi")) else resp_url.rstrip("/")
    return {
        "base": base,
        "patch_list": "{}/{}_patch_list.txt".format(base, pdb_id),
        "patch_cif": "{}/{}_patch.cif".format(base, pdb_id),
        "zip": "{}/{}_BindUP-Alpha_Results.zip".format(base, sample_id),
        "results_page": resp_url,
    }


def _looks_ready(text: str) -> bool:
    return bool(text) and ("Patch 1" in text or "Largest Positive" in text)


def poll_patch_list(session, url: str, interval: float, timeout: float):
    """GET ``url`` every ``interval`` s until it returns a finished patch list
    (or ``timeout`` s elapse). Returns the text, or None on timeout.

    The job is async ("calculating ..."); the patch_list.txt 404s / is absent
    until done, then appears with the ``Patch 1:`` content — the cleanest
    completion signal. No cookies/session token are needed (the job is keyed by
    the redirect URL's job id)."""
    waited = 0.0
    while waited <= timeout:
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 200 and _looks_ready(r.text):
                return r.text
        except Exception as e:  # noqa: BLE001
            logger.debug("poll error on %s: %s", url, e)
        time.sleep(interval)
        waited += interval
    return None


def rewrite_header(text: str, source_pdb: str, orig_chain: str) -> str:
    """Rewrite a single-upload patch_list so ``bindup_parse.py`` keys it
    to the real ``(source_pdb, orig_chain)``.

    Single mode writes ``PDB file: <upload>`` (not batch's ``PDB ID: <pdb>``)
    and ``Chain A`` (the renamed chain). We normalise the header line to
    ``PDB ID: <source_pdb>`` (what the parser's regex expects) and the chain
    line to ``Chain <orig_chain>``. Residue numbers are left untouched — they're
    the original author numbering preserved in the uploaded single-chain PDB, so
    the parser's author->label_seq map (built from the real structure's
    orig_chain) aligns them to GT exactly as for the batch results.
    """
    text = re.sub(r"(?im)^[ \t]*PDB\s*(?:ID|file)\s*[:\s].*$",
                  "PDB ID: " + source_pdb, text)
    text = re.sub(r"(?im)^[ \t]*Chain[ \t]+\S+[ \t]*$",
                  "Chain " + orig_chain, text)
    return text


def submit_one(session, url: str, pdb_path: Path, fields: dict) -> tuple:
    """POST one file. Returns (status_code, response_text, response_url)."""
    files = {"PDB_file": (pdb_path.name, pdb_path.read_bytes(), "chemical/x-pdb")}
    data = dict(fields)
    resp = session.post(url, data=data, files=files, timeout=180)
    return resp.status_code, resp.text, resp.url


def process_sample(session, args, row: dict) -> dict:
    sid = row["sample_id"]
    res = {"sample_id": sid, "status": "ok", "error": None}
    pdb = args.uploads_dir / row.get("upload_file", sid + ".pdb")
    if not pdb.is_file():
        res.update(status="failed", error="upload PDB missing: {}".format(pdb))
        return res

    out_pl = args.results_dir / sid / "{}_patch_list.txt".format(sid)
    if args.resume and out_pl.is_file() and out_pl.stat().st_size > 0:
        res["status"] = "skipped"
        return res

    fields = build_fields(args.chain, args.patch_num, args.email)
    if args.dry_run:
        logger.info("[dry-run] POST %s  file=%s  chain=%s patch_num=%s",
                    args.url, pdb.name, args.chain, args.patch_num)
        return res

    out_dir = args.results_dir / sid
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        code, text, resp_url = submit_one(session, args.url, pdb, fields)
    except Exception as e:  # noqa: BLE001
        res.update(status="failed", error="POST error: {}".format(e))
        return res

    (out_dir / "submit_response.html").write_text(text, encoding="utf-8", errors="replace")
    if any(m in text.lower() for m in _CAPTCHA_MARKERS):
        res.update(status="failed", error="CAPTCHA in response")
        return res

    urls = job_urls(resp_url, row["pdb_id"], sid)
    res["job_url"] = urls["results_page"]
    # Async job: poll the patch_list until it materialises.
    content = poll_patch_list(session, urls["patch_list"],
                              args.poll_interval, args.poll_timeout)
    if content is None:
        res.update(status="submitted_no_file",
                   error="patch_list not ready within {}s ({})".format(
                       args.poll_timeout, urls["patch_list"]))
        return res

    got = []
    if args.rewrite_header:
        content = rewrite_header(content, row["pdb_id"], row["orig_chain"])
    (out_dir / "{}_patch_list.txt".format(sid)).write_text(content, encoding="utf-8")
    got.append("patch_list")
    # best-effort bundle / cif
    for key, suffix in (("zip", ".zip"), ("patch_cif", "_patch.cif")):
        try:
            rb = session.get(urls[key], timeout=120)
            if rb.status_code == 200 and rb.content:
                (out_dir / (sid + suffix)).write_bytes(rb.content)
                got.append(key)
        except Exception:  # noqa: BLE001
            pass
    res["downloaded"] = got
    return res


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--uploads-dir", type=Path,
                   default=Path("data/batch_test_v7/bindup_single_uploads"))
    p.add_argument("--manifest", type=Path, default=None,
                   help="default <uploads-dir>/manifest.csv")
    p.add_argument("--results-dir", type=Path,
                   default=Path("data/batch_test_v7/bindup_results"))
    p.add_argument("--url", default=ACTION_URL)
    p.add_argument("--chain", default="A", help="chain to analyse (uploads are 'A')")
    p.add_argument("--patch-num", type=int, default=3, choices=[1, 2, 3])
    p.add_argument("--email", default=None, help="optional notification email")
    p.add_argument("--sample", default=None, help="only submit this sample_id")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--sleep", type=float, default=5.0,
                   help="seconds between submissions")
    p.add_argument("--poll-interval", type=float, default=5.0,
                   help="seconds between result polls")
    p.add_argument("--poll-timeout", type=float, default=300.0,
                   help="max seconds to wait for one job's patch_list")
    p.add_argument("--no-rewrite-header", dest="rewrite_header",
                   action="store_false",
                   help="keep BindUP's 'Chain A'/filename header as-is")
    p.add_argument("--resume", action="store_true",
                   help="skip samples whose <sid>_patch_list.txt already exists")
    p.add_argument("--no-proxy", action="store_true")
    p.add_argument("--insecure", action="store_true",
                   help="skip TLS verification (Technion server omits an "
                        "intermediate CA that Python's certifi can't resolve)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    args.uploads_dir = args.uploads_dir.expanduser().resolve()
    args.results_dir = args.results_dir.expanduser().resolve()
    manifest_path = args.manifest or (args.uploads_dir / "manifest.csv")

    rows = load_manifest(manifest_path)
    if args.sample:
        rows = [r for r in rows if r["sample_id"] == args.sample]
        if not rows:
            logger.error("sample %s not in manifest", args.sample)
            return 1
    else:
        end = args.end if args.end is not None else len(rows)
        rows = rows[args.start:end]
        if args.limit:
            rows = rows[:args.limit]

    session = None if args.dry_run else make_session(args.no_proxy, args.insecure)

    n_ok = n_fail = n_partial = n_skip = 0
    total = len(rows)
    for i, row in enumerate(rows, 1):
        res = process_sample(session, args, row)
        st = res["status"]
        if st == "skipped":
            n_skip += 1
            logger.info("[%d/%d] %s skipped (already have patch_list)",
                        i, total, res["sample_id"])
            continue
        if st == "ok":
            n_ok += 1
            logger.info("[%d/%d] %s OK downloaded=%s", i, total, res["sample_id"],
                        res.get("downloaded"))
        elif st == "submitted_no_file":
            n_partial += 1
            logger.warning("[%d/%d] %s submitted but no result file auto-grabbed "
                           "(see submit_response.html; urls=%s)",
                           i, total, res["sample_id"], res.get("result_urls"))
        else:
            n_fail += 1
            logger.error("[%d/%d] %s FAILED: %s", i, total, res["sample_id"],
                         res.get("error"))
        if not args.dry_run and i < total and args.sleep:
            time.sleep(args.sleep)

    logger.info("done: %d ok, %d skipped, %d partial, %d failed (of %d)",
                n_ok, n_skip, n_partial, n_fail, total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
