#!/usr/bin/env python3
"""Submit a PDB-ID list to BindUP's **batch** mode (no CAPTCHA).

BindUP's home page (https://bindup.technion.ac.il/) carries a second form,
``BindUP_batch_form``, that posts multipart/form-data to::

    https://bindup.technion.ac.il/cgi-bin/BindUP/BindUP_batch.cgi

fields (confirmed via --inspect):
    input_method = text            (paste IDs) | file (upload a list file)
    list         = <newline/space-separated PDB IDs>     (text mode)
    list_file    = <upload>                               (file mode)
    is_pos_patch = yes             (positive patches; default on)
    is_neg_patch = yes             (optional negative patches)
    patch_num    = 1 | 2 | 3       (how many patches per chain; default 3)
    is_email     = yes  + email    (notify by email when done)
    job_name     = <free text>

Results are returned on a job page (and by email when ``--email`` is set);
download them later with the same flow used for the test split
(``bindup_parse.py`` + ``bindup_sample_map.csv``).

TLS note: the Technion server omits an intermediate CA that Python's certifi
can't resolve (browsers fetch it via AIA). Use ``--insecure`` to skip
verification — we only exchange public PDB IDs + public results.

Modes
-----
* default   : POST the ``--ids`` list (optionally chunked by ``--batch-size``).
* --inspect : GET the form page and dump the batch form's fields.
* --dry-run : build the request(s) and print them, do not send.

Usage
-----
::

    # confirm the form first
    python scripts/riboseer/submit_bindup_batch.py --inspect --insecure

    # submit all training PDB IDs as one job
    python scripts/riboseer/submit_bindup_batch.py \
        --ids data/batch_train_v7/bindup_pdb_ids.txt \
        --email you@example.com \
        --job-name riboseer_train --insecure
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError as _e:  # pragma: no cover
    requests = None  # type: ignore
    _REQUESTS_ERR = _e
else:
    _REQUESTS_ERR = None

BASE_URL = "https://bindup.technion.ac.il/"
ACTION_URL = "https://bindup.technion.ac.il/cgi-bin/BindUP/BindUP_batch.cgi"
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
_CAPTCHA_MARKERS = ("captcha", "recaptcha", "hcaptcha", "are you human")


def make_session(no_proxy: bool = False, insecure: bool = False):
    if requests is None:
        raise RuntimeError("the 'requests' package is required: {}".format(_REQUESTS_ERR))
    s = requests.Session()
    if no_proxy:
        s.trust_env = False
        s.proxies = {"http": None, "https": None}
    if insecure:
        s.verify = False
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:  # noqa: BLE001
            pass
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def read_ids(path: Path) -> list[str]:
    """One PDB ID per line (blank / # comment lines skipped)."""
    if not path.is_file():
        raise FileNotFoundError("ids file not found: {}".format(path))
    out: list[str] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s.split()[0])
    return out


def chunk(items: list[str], size: int) -> list[list[str]]:
    if size <= 0:
        return [items]
    return [items[i:i + size] for i in range(0, len(items), size)]


def build_fields(ids: list[str], patch_num: int, pos: bool, neg: bool,
                 email: Optional[str], job_name: Optional[str]) -> dict:
    fields = {
        "input_method": "text",
        "list": "\n".join(ids),
        "patch_num": str(patch_num),
    }
    if pos:
        fields["is_pos_patch"] = "yes"
    if neg:
        fields["is_neg_patch"] = "yes"
    if email:
        fields["is_email"] = "yes"
        fields["email"] = email
    if job_name:
        fields["job_name"] = job_name
    return fields


def run_inspect(args) -> int:
    s = make_session(args.no_proxy, args.insecure)
    try:
        r = s.get(args.base_url, timeout=30)
    except Exception as e:  # noqa: BLE001
        print("ERROR: could not reach {} ({})".format(args.base_url, e),
              file=sys.stderr)
        return 3
    print("GET {} -> {} ({} bytes)\n".format(args.base_url, r.status_code, len(r.text)))
    # isolate the batch form block if possible
    m = re.search(r"(?is)<form[^>]*BindUP_batch[^>]*>.*?</form>", r.text)
    block = m.group(0) if m else r.text
    forms = re.findall(r"(?is)<form[^>]*>", r.text)
    for f in forms:
        print("FORM:", f.strip()[:180])
    print("\nbatch-form fields:")
    for fld in re.findall(r"(?is)<(?:input|textarea|select)[^>]*>", block):
        name = re.search(r"""name\s*=\s*['"]?([^'">\s]+)""", fld)
        typ = re.search(r"""type\s*=\s*['"]?([^'">\s]+)""", fld)
        print("  name={!s:14} type={!s:9} {}".format(
            name.group(1) if name else "-",
            typ.group(1) if typ else "-", fld.strip()[:80]))
    if any(mk in r.text.lower() for mk in _CAPTCHA_MARKERS):
        print("\n! CAPTCHA markers detected.", file=sys.stderr)
    return 0


def _post_one(args, ids: list[str], job_name: str, out: Path) -> int:
    fields = build_fields(ids, args.patch_num, not args.no_pos, args.neg,
                          args.email, job_name)
    if args.dry_run:
        print("[dry-run] POST {}".format(args.url))
        print("          input_method = text   list = <{} ids>".format(len(ids)))
        print("          patch_num = {}  is_pos_patch = {}  is_neg_patch = {}".format(
            args.patch_num, "yes" if not args.no_pos else "(off)",
            "yes" if args.neg else "(off)"))
        print("          email = {}  job_name = {}".format(
            args.email or "(none)", job_name or "(none)"))
        return 0

    s = make_session(args.no_proxy, args.insecure)
    files = {k: (None, v) for k, v in fields.items()}
    try:
        resp = s.post(args.url, files=files, timeout=args.timeout)
    except Exception as e:  # noqa: BLE001
        print("ERROR: POST failed ({})".format(e), file=sys.stderr)
        return 3
    out.write_text(resp.text, encoding="utf-8", errors="replace")
    print("POST {} -> {} ({} bytes); response -> {}".format(
        args.url, resp.status_code, len(resp.text), out))
    low = resp.text.lower()
    if any(mk in low for mk in _CAPTCHA_MARKERS):
        print("! CAPTCHA in response — submit likely NOT accepted; paste "
              "manually at {}".format(BASE_URL), file=sys.stderr)
        return 4

    # BindUP keys each job by a numeric id == the server's unix timestamp
    # at submission, and serves it at https://<host>/<jobid>/results.html.
    # That id is NOT in resp.url (no redirect — the calculating page comes
    # straight back from the CGI), so recover it from:
    #   1. the body — a "/BindUP/<jobid>/" path (present in the 500
    #      "mkdir ... failed" error page), else
    #   2. the response Date header (server unix time ≈ the job id).
    job_id = None
    m = re.search(r"/BindUP/(\d+)/", resp.text)
    if m:
        job_id = m.group(1)
    elif resp.headers.get("Date"):
        try:
            from email.utils import parsedate_to_datetime
            job_id = str(int(parsedate_to_datetime(
                resp.headers["Date"]).timestamp()))
        except Exception:  # noqa: BLE001
            job_id = None
    if job_id:
        results_url = "{}{}/results.html".format(BASE_URL, job_id)
        print("  job id (≈submit unix ts):", job_id)
        print("  results URL (poll when ready):", results_url)
        (out.with_name("bindup_job_info.txt")).write_text(
            "job_id={}\nresults_url={}\nhttp_status={}\njob_name={}\n".format(
                job_id, results_url, resp.status_code, job_name),
            encoding="utf-8")

    if "mkdir" in low and "failed" in low:
        print("! BindUP server error: it could not create the job dir "
              "(disk/permission issue on the Technion side). Retry later.",
              file=sys.stderr)
        return 5
    if resp.status_code == 200:
        print("  submitted {} PDB IDs ({}); results by email to {}".format(
            len(ids), job_name, args.email or "(no email set)"))
    return 0


def submit(args) -> int:
    try:
        ids = read_ids(args.ids)
    except FileNotFoundError as e:
        print("ERROR: {}".format(e), file=sys.stderr)
        return 2
    if not ids:
        print("ERROR: no PDB IDs in {}".format(args.ids), file=sys.stderr)
        return 2

    batches = chunk(ids, args.batch_size)
    out_dir = args.out_dir or args.ids.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base_job = args.job_name or args.ids.stem

    rc_all = 0
    for i, b in enumerate(batches, 1):
        job_name = base_job if len(batches) == 1 else "{}_b{}".format(base_job, i)
        out = out_dir / "bindup_batch_{}.response.html".format(
            "all" if len(batches) == 1 else "b{}".format(i))
        rc = _post_one(args, b, job_name, out)
        rc_all = rc_all or rc
    return rc_all


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inspect", action="store_true",
                   help="GET the form page, dump the batch form fields, exit")
    p.add_argument("--ids", type=Path, default=None,
                   help="PDB-ID list (one per line)")
    p.add_argument("--email", default=None, help="notification email")
    p.add_argument("--job-name", default=None, help="job name (default: ids stem)")
    p.add_argument("--patch-num", type=int, default=3, choices=[1, 2, 3])
    p.add_argument("--no-pos", action="store_true",
                   help="disable positive patches (default: on)")
    p.add_argument("--neg", action="store_true",
                   help="also request negative patches")
    p.add_argument("--batch-size", type=int, default=0,
                   help="split the list into jobs of this size (0 = one job)")
    p.add_argument("--url", default=ACTION_URL)
    p.add_argument("--base-url", default=BASE_URL)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="where to save response HTML (default: <ids> dir)")
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--no-proxy", action="store_true")
    p.add_argument("--insecure", action="store_true",
                   help="skip TLS verify (Technion omits an intermediate CA)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    if args.inspect:
        return run_inspect(args)
    if args.ids is None:
        print("ERROR: --ids is required to submit (or use --inspect)",
              file=sys.stderr)
        return 2
    if not args.email and not args.dry_run:
        print("WARNING: no --email set; batch results may only be on the job "
              "page (no email notification).", file=sys.stderr)
    return submit(args)


if __name__ == "__main__":
    raise SystemExit(main())
