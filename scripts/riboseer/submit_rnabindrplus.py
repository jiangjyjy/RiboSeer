#!/usr/bin/env python3
"""Submit a FASTA batch to the RNABindRPlus web server.

The submission form (discovered at
http://ailab-projects2.ist.psu.edu/RNABindRPlus/) has **no CAPTCHA** and posts
multipart/form-data to::

    https://ailab-projects.ist.psu.edu/RNABindRPlus/cgi-bin/predict.cgi

fields: ``email``, ``JobTitle``, ``QuerySeq`` (the FASTA text),
``protSimilarityThr`` (default 95), ``rmSimilarProt`` (checkbox, "Yes").
Results are returned **by email**, so one POST per batch FASTA file = one job =
one email. Each sequence takes ~10 min, so submit the four batches as four jobs.

Modes
-----
* default  : POST one ``--fasta`` batch and save the server response HTML.
* --inspect: GET the form page and print every ``<form>`` / field name (use to
  re-confirm the field names if the server markup changes).
* --dry-run: build the request and print it, but do not send.

Usage
-----
::

    # submit batch 1
    python scripts/riboseer/submit_rnabindrplus.py \
        --fasta data/batch_test_v7/rnabindrplus_input_batch1.fasta \
        --email you@example.com --job-title riboseer_b1

    # re-confirm the form fields
    python scripts/riboseer/submit_rnabindrplus.py --inspect
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

BASE_URL = "http://ailab-projects2.ist.psu.edu/RNABindRPlus/"
ACTION_URL = "https://ailab-projects.ist.psu.edu/RNABindRPlus/cgi-bin/predict.cgi"
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
# Heuristic markers that a returned page is a CAPTCHA / human-check wall.
_CAPTCHA_MARKERS = ("captcha", "recaptcha", "g-recaptcha", "hcaptcha",
                    "are you human", "verify you are")


def make_session(no_proxy: bool = False):
    if requests is None:
        raise RuntimeError("the 'requests' package is required: {}".format(_REQUESTS_ERR))
    s = requests.Session()
    if no_proxy:
        s.trust_env = False
        s.proxies = {"http": None, "https": None}
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def run_inspect(args) -> int:
    s = make_session(args.no_proxy)
    try:
        r = s.get(args.base_url, timeout=30)
    except Exception as e:  # noqa: BLE001
        print("ERROR: could not reach {} ({})".format(args.base_url, e), file=sys.stderr)
        return 3
    print("GET {} -> {} ({} bytes)\n".format(args.base_url, r.status_code, len(r.text)))
    forms = re.findall(r"(?is)<form[^>]*>", r.text)
    for f in forms:
        print("FORM:", f.strip())
    fields = re.findall(r"(?is)<(?:input|textarea|select)[^>]*>", r.text)
    print("\nfields:")
    for fld in fields:
        name = re.search(r"""name\s*=\s*['"]?([^'">\s]+)""", fld)
        typ = re.search(r"""type\s*=\s*['"]?([^'">\s]+)""", fld)
        print("  name={!s:20} type={!s:10} {}".format(
            name.group(1) if name else "-", typ.group(1) if typ else "-", fld.strip()[:80]))
    if any(m in r.text.lower() for m in _CAPTCHA_MARKERS):
        print("\n! CAPTCHA markers detected — automated submit may be blocked.",
              file=sys.stderr)
    return 0


def submit(args) -> int:
    fasta = args.fasta
    if fasta is None or not fasta.is_file():
        print("ERROR: --fasta file required and must exist: {}".format(fasta),
              file=sys.stderr)
        return 2
    query_seq = fasta.read_text(encoding="utf-8")
    job_title = args.job_title or fasta.stem

    fields = {
        "email": args.email,
        "JobTitle": job_title,
        "QuerySeq": query_seq,
        "protSimilarityThr": str(args.similarity_thr),
        ".submit": "Submit",
    }
    if args.rm_similar:
        fields["rmSimilarProt"] = "Yes"

    n_seqs = query_seq.count(">")
    if args.dry_run:
        print("[dry-run] POST {}".format(args.url))
        print("          email      = {}".format(args.email))
        print("          JobTitle   = {}".format(job_title))
        print("          QuerySeq   = <{} seqs, {} chars>".format(n_seqs, len(query_seq)))
        print("          protSimilarityThr = {}  rmSimilarProt = {}".format(
            args.similarity_thr, "Yes" if args.rm_similar else "(off)"))
        return 0

    s = make_session(args.no_proxy)
    # multipart/form-data: pass every field as a (None, value) file part.
    files = {k: (None, v) for k, v in fields.items()}
    try:
        resp = s.post(args.url, files=files, timeout=args.timeout)
    except Exception as e:  # noqa: BLE001
        print("ERROR: POST failed ({})".format(e), file=sys.stderr)
        return 3

    out = args.out or fasta.with_suffix(".response.html")
    out.write_text(resp.text, encoding="utf-8", errors="replace")
    print("POST {} -> {} ({} bytes); response saved -> {}".format(
        args.url, resp.status_code, len(resp.text), out))

    low = resp.text.lower()
    if any(m in low for m in _CAPTCHA_MARKERS):
        print("! CAPTCHA detected in response — submission likely NOT accepted. "
              "Paste the FASTA manually at {}".format(BASE_URL), file=sys.stderr)
        return 4
    # surface anything that looks like a job id / confirmation
    jid = re.search(r"(job[\s_-]*id[^A-Za-z0-9]{0,5}[A-Za-z0-9_\-]+)", resp.text, re.I)
    if jid:
        print("  ", jid.group(1))
    if resp.status_code == 200:
        print("  submitted {} sequence(s); results will arrive by email to {}".format(
            n_seqs, args.email))
    return 0


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inspect", action="store_true",
                   help="GET the form page, dump forms/fields, then exit")
    p.add_argument("--fasta", type=Path, default=None,
                   help="batch FASTA file to submit (pasted into QuerySeq)")
    p.add_argument("--email", default="", help="notification email (required to submit)")
    p.add_argument("--job-title", default=None, help="job title (default: fasta stem)")
    p.add_argument("--url", default=ACTION_URL, help="form action URL")
    p.add_argument("--base-url", default=BASE_URL, help="form page URL (for --inspect)")
    p.add_argument("--similarity-thr", type=int, default=95,
                   help="protSimilarityThr (homolog %% identity, default 95)")
    p.add_argument("--rm-similar", action="store_true",
                   help="set rmSimilarProt=Yes (remove highly similar homologs)")
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--no-proxy", action="store_true",
                   help="bypass HTTP(S)_PROXY env vars")
    p.add_argument("--dry-run", action="store_true",
                   help="build the request and print it, do not send")
    p.add_argument("--out", type=Path, default=None,
                   help="where to save the response HTML (default <fasta>.response.html)")
    args = p.parse_args(argv)

    if args.inspect:
        return run_inspect(args)
    if not args.email and not args.dry_run:
        print("ERROR: --email is required to submit (or use --dry-run / --inspect)",
              file=sys.stderr)
        return 2
    return submit(args)


if __name__ == "__main__":
    raise SystemExit(main())
