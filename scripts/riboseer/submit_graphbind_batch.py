"""Batch-submit protein PDB files to the BindWeb server for RNA-binding prediction.

BindWeb (http://www.csbio.sjtu.edu.cn/bioinf/BindWeb/) wraps GraphBind and only
exposes a web form, so this script drives that form with ``requests`` /
``BeautifulSoup``:

  * **submit mode** (default) — for each sample in ``graphbind_submission.csv``
    it uploads the single-chain PDB, sets chain ``A`` and ligand type ``RNA``,
    POSTs the form, scrapes the returned ``job_id`` / result URL, and appends a
    row to ``graphbind_jobs.csv``. A 5 s gap separates submissions.
  * **poll mode** (``--poll``) — periodically GETs each job's result page and,
    once it stops looking "queued/running", downloads the page (and any linked
    result files) into ``graphbind_results/<sample_id>/``.
  * **inspect mode** (``--inspect``) — GET the form page and print every form,
    its action URL and field names. Use this to confirm/override field names
    when the server markup changes.

Because the live form could not be observed while writing this (the server was
returning 502), the submit path *auto-discovers* the form at run time: it reads
the GET page, finds the form containing a file input, preserves all hidden
defaults, and locates the file / chain / ligand / email fields by name and by
option text. Every auto-detected field can be overridden from the CLI, and the
first real submission should be sanity-checked with ``--dry-run`` /
``--limit 1``.

Usage
-----
::

    # inspect the form once the server is up (discover field names)
    python scripts/riboseer/submit_graphbind_batch.py --inspect

    # dry-run the first sample (discover + show the POST, do not send)
    python scripts/riboseer/submit_graphbind_batch.py \\
        --inputs-dir data/batch_test_v7/graphbind_inputs/ \\
        --submission-csv data/batch_test_v7/graphbind_submission.csv \\
        --output-csv data/batch_test_v7/graphbind_jobs.csv \\
        --email you@example.com --limit 1 --dry-run

    # batch submit samples [0, 30)
    python scripts/riboseer/submit_graphbind_batch.py \\
        --inputs-dir data/batch_test_v7/graphbind_inputs/ \\
        --submission-csv data/batch_test_v7/graphbind_submission.csv \\
        --output-csv data/batch_test_v7/graphbind_jobs.csv \\
        --email you@example.com --start 0 --end 30 --resume

    # poll all submitted jobs until results are ready
    python scripts/riboseer/submit_graphbind_batch.py \\
        --poll \\
        --jobs-csv data/batch_test_v7/graphbind_jobs.csv \\
        --results-dir data/batch_test_v7/graphbind_results/
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "http://www.csbio.sjtu.edu.cn/bioinf/BindWeb/"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
JOBS_FIELDNAMES = ["sample_id", "job_id", "result_url", "submit_time"]

# Heuristic keywords for poll-mode completion detection.
PENDING_MARKERS = (
    "running", "in the queue", "please wait", "is being processed",
    "not finished", "pending", "submitted", "refresh", "waiting",
)
DONE_MARKERS = (
    "binding residues", "prediction result", "binding sites",
    "download", "binding probability", "predicted",
)


# --------------------------------------------------------------------------- #
# session / form discovery
# --------------------------------------------------------------------------- #
def make_session(no_proxy: bool = False) -> requests.Session:
    s = requests.Session()
    if no_proxy:
        # ignore HTTP(S)_PROXY env vars; some proxies cannot reach the SJTU host
        s.trust_env = False
        s.proxies = {"http": None, "https": None}
    s.headers.update({"User-Agent": USER_AGENT})
    retry = requests.adapters.Retry(
        total=3, backoff_factor=1.0,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
    )
    adapter = requests.adapters.HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


class FormSpec:
    """A discovered HTML form ready to be filled and POSTed."""

    def __init__(self, action: str, method: str, defaults: dict,
                 file_field: Optional[str], chain_field: Optional[str],
                 ligand_field: Optional[str], ligand_value: Optional[str],
                 email_field: Optional[str]):
        self.action = action
        self.method = method
        self.defaults = defaults
        self.file_field = file_field
        self.chain_field = chain_field
        self.ligand_field = ligand_field
        self.ligand_value = ligand_value
        self.email_field = email_field

    def describe(self) -> str:
        return (
            f"  action       : {self.action}\n"
            f"  method       : {self.method}\n"
            f"  file field   : {self.file_field}\n"
            f"  chain field  : {self.chain_field}\n"
            f"  ligand field : {self.ligand_field} = {self.ligand_value!r}\n"
            f"  email field  : {self.email_field}\n"
            f"  hidden/default fields: {self.defaults}"
        )


def _iter_forms(soup: BeautifulSoup):
    return soup.find_all("form")


def _form_inputs(form):
    """Collect (defaults, file_fields, selectable_options) from a form."""
    defaults: dict = {}
    file_fields: list = []
    # option_text_by_field: field_name -> {option_text_lower: option_value}
    options: dict = {}

    for inp in form.find_all("input"):
        name = inp.get("name")
        if not name:
            continue
        itype = (inp.get("type") or "text").lower()
        if itype == "file":
            file_fields.append(name)
        elif itype in ("radio", "checkbox"):
            val = inp.get("value", "on")
            # label text is hard to map reliably; index by value text too
            options.setdefault(name, {})[str(val).lower()] = val
            if inp.has_attr("checked"):
                defaults[name] = val
        elif itype in ("submit", "button", "image", "reset"):
            continue
        else:
            defaults[name] = inp.get("value", "")

    for sel in form.find_all("select"):
        name = sel.get("name")
        if not name:
            continue
        opts = {}
        default_val = None
        for opt in sel.find_all("option"):
            val = opt.get("value", opt.get_text(strip=True))
            txt = opt.get_text(strip=True)
            opts[txt.lower()] = val
            opts[str(val).lower()] = val
            if opt.has_attr("selected"):
                default_val = val
        options[name] = opts
        if default_val is not None:
            defaults[name] = default_val

    for ta in form.find_all("textarea"):
        name = ta.get("name")
        if name:
            defaults[name] = ta.get_text()

    return defaults, file_fields, options


def discover_form(session: requests.Session, page_url: str,
                  overrides: dict) -> FormSpec:
    r = session.get(page_url, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")

    chosen = None
    for form in _iter_forms(soup):
        _, file_fields, _ = _form_inputs(form)
        if file_fields:
            chosen = form
            break
    if chosen is None:
        # fall back to the first form on the page
        forms = _iter_forms(soup)
        if not forms:
            raise RuntimeError(f"no <form> found at {page_url}")
        chosen = forms[0]

    defaults, file_fields, options = _form_inputs(chosen)
    action = urljoin(page_url, chosen.get("action") or page_url)
    method = (chosen.get("method") or "post").lower()

    def pick(names: dict, *needles) -> Optional[str]:
        for nm in names:
            low = nm.lower()
            if any(n in low for n in needles):
                return nm
        return None

    file_field = overrides.get("file_field") or (file_fields[0] if file_fields else None)
    chain_field = overrides.get("chain_field") or pick(defaults, "chain")

    # ligand / function field: a select or radio group with an "RNA" option
    ligand_field = overrides.get("ligand_field")
    ligand_value = overrides.get("ligand_value")
    if ligand_field is None:
        for name, opts in options.items():
            if "rna" in opts:
                ligand_field = name
                ligand_value = ligand_value or opts["rna"]
                break
        if ligand_field is None:
            ligand_field = pick(defaults, "ligand", "function", "type", "mode")
    if ligand_value is None and ligand_field in options and "rna" in options[ligand_field]:
        ligand_value = options[ligand_field]["rna"]
    if ligand_value is None:
        ligand_value = "RNA"

    email_field = overrides.get("email_field") or pick(defaults, "email", "mail")

    return FormSpec(action, method, defaults, file_field, chain_field,
                    ligand_field, ligand_value, email_field)


# --------------------------------------------------------------------------- #
# submit
# --------------------------------------------------------------------------- #
RESULT_RE = re.compile(r"results/([A-Za-z0-9_\-]+)/\1\.html")
ANY_RESULT_HREF_RE = re.compile(r"""['"]([^'"]*results/[^'"]+\.html)['"]""")
JOBID_RE = re.compile(r"job[\s_]*id[^A-Za-z0-9]{0,5}([A-Za-z0-9_\-]+)", re.I)


def parse_submit_response(resp_text: str, base: str):
    """Return (job_id, result_url) scraped from the submission response."""
    m = RESULT_RE.search(resp_text)
    if m:
        job_id = m.group(1)
        return job_id, urljoin(base, m.group(0))
    m = ANY_RESULT_HREF_RE.search(resp_text)
    if m:
        url = urljoin(base, m.group(1))
        jm = re.search(r"results/([A-Za-z0-9_\-]+)/", url)
        return (jm.group(1) if jm else ""), url
    # Maybe a redirect-style page that just states the job id.
    m = JOBID_RE.search(resp_text)
    if m:
        job_id = m.group(1)
        return job_id, urljoin(base, f"results/{job_id}/{job_id}.html")
    return None, None


def submit_one(session: requests.Session, spec: FormSpec, pdb_path: Path,
               chain: str, email: str, dry_run: bool, debug_dir: Optional[Path],
               extra: Optional[dict] = None):
    data = dict(spec.defaults)
    if spec.chain_field:
        data[spec.chain_field] = chain
    if spec.ligand_field:
        data[spec.ligand_field] = spec.ligand_value
    if spec.email_field and email:
        data[spec.email_field] = email
    if extra:
        data.update(extra)

    if not spec.file_field:
        raise RuntimeError("no file upload field detected; pass --file-field")

    files = {spec.file_field: (pdb_path.name, pdb_path.read_bytes(), "chemical/x-pdb")}

    if dry_run:
        print(f"[dry-run] POST {spec.action}")
        print(f"          data  = {data}")
        print(f"          files = {{{spec.file_field}: {pdb_path.name}}}")
        return None, None

    resp = session.post(spec.action, data=data, files=files, timeout=60)
    resp.raise_for_status()
    job_id, result_url = parse_submit_response(resp.text, spec.action)
    if job_id is None and debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        dbg = debug_dir / f"submit_{pdb_path.stem}.html"
        dbg.write_text(resp.text, encoding="utf-8", errors="replace")
        print(f"      ! could not parse job_id; saved response -> {dbg}", file=sys.stderr)
    return job_id, result_url


def load_submission(submission_csv: Path):
    with submission_csv.open(newline="") as f:
        return list(csv.DictReader(f))


def load_done_samples(output_csv: Path) -> set:
    if not output_csv.exists():
        return set()
    with output_csv.open(newline="") as f:
        return {r["sample_id"] for r in csv.DictReader(f) if r.get("job_id")}


def append_job_row(output_csv: Path, row: dict):
    new = not output_csv.exists()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=JOBS_FIELDNAMES)
        if new:
            w.writeheader()
        w.writerow(row)


def run_submit(args):
    session = make_session(no_proxy=args.no_proxy)
    overrides = {
        "file_field": args.file_field,
        "chain_field": args.chain_field,
        "ligand_field": args.ligand_field,
        "ligand_value": args.ligand_value,
        "email_field": args.email_field,
    }
    overrides = {k: v for k, v in overrides.items() if v}

    extra_fields = {}
    for kv in (args.set or []):
        if "=" not in kv:
            print(f"ERROR: --set expects KEY=VALUE, got {kv!r}", file=sys.stderr)
            return 2
        k, v = kv.split("=", 1)
        extra_fields[k] = v

    print(f"discovering form at {args.url} ...")
    try:
        spec = discover_form(session, args.url, overrides)
    except requests.RequestException as e:
        print(f"ERROR: could not reach BindWeb ({e}). The server may be down "
              f"(it was returning 502 when this script was written). Retry later "
              f"or verify the URL.", file=sys.stderr)
        return 3
    print(spec.describe())
    if spec.file_field is None:
        print("ERROR: no file upload field found. Run --inspect and pass "
              "--file-field/--chain-field/--ligand-field explicitly.", file=sys.stderr)
        return 2

    rows = load_submission(args.submission_csv)
    end = args.end if args.end is not None else len(rows)
    rows = rows[args.start:end]
    if args.limit:
        rows = rows[: args.limit]

    done = load_done_samples(args.output_csv) if args.resume else set()
    inputs_dir = args.inputs_dir
    debug_dir = args.output_csv.parent / "graphbind_submit_debug"

    n_ok = n_skip = n_fail = 0
    for i, r in enumerate(rows):
        sid = r["sample_id"]
        fname = r.get("pdb_filename") or f"{sid}.pdb"
        if args.resume and sid in done:
            n_skip += 1
            print(f"[{i+1}/{len(rows)}] {sid}: already submitted, skip")
            continue
        pdb_path = inputs_dir / fname
        if not pdb_path.exists():
            n_fail += 1
            print(f"[{i+1}/{len(rows)}] {sid}: MISSING {pdb_path}", file=sys.stderr)
            continue
        try:
            job_id, result_url = submit_one(
                session, spec, pdb_path, args.chain, args.email,
                args.dry_run, debug_dir, extra=extra_fields)
        except Exception as e:
            n_fail += 1
            print(f"[{i+1}/{len(rows)}] {sid}: SUBMIT ERROR {e}", file=sys.stderr)
            continue
        if args.dry_run:
            continue
        if not job_id:
            n_fail += 1
            print(f"[{i+1}/{len(rows)}] {sid}: no job_id parsed (see debug dir)", file=sys.stderr)
        else:
            n_ok += 1
            append_job_row(args.output_csv, {
                "sample_id": sid,
                "job_id": job_id,
                "result_url": result_url or "",
                "submit_time": datetime.now().isoformat(timespec="seconds"),
            })
            print(f"[{i+1}/{len(rows)}] {sid}: job_id={job_id}  -> {result_url}")
        if i < len(rows) - 1:
            time.sleep(args.sleep)

    print(f"\nsubmitted={n_ok}  skipped={n_skip}  failed={n_fail}")
    if not args.dry_run:
        print(f"jobs csv: {args.output_csv}")
    return 0


# --------------------------------------------------------------------------- #
# poll
# --------------------------------------------------------------------------- #
META_REFRESH_RE = re.compile(r"""http-equiv\s*=\s*['"]?refresh""", re.I)


def page_is_done(text: str) -> bool:
    # BindWeb's pending result page carries a <meta http-equiv="refresh"> tag
    # that auto-reloads every 10 s; the finished page drops it. That tag is the
    # most reliable completion signal, so treat its presence as "still running".
    if META_REFRESH_RE.search(text):
        return False
    low = text.lower()
    if any(p in low for p in ("in the queue", "is running", "please wait",
                              "not finished", "is being processed")):
        return False
    return any(m in low for m in DONE_MARKERS)


def download_linked_results(session, result_url, text, out_dir: Path):
    """Grab result files (.txt/.pdb/.csv/.dat) linked from the result page."""
    soup = BeautifulSoup(text, "lxml")
    saved = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if re.search(r"\.(txt|pdb|csv|dat|out|res)$", href, re.I):
            url = urljoin(result_url, href)
            try:
                rr = session.get(url, timeout=60)
                if rr.status_code == 200 and rr.content:
                    name = Path(url.split("?")[0]).name
                    (out_dir / name).write_bytes(rr.content)
                    saved.append(name)
            except Exception as e:
                print(f"      ! failed to download {url}: {e}", file=sys.stderr)
    return saved


def run_poll(args):
    session = make_session(no_proxy=args.no_proxy)
    with args.jobs_csv.open(newline="") as f:
        jobs = [r for r in csv.DictReader(f) if r.get("result_url")]
    if not jobs:
        print("no jobs with result_url in jobs csv", file=sys.stderr)
        return 2

    pending = {j["sample_id"]: j for j in jobs}
    done_samples: set = set()
    rounds = 0
    while pending and (args.max_rounds == 0 or rounds < args.max_rounds):
        rounds += 1
        print(f"--- poll round {rounds}: {len(pending)} pending ---")
        for sid, j in list(pending.items()):
            url = j["result_url"]
            out_dir = args.results_dir / sid
            try:
                r = session.get(url, timeout=60)
            except Exception as e:
                print(f"  {sid}: GET error {e}", file=sys.stderr)
                continue
            if r.status_code == 404:
                print(f"  {sid}: result not ready (404)")
                continue
            if r.status_code != 200:
                print(f"  {sid}: HTTP {r.status_code}")
                continue
            if page_is_done(r.text):
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "result.html").write_text(
                    r.text, encoding="utf-8", errors="replace")
                files = download_linked_results(session, url, r.text, out_dir)
                done_samples.add(sid)
                del pending[sid]
                print(f"  {sid}: DONE -> {out_dir}  files={['result.html', *files]}")
            else:
                print(f"  {sid}: still running")
        if pending and (args.max_rounds == 0 or rounds < args.max_rounds):
            time.sleep(args.poll_interval)

    print(f"\ndone={len(done_samples)}  still_pending={len(pending)}")
    if pending:
        print("pending:", ", ".join(sorted(pending)))
    return 0


# --------------------------------------------------------------------------- #
# inspect
# --------------------------------------------------------------------------- #
def run_inspect(args):
    session = make_session(no_proxy=args.no_proxy)
    try:
        r = session.get(args.url, timeout=30)
    except requests.RequestException as e:
        print(f"ERROR: could not reach {args.url} ({e}). Server may be down.",
              file=sys.stderr)
        return 3
    print(f"GET {args.url} -> {r.status_code}, {len(r.text)} bytes\n")
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    forms = _iter_forms(soup)
    if not forms:
        print("(no <form> elements found)")
        return 0
    for idx, form in enumerate(forms):
        defaults, file_fields, options = _form_inputs(form)
        action = urljoin(args.url, form.get("action") or args.url)
        print(f"=== form #{idx} ===")
        print(f"  action: {action}")
        print(f"  method: {(form.get('method') or 'get').lower()}")
        print(f"  file fields: {file_fields}")
        print(f"  text/hidden defaults: {defaults}")
        for name, opts in options.items():
            print(f"  select/radio '{name}' options: {sorted(set(opts.values()), key=str)}")
        print()
    return 0


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=BASE_URL, help="BindWeb form page URL")
    p.add_argument("--no-proxy", action="store_true",
                   help="bypass HTTP(S)_PROXY env (needed if the local proxy cannot reach SJTU)")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--poll", action="store_true", help="poll result pages")
    mode.add_argument("--inspect", action="store_true",
                      help="GET form page and dump form fields, then exit")

    # submit args
    p.add_argument("--inputs-dir", type=Path,
                   default=Path("data/batch_test_v7/graphbind_inputs"))
    p.add_argument("--submission-csv", type=Path,
                   default=Path("data/batch_test_v7/graphbind_submission.csv"))
    p.add_argument("--output-csv", type=Path,
                   default=Path("data/batch_test_v7/graphbind_jobs.csv"))
    p.add_argument("--email", default="", help="notification email (optional)")
    p.add_argument("--chain", default="A", help="chain ID submitted to BindWeb")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--limit", type=int, default=0, help="cap number submitted (after start/end)")
    p.add_argument("--sleep", type=float, default=5.0, help="seconds between submissions")
    p.add_argument("--resume", action="store_true", help="skip samples already in output csv")
    p.add_argument("--dry-run", action="store_true", help="discover form + show POST, do not send")

    # form field overrides (used if auto-detect is wrong)
    p.add_argument("--file-field")
    p.add_argument("--chain-field")
    p.add_argument("--ligand-field")
    p.add_argument("--ligand-value")
    p.add_argument("--email-field")
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="set/override an extra POST field (repeatable), "
                        "e.g. --set pdbinput=file")

    # poll args
    p.add_argument("--jobs-csv", type=Path,
                   default=Path("data/batch_test_v7/graphbind_jobs.csv"))
    p.add_argument("--results-dir", type=Path,
                   default=Path("data/batch_test_v7/graphbind_results"))
    p.add_argument("--poll-interval", type=float, default=60.0)
    p.add_argument("--max-rounds", type=int, default=0, help="0 = until all done")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.inspect:
        return run_inspect(args)
    if args.poll:
        return run_poll(args)
    return run_submit(args)


if __name__ == "__main__":
    raise SystemExit(main())
