"""Acquire ONLY an authorized official LRW archive. No credentials or URLs are invented.

Approval alone does not supply credentials. Use the actual approved URL and local
environment variables, or verify/extract an already downloaded official archive.
The single complete archive MD5 is published on the official Oxford LRW page.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import time
from urllib.parse import urlsplit
from urllib.request import (Request, build_opener, HTTPBasicAuthHandler,
                            HTTPPasswordMgrWithDefaultRealm, HTTPRedirectHandler)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OFFICIAL_PAGE = "https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html"
OFFICIAL_MD5 = "0dee65af9c2087d22602e6bc2b91a744"


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


class SameOriginRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        previous, next_url = urlsplit(request.full_url), urlsplit(newurl)
        if (previous.scheme, previous.netloc) != (next_url.scheme, next_url.netloc):
            raise ValueError("Cross-origin/downgrade redirects rejected; use the approved final HTTPS URL.")
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def authorized_opener(url):
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.username or parts.password or parts.hostname not in {"thor.robots.ox.ac.uk", "www.robots.ox.ac.uk"}:
        raise ValueError("Use the exact approved Oxford HTTPS URL without embedded credentials.")
    manager = HTTPPasswordMgrWithDefaultRealm()
    username, password = os.environ.get("LIPNET_LRW_USERNAME"), os.environ.get("LIPNET_LRW_PASSWORD")
    if username and password:
        manager.add_password(None, url, username, password)
    return build_opener(SameOriginRedirect(), HTTPBasicAuthHandler(manager))


def download(url, destination, opener, *, reserve_bytes=20 * 2**30, deadline=None, attempts=3):
    """Retain .part, check entity/range/length, reject login HTML; never log signed URLs."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    partial = destination.with_name(destination.name + ".part")
    sidecar = partial.with_name(partial.name + ".json")
    source_id = hashlib.sha256(url.encode()).hexdigest()
    for attempt in range(attempts):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Time budget reached; partial download retained.")
        previous = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        offset = partial.stat().st_size if partial.exists() else 0
        if offset and previous.get("source_id") != source_id:
            raise ValueError("Partial file belongs to another source; preserved.")
        if offset and offset == previous.get("expected_bytes"):
            return partial
        headers = {"Accept-Encoding": "identity", "User-Agent": "LipNet-authorized-LRW-acquisition/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
            if previous.get("etag") or previous.get("last_modified"):
                headers["If-Range"] = previous.get("etag") or previous["last_modified"]
        try:
            with opener.open(Request(url, headers=headers), timeout=30) as response:
                if response.status not in (200, 206) or "html" in response.headers.get("Content-Type", "").lower():
                    raise ValueError("Expected an archive, not a login/HTML response.")
                if response.status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not match or int(match[1]) != offset or int(match[2]) < offset or int(match[2]) >= int(match[3]):
                        raise ValueError("Invalid HTTP resume range.")
                    expected, mode = int(match[3]), "ab"
                    if previous.get("expected_bytes") not in (None, expected):
                        raise ValueError("Resume source size changed; existing bytes preserved.")
                    if previous.get("etag") and response.headers.get("ETag") and previous["etag"] != response.headers["ETag"]:
                        raise ValueError("Resume entity changed; existing bytes preserved.")
                else:
                    expected, offset, mode = int(response.headers.get("Content-Length", 0)), 0, "wb"
                if expected <= 0:
                    raise ValueError("A verifiable Content-Length/Content-Range is required.")
                if shutil.disk_usage(destination.parent).free < reserve_bytes + expected - offset:
                    raise OSError("Insufficient disk for remaining archive bytes plus reserve.")
                metadata = {"source_id": source_id, "expected_bytes": expected, "etag": response.headers.get("ETag"),
                            "last_modified": response.headers.get("Last-Modified"), "last_http_status": response.status}
                # Inspect first bytes BEFORE truncating an old partial when Range was ignored.
                block = response.read(1024 * 1024)
                if offset == 0 and (block.lstrip().lower().startswith((b"<!doctype html", b"<html")) or not block.startswith(b"BZh")):
                    raise ValueError("Not a bzip2 LRW archive; source response rejected.")
                atomic_json(sidecar, metadata)
                with partial.open(mode) as stream:
                    while block:
                        if deadline is not None and time.monotonic() >= deadline:
                            raise TimeoutError("Time budget reached; .part retained.")
                        stream.write(block)
                        block = response.read(1024 * 1024)
                if partial.stat().st_size != expected:
                    raise ValueError("Truncated/oversized download; .part retained.")
                return partial
        except (OSError, ValueError):
            if attempt + 1 == attempts:
                raise
    raise RuntimeError("No download attempt completed.")


def safe_member(name, root):
    # Explicit Windows rules, including drive paths and alternate data streams.
    if "\\" in name or ":" in name:
        raise ValueError("Unsafe Windows archive path.")
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Archive path traversal rejected.")
    target = root.joinpath(*relative.parts).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Archive path escapes output.")
    return target


def extract_archive(archive, output, *, reserve_bytes=20 * 2**30):
    """Two streaming passes: validate member policy/space first; extract preserving bytes.

    Links/devices are forbidden. Existing members must match archive content exactly.
    No source clips are re-encoded or overwritten; partial extracted members resume.
    """
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    total, names = 0, set()
    with tarfile.open(archive, "r|bz2") as source:
        for item in source:
            target = safe_member(item.name, output)
            if not item.isdir() and not item.isfile():
                raise ValueError("Archive links/devices are not allowed.")
            if str(target).casefold() in names:
                raise ValueError("Duplicate/case-colliding archive paths.")
            names.add(str(target).casefold())
            if item.isfile():
                if target.exists() and not target.is_file():
                    raise ValueError("Archive file conflicts with an existing directory.")
                if not target.exists():
                    total += item.size
    if shutil.disk_usage(output).free < total + reserve_bytes:
        raise OSError(f"Extraction needs {total:,} additional bytes plus reserve.")
    written = reused = 0
    with tarfile.open(archive, "r|bz2") as source:
        for item in source:
            target = safe_member(item.name, output)
            if item.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            member = source.extractfile(item)
            temporary = target.with_name(target.name + ".lrw-part")
            if target.exists():
                h = hashlib.sha256()
                for block in iter(lambda: member.read(1024 * 1024), b""):
                    h.update(block)
                if target.stat().st_size != item.size or digest(target) != h.hexdigest():
                    raise ValueError(f"Existing extracted file differs; preserved: {target}")
                reused += 1
            else:
                if temporary.exists():
                    if temporary.stat().st_size > item.size:
                        raise ValueError("Extracted partial is larger than its archive member; preserved.")
                    with temporary.open("rb") as prefix:
                        for block in iter(lambda: prefix.read(1024 * 1024), b""):
                            if member.read(len(block)) != block:
                                raise ValueError("Extracted partial differs from archive prefix; preserved.")
                with temporary.open("ab" if temporary.exists() else "xb") as destination:
                    shutil.copyfileobj(member, destination, length=1024 * 1024)
                if temporary.stat().st_size != item.size:
                    raise ValueError("Truncated archive member; partial retained.")
                if target.exists():
                    raise FileExistsError("Concurrent extraction target is protected.")
                temporary.rename(target)
                written += 1
    return {"written_files": written, "reused_verified_files": reused, "additional_bytes": total}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Exact BBC-approved Oxford URL, never guessed")
    parser.add_argument("--archive", type=Path, required=True, help="Local full lrw-v1.tar.bz2 path; download target if --url supplied")
    parser.add_argument("--extract-to", type=Path)
    parser.add_argument("--agreement-confirmed", action="store_true")
    parser.add_argument("--max-minutes", type=float, default=10)
    parser.add_argument("--reserve-gib", type=float, default=20)
    args = parser.parse_args()
    if not args.agreement_confirmed:
        parser.error("BBC agreement/authorized access must already be in place. This flag does not obtain approval.")
    if args.max_minutes <= 0 or args.reserve_gib < 0:
        parser.error("Positive time budget and nonnegative disk reserve required.")
    try:
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        path = download(args.url, args.archive, authorized_opener(args.url),
                        reserve_bytes=int(args.reserve_gib * 2**30), deadline=time.monotonic() + args.max_minutes * 60) if args.url else args.archive
        if digest(path, "md5") != OFFICIAL_MD5:
            raise ValueError("Archive does not match the publisher's full LRW MD5; preserved, not extracted.")
        fingerprint = digest(path)
        if path != args.archive:
            if args.archive.exists():
                raise FileExistsError("Existing final archive is protected.")
            path.rename(args.archive)
        report = {"publisher": OFFICIAL_PAGE, "publisher_md5": OFFICIAL_MD5,
                  "local_sha256": fingerprint, "bytes": args.archive.stat().st_size,
                  "agreement_confirmed_by_user": True, "checked_utc": datetime.now(timezone.utc).isoformat()}
        if args.extract_to:
            report["extraction"] = extract_archive(args.archive, args.extract_to, reserve_bytes=int(args.reserve_gib * 2**30))
        atomic_json(args.archive.with_name(args.archive.name + ".verified.json"), report)
        print(json.dumps(report, indent=2))
    except Exception as error:
        # URL/HTTP exceptions can contain secrets from an approved signed URL.
        detail = str(error) if isinstance(error, (ValueError, FileExistsError)) else "Check approved access, archive integrity, and free disk."
        parser.exit(2, f"LRW acquisition stopped ({type(error).__name__}): {detail} Existing files/partials preserved.\n")


if __name__ == "__main__":
    main()
