#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import fnmatch
from urllib.parse import urljoin, quote
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

# --- Helpers ---------------------------------------------------------------

def iter_html_files(root_dir):
    # Yield absolute file paths for .html/.htm
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in {".html", ".htm"}:
                yield os.path.join(dirpath, name)

def rel_url_path(path_abs, root_dir):
    # Compute a POSIX-style relative URL path (URL-encoded)
    rel = os.path.relpath(path_abs, root_dir)
    rel = rel.replace(os.sep, "/")
    # Encode each path segment safely
    parts = [quote(p) for p in rel.split("/")]
    return "/".join(parts)

def file_lastmod_utc_str(path_abs):
    # Format file mtime as ISO 8601 with explicit +00:00 offset
    ts = os.path.getmtime(path_abs)
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

def format_priority(value):
    # Ensure two decimal places as string
    return f"{float(value):.2f}"

def guess_priority(rel_path, depth, args, lower_priority_globs):
    # Root priority handled separately
    # If rel_path matches any "lower" glob -> deep priority
    for pattern in lower_priority_globs:
        if fnmatch.fnmatch(rel_path, pattern):
            return args.priority_deep
    # Depth-based rule
    if depth >= args.deep_depth:
        return args.priority_deep
    return args.priority_default

def max_lastmod_from_files(paths):
    # Return latest mtime among given files (UTC ISO string)
    if not paths:
        now = datetime.now(timezone.utc)
        return now.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    latest_ts = max(os.path.getmtime(p) for p in paths)
    dt = datetime.fromtimestamp(latest_ts, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

# --- Main ------------------------------------------------------------------

def build_sitemap(root_dir, base_url, output, include_root, args):
    # Normalize base_url to end with '/'
    if not base_url.endswith("/"):
        base_url += "/"

    # Collect pages
    files = sorted(iter_html_files(root_dir))
    items = []

    # Optional root ("/") entry, lastmod = mtime of root index.* if present, else max of all
    root_index_candidates = [
        os.path.join(root_dir, "index.html"),
        os.path.join(root_dir, "index.htm"),
    ]
    if include_root:
        existing_indices = [p for p in root_index_candidates if os.path.isfile(p)]
        if existing_indices:
            lastmod_root = file_lastmod_utc_str(existing_indices[0])
        else:
            lastmod_root = max_lastmod_from_files(files)
        items.append({
            "loc": base_url,  # exactly the base URL
            "lastmod": lastmod_root,
            "priority": args.priority_root,
            "sort_key": (0, ""),  # ensure root goes first
        })

    # File entries
    for path_abs in files:
        rel = rel_url_path(path_abs, root_dir)  # e.g., "page.html" or "blog/post.html"
        depth = rel.count("/")  # 0 for top-level, 1 for "a/b.html", etc.
        url = urljoin(base_url, rel)
        lastmod = file_lastmod_utc_str(path_abs)
        priority = guess_priority(rel, depth, args, args.lower_priority_glob or [])

        items.append({
            "loc": url,
            "lastmod": lastmod,
            "priority": priority,
            "sort_key": (1, rel.lower()),
        })

    # Sort: root first, then alphabetical
    items.sort(key=lambda x: x["sort_key"])

    # Build XML
    ns = {
        "": "http://www.sitemaps.org/schemas/sitemap/0.9",
        "xsi": "http://www.w3.org/2001/XMLSchema-instance",
    }
    ET.register_namespace("", ns[""])
    ET.register_namespace("xsi", ns["xsi"])

    urlset = ET.Element(
        ET.QName(ns[""], "urlset"),
        {
            ET.QName(ns["xsi"], "schemaLocation"): (
                "http://www.sitemaps.org/schemas/sitemap/0.9 "
                "http://www.sitemaps.org/schemas/sitemap/0.9/sitemap.xsd"
            )
        },
    )

    # Optional generator comment to mirror your example
    urlset.append(ET.Comment("  created with local sitemap generator  "))

    for it in items:
        url_el = ET.SubElement(urlset, ET.QName(ns[""], "url"))
        loc_el = ET.SubElement(url_el, ET.QName(ns[""], "loc"))
        loc_el.text = it["loc"]

        lastmod_el = ET.SubElement(url_el, ET.QName(ns[""], "lastmod"))
        lastmod_el.text = it["lastmod"]

        pr_el = ET.SubElement(url_el, ET.QName(ns[""], "priority"))
        pr_el.text = format_priority(it["priority"])

    tree = ET.ElementTree(urlset)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    # Write with XML declaration and UTF-8
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output, len(items)

def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Scan local folder for HTML files and generate sitemap.xml"
    )
    p.add_argument("--root-dir", default=".", help="Root directory to scan (default: current)")
    p.add_argument("--base-url", required=True, help="Base site URL, e.g. https://example.com/")
    p.add_argument("--output", default="sitemap.xml", help="Output sitemap path (default: ./sitemap.xml)")
    p.add_argument("--include-root", action="store_true", help="Include root '/' URL entry")
    p.add_argument("--priority-root", type=float, default=1.00, help="Priority for root URL (default: 1.00)")
    p.add_argument("--priority-default", type=float, default=0.80, help="Priority for normal pages (default: 0.80)")
    p.add_argument("--priority-deep", type=float, default=0.64, help="Priority for deep pages (default: 0.64)")
    p.add_argument("--deep-depth", type=int, default=2,
                   help="Depth threshold for 'deep' pages, counting slashes in relative path "
                        "(default: 2 -> only pages in subfolders, not top-level files)")
    p.add_argument("--lower-priority-glob", nargs="*", default=[],
                   help="Glob patterns to force deep priority, e.g.: report-*.html data-*.html")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    out, count = build_sitemap(
        root_dir=os.path.abspath(args.root_dir),
        base_url=args.base_url.strip(),
        output=os.path.abspath(args.output),
        include_root=args.include_root,
        args=args
    )
    print(f"Written {count} urls to {out}")

if __name__ == "__main__":
    main()
