#!/usr/bin/env python3
"""Update Nav.tsx to route the 'Try Live Demo' button to /demo instead of #demo."""
from pathlib import Path

p = Path(r"c:\Users\ravik\OneDrive\Desktop\lip alphabet\frontend\src\components\Nav.tsx")
src = p.read_text(encoding="utf-8")

# Replace the desktop CTA
old1 = '<a className="btn btn-ghost nav-cta" href="#demo" onClick={() => setOpen(false)}>\n          Try Live Demo\n        </a>'
new1 = '<a className="btn btn-ghost nav-cta" href="/demo" onClick={() => setOpen(false)}>\n          Try Live Demo\n        </a>'
if old1 in src:
    src = src.replace(old1, new1, 1)
    print("Updated desktop CTA → /demo")
else:
    print("WARN: desktop CTA pattern not found")

# Replace the mobile CTA
old2 = '<a className="btn btn-solid" href="#demo" onClick={() => setOpen(false)}>\n          Try Live Demo\n        </a>'
new2 = '<a className="btn btn-solid" href="/demo" onClick={() => setOpen(false)}>\n          Try Live Demo\n        </a>'
if old2 in src:
    src = src.replace(old2, new2, 1)
    print("Updated mobile CTA → /demo")
else:
    print("WARN: mobile CTA pattern not found")

p.write_text(src, encoding="utf-8")
print("Nav.tsx updated")
