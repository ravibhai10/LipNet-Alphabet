with open("app/main.py", "r", encoding="utf-8") as f:
    content = f.read()

target = """def _spa_response(path: str):
    # Try compiled React bundle first
    full = (REACT_BUILD / path.lstrip("/")).resolve()
    if _is_safe_subpath(full, REACT_BUILD) and full.exists() and full.is_file():
        mt = _media_type(full)
        return FileResponse(str(full), media_type=mt)
    idx = REACT_BUILD / "index.html"
    if idx.exists() and idx.is_file():
        return FileResponse(str(idx))"""

replacement = """def _spa_response(path: str):
    clean_path = path.lstrip("/")
    # Try compiled React bundle first
    if clean_path:
        full = (REACT_BUILD / clean_path).resolve()
        if _is_safe_subpath(full, REACT_BUILD) and full.exists() and full.is_file():
            mt = _media_type(full)
            return FileResponse(str(full), media_type=mt)
        if clean_path.startswith("assets/") or full.suffix in {".js", ".css", ".map", ".woff", ".woff2", ".ttf", ".png", ".jpg", ".jpeg", ".svg", ".ico", ".json"}:
            return Response("Asset not found", status_code=404, media_type="text/plain")

    idx = REACT_BUILD / "index.html"
    if idx.exists() and idx.is_file():
        return FileResponse(
            str(idx),
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )"""

assert target in content, "Target block not found in app/main.py"
content = content.replace(target, replacement, 1)

with open("app/main.py", "w", encoding="utf-8") as f:
    f.write(content)

print("[SUCCESS] Patched _spa_response in app/main.py")
