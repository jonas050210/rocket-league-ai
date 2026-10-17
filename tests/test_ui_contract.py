"""Static contract between the offline client and the HTTP surface.

This catches a class of defect that neither unit nor browser tests caught:
the client calling an endpoint that the backend never registered (the old
"Checkpoint league" button returned 404 while its service test passed).
"""

import re
from pathlib import Path

from rocket_league_ai.api import create_app


def client_paths():
    source = (Path(__file__).parents[1] / "src/rocket_league_ai/web/app.js").read_text()
    paths = set(re.findall(r'api\(\s*["\'`]([^"\'`$]+)', source))
    # Template literals such as evaluations/${id}/trajectory.
    paths |= set(re.findall(r'api\(\s*`([^`]+)`', source))
    cleaned = set()
    for path in paths:
        path = re.sub(r"\$\{[^}]+\}", "{}", path).split("?", 1)[0].strip("/")
        if path:
            cleaned.add(path)
    # A quoted capture of a template literal stops at "${"; keep the complete
    # template path and drop its bare prefix (e.g. "runs" vs "runs/{}/control").
    return {path for path in cleaned if not any(other.startswith(path + "/") for other in cleaned)}


def route_paths(app):
    paths = set()
    for route in app.routes:
        path = getattr(route, "path", "")
        if path.startswith("/api/"):
            paths.add(path[len("/api/"):].strip("/"))
    return paths


def matches(path, routes):
    parts = path.split("/")
    for route in routes:
        candidate = route.split("/")
        if len(candidate) != len(parts):
            continue
        if all(wanted in ("{}", actual) or actual.startswith("{") for wanted, actual in zip(parts, candidate, strict=True)):
            return True
    return False


def test_every_client_api_path_is_registered(tmp_path):
    app = create_app(tmp_path / "workspace", shutdown_timeout=1)
    routes = route_paths(app)
    missing = sorted(path for path in client_paths() if not matches(path, routes))
    assert not missing, f"client calls unregistered API paths: {missing}"


def test_league_routes_are_reachable(tmp_path):
    app = create_app(tmp_path / "workspace", shutdown_timeout=1)
    routes = route_paths(app)
    for path in ("leagues", "leagues/{}/stop"):
        assert matches(path, routes), path
