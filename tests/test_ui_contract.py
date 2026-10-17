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


def test_desktop_smoke_matches_the_client_navigation_contract():
    """The Qt smoke hard-codes the nav count and per-view headings.

    CI's native step is the only place that executes it, so a stale constant
    (7 vs 6 after Configuration moved into Training) failed there and nowhere
    else. Pin the Python predicate to the client's own `names` map.
    """
    root = Path(__file__).parents[1]
    desktop = (root / "src/rocket_league_ai/desktop.py").read_text()
    app = (root / "src/rocket_league_ai/web/app.js").read_text()

    names_block = re.search(r"const names = \{(.*?)\n\};", app, re.S).group(1)
    nav = re.findall(r"^\s{2}(\w+):", names_block, re.M)
    assert nav == ["dashboard", "training", "monitoring", "models", "visualization", "benchmark"]

    expected = int(re.search(r"\[data-nav\]'\)\.length === (\d+)", desktop).group(1))
    assert expected == len(nav)

    routes = re.findall(r'\("(\w+)", "([^"]+)"\)', desktop)
    assert [route for route, _ in routes] == [*nav, "dashboard"]
    for _, heading in routes:
        assert f'"{heading}"' in app, f"desktop smoke expects heading {heading!r}, not present in app.js"
