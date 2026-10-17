"""Expose bounded CI results in GitHub Checks without extra token permissions.

The caller must retain the original test exit status. This only reports existing
JUnit evidence; it never changes a failure into a success or starts a learner.
"""

import platform
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def escaped(message: str) -> str:
    return message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def main():
    path = Path(sys.argv[1])
    title = sys.argv[2]
    if not path.is_file():
        print("::error title=Missing test evidence::" + escaped(str(path)), flush=True)
        return 1
    tree = ET.parse(path)
    suites = list(tree.iter("testsuite"))
    counts = {key: sum(int(suite.get(key, "0")) for suite in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    seconds = sum(float(suite.get("time", "0")) for suite in suites)
    summary = f"Python {platform.python_version()}; {counts}; {seconds:.3f}s"
    print(f"::notice title={title}::" + escaped(summary), flush=True)
    for case in tree.iter("testcase"):
        for failure in [*case.findall("failure"), *case.findall("error")]:
            # Local disposable CI data only; bound annotations to avoid log flooding.
            message = case.get("name", "unknown") + "\n" + (failure.text or failure.get("message", ""))
            print(f"::error title={title}::" + escaped(message[:12000]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
