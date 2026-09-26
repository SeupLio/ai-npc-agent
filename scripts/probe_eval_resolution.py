"""评测的分辨率：剂量-反应曲线（命令行入口）。

真正的逻辑在 `npc_agent/eval/resolution.py`（这样它才能被 `cli resolution`
和 `scripts/regen_docs.py` 共用，并被新鲜度护栏守住）。

    python scripts/probe_eval_resolution.py
    python scripts/probe_eval_resolution.py --json out.json --html out.html

**离线、确定性、0 次模型调用。**
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from npc_agent.eval.resolution import (  # noqa: E402
    render_resolution,
    render_resolution_html,
    sweep,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="评测的剂量-反应曲线（离线）")
    ap.add_argument("--json", dest="json_path", default="")
    ap.add_argument("--html", dest="html_path", default="")
    args = ap.parse_args()

    result = sweep()
    render_resolution(result)

    if args.json_path:
        pathlib.Path(args.json_path).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nJSON → {args.json_path}")
    if args.html_path:
        pathlib.Path(args.html_path).write_text(
            render_resolution_html(result), encoding="utf-8"
        )
        print(f"HTML → {args.html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
