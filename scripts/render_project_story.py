"""把 `docs/PROJECT_STORY.md` 渲染成一份**自包含**的 HTML，方便阅读/演示。

## 为什么不把 HTML 放进 `docs/`

`tests/test_docs_freshness.py` 会把 `docs/*.html` **全部**当成报告来管 ——
每一个都必须登记进 `OFFLINE_REPORTS`（离线可复现）或 `SNAPSHOT_REPORTS`
（要模型 + 额度的一次跑批快照）。这份是**文档**，不是报告，不该混进去。

所以：**唯一真相来源是 `docs/PROJECT_STORY.md`**（随仓库提交、被 git 管），
HTML 只是它的一份渲染产物，写到仓库外面，随时可以重新生成。

用法：
    python scripts/render_project_story.py                    # 默认写到工作区根
    python scripts/render_project_story.py --out /tmp/x.html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "docs" / "PROJECT_STORY.md"
DEFAULT_OUT = ROOT.parent / "项目介绍与面试作战手册.html"

# 视觉语言和 `docs/*.html` 的报告保持一致（同一套 CSS 变量），
# 这样"项目文档"和"项目报告"看起来是一个东西。
CSS = """  :root { --ink:#1f2430; --muted:#6b7280; --line:#e5e7eb; --bg:#fff;
          --soft:#f7f8fa; --accent:#2f6fd0; }
  * { box-sizing: border-box; }
  body { margin:0; padding:40px 28px 64px; background:var(--bg); color:var(--ink);
         font:15px/1.75 -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
              "Hiragino Sans GB", "Microsoft YaHei", sans-serif; }
  .wrap { max-width:900px; margin:0 auto; }
  h1 { font-size:28px; margin:0 0 18px; letter-spacing:-0.01em; }
  h2 { font-size:20px; margin:44px 0 14px; padding-bottom:8px;
       border-bottom:2px solid var(--line); }
  h3 { font-size:16.5px; margin:30px 0 10px; }
  h4 { font-size:15px; margin:22px 0 8px; }
  blockquote { margin:14px 0; padding:10px 16px; background:var(--soft);
               border-left:4px solid var(--accent); border-radius:6px;
               color:#374151; }
  blockquote p { margin:6px 0; }
  table { width:100%; border-collapse:collapse; font-size:13.5px; margin:16px 0; }
  th, td { padding:9px 10px; border-bottom:1px solid var(--line); text-align:left;
           vertical-align:top; }
  th { background:var(--soft); font-weight:600; font-size:12.5px; color:#374151; }
  tbody tr:hover { background:#fbfcfe; }
  code { font-family:ui-monospace, Menlo, Consolas, monospace; font-size:13px;
         background:#f2f3f5; padding:1px 5px; border-radius:4px; }
  pre { background:var(--soft); border:1px solid var(--line); border-radius:8px;
        padding:14px 16px; overflow-x:auto; }
  pre code { background:none; padding:0; font-size:12.8px; line-height:1.6; }
  hr { border:0; border-top:1px solid var(--line); margin:36px 0; }
  a { color:var(--accent); }
  footer { margin-top:56px; padding-top:16px; border-top:1px solid var(--line);
           color:var(--muted); font-size:13px; }
  @media (prefers-color-scheme: dark) {
    :root { --ink:#e6e8ec; --muted:#9aa1ad; --line:#333842; --bg:#1b1e24;
            --soft:#23272f; --accent:#6ba3f0; }
    blockquote { color:#c6cad2; }
    th { color:#c6cad2; }
    code { background:#2a2f38; }
    tbody tr:hover { background:#20242b; }
  }
"""

PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>项目介绍与面试作战手册 —— ai-npc-agent</title>
<style>
{css}</style>
</head>
<body>
<div class="wrap">
{body}
<footer>
  由 <code>scripts/render_project_story.py</code> 从
  <code>docs/PROJECT_STORY.md</code> 渲染 —— <b>唯一真相来源是那份 Markdown</b>，
  改内容请改它再重新渲染。
</footer>
</div>
</body>
</html>
"""


def render(md_text: str) -> str:
    import markdown  # 延迟导入：没装这个包时，脚本的 --help 仍然能用

    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
        output_format="html5",
    )
    return PAGE.format(css=CSS, body=body)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="输出 HTML 路径")
    args = parser.parse_args()

    if not SOURCE.exists():
        print(f"找不到 {SOURCE}", file=sys.stderr)
        return 1

    html = render(SOURCE.read_text(encoding="utf-8"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8", newline="\n")
    print(f"[render] {SOURCE.name} → {out}  ({len(html)} 字符)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
