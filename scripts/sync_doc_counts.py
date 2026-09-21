"""一次性把 README 里几处**手抄的测试数**同步到实际值。

为什么要写成脚本而不是手改：同一个文件里改 4 处，逐个手改很容易漏，
而漏掉的那处正是 `test_readme_test_counts_match_reality` 要抓的东西。
这里每处替换都**断言目标串存在** —— 找不到就炸，而不是静默少改一处。

`收集到的是 **NNN** 条` 那句不在护栏的正则里（护栏只钉目录树/路线图/输出行），
但它同样是写给读者看的断言，所以一起改。
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"

# 收集数从 pytest 现拿，不手抄 —— 手抄的那份就是上一轮过期的原因。
proc = subprocess.run(
    [sys.executable, "-m", "pytest", "tests", "--collect-only", "-p", "no:cacheprovider"],
    cwd=ROOT,
    capture_output=True,
    text=True,
)
m = re.search(r"(\d+) tests collected", proc.stdout)
if not m:
    sys.exit(f"数不出收集数：\n{proc.stdout[-800:]}\n{proc.stderr[-800:]}")
collected = int(m.group(1))

# 跳过数**不手抄** —— 从测试里那份"唯一真相来源"名单推导。
#
# 这里曾经写死 `skipped = 2`，还注释了那两条是谁。后来其中一条
# （慢速报告校验）不再默认跳过了，写死的 2 就变成假话，而
# `test_readme_test_counts_match_reality` 会红 —— 于是每次改跳过名单
# 都得记得回来改这个脚本。**能推导的别手抄。**
import importlib.util


def _default_skip_count() -> int:
    # `reports/` 不是包，直接跑脚本时 sys.path[0] 是 `reports/` 而不是仓库根 ——
    # 不插这一行，下面 exec_module 里的 `import npc_agent` 会炸。
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "_hygiene_under_sync", ROOT / "tests" / "test_test_hygiene.py"
    )
    if not (spec and spec.loader):
        sys.exit("加载不了 tests/test_test_hygiene.py —— 跳过数就推不出来了")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return int(mod._default_skip_count())


skipped = _default_skip_count()
passed = collected - skipped

text = README.read_text(encoding="utf-8")
edits = (
    (r"tests/\s+\d+ 个单元与端到端测试", f"tests/                  {collected} 个单元与端到端测试"),
    (r"六维评测 harness \+ \d+ 个测试", f"六维评测 harness + {collected} 个测试"),
    (r"# \d+ passed, \d+ skipped", f"# {passed} passed, {skipped} skipped"),
    (r"收集到的是 \*\*\d+\*\* 条", f"收集到的是 **{collected}** 条"),
    # 同一句正文里还印着"默认跳过 N 条"。它曾经**不在**这份清单里，
    # 于是收集数被同步了、跳过数留在原地 —— 一句没人守的假话。
    (r"默认跳过 \*\*\d+\*\* 条", f"默认跳过 **{skipped}** 条"),
)

for pattern, new in edits:
    hits = re.findall(pattern, text)
    if not hits:
        sys.exit(f"README 里找不到 {pattern!r} —— 正则过期了，先修这个脚本")
    text = re.sub(pattern, new, text)
    print(f"  {hits} -> {new.strip()}")

# `docs/ENGINEERING.md` 的「测试」一节印的是同一批数字，但它**没有任何东西在守**：
# 同步脚本（`_sync_handbook_counts.py`）管的是仓库外那份 HTML 面试手册，
# 这份 markdown 就烂在了 `# 656 passed, 2 skipped`（实际 970/1）而没人发现。
# 现在由 `tests/test_test_hygiene.py::test_the_handbook_test_section_matches_the_readme`
# 钉住；这里顺手一起改，免得每次都要手改两处。
HANDBOOK_MD = ROOT / "docs" / "ENGINEERING.md"
hb_all = HANDBOOK_MD.read_text(encoding="utf-8")
# ⚠️ 只改「## 测试」这一节。别处的散文**会引用**同样的形状（附十九里就写着
# "曾经印着 `# 656 passed, 2 skipped`），整份 sub 会把散文也改掉。
_sec = re.search(r"^## 测试\s*$(?P<body>.*?)(?=^## |\Z)", hb_all, re.M | re.S)
if not _sec:
    sys.exit("手册 markdown 里找不到「## 测试」这一节 —— 先修这个脚本")
hb = _sec.group("body")
handbook_edits = (
    (r"# \d+ passed, \d+ skipped", f"# {passed} passed, {skipped} skipped"),
    (
        r"收集到的是 \*\*\d+\*\* 条，默认跳过 \*\*\d+\*\* 条",
        f"收集到的是 **{collected}** 条，默认跳过 **{skipped}** 条",
    ),
)
for pattern, new in handbook_edits:
    hits = re.findall(pattern, hb)
    if not hits:
        sys.exit(f"手册「## 测试」一节里找不到 {pattern!r} —— 正则过期了，先修这个脚本")
    hb = re.sub(pattern, new, hb)
    print(f"  [ENGINEERING.md] {hits} -> {new}")
HANDBOOK_MD.write_text(
    hb_all[: _sec.start("body")] + hb + hb_all[_sec.end("body") :], encoding="utf-8"
)

README.write_text(text, encoding="utf-8")
print(f"\n收集 {collected} / 通过 {passed} / 跳过 {skipped} —— README 已同步")
