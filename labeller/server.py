"""저장소 루트에서도 `uv run labeller/server.py` 로 라벨러 실행."""
import subprocess
import sys
from pathlib import Path

training = Path(__file__).resolve().parents[1] / "training"
try:
    result = subprocess.run(
        ["uv", "run", "--project", str(training), "python", "-u",
         str(training / "labeller" / "server.py"), *sys.argv[1:]],
        cwd=training,
    )
    raise SystemExit(result.returncode)
except KeyboardInterrupt:
    raise SystemExit(130) from None
