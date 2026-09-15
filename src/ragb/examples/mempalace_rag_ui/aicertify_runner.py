"""Small stdin/stdout worker used with an isolated AICertify environment."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import sys


def main() -> int:
    payload = json.load(sys.stdin)
    captured = io.StringIO()
    try:
        # AICertify and its evaluator dependencies may log to stdout. Keep the
        # worker protocol strictly JSON so the parent service can parse it.
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            from aicertify import application, regulations

            compliance_app = application.create(
                name=payload["application_name"],
                model_name=payload["model_name"],
                model_version=payload.get("model_version"),
                model_metadata=payload.get("model_metadata"),
            )
            compliance_app.add_interactions(payload["interactions"])
            regulation_set = regulations.create("mempalace_quivr")
            regulation_set.add(payload["policy"])
            results = asyncio.run(
                compliance_app.evaluate(
                    regulations=regulation_set,
                    generate_report=True,
                    report_format=payload["report_format"],
                    output_dir=payload["output_dir"],
                )
            )
        print(json.dumps({"results": results}, default=str))
        return 0
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
