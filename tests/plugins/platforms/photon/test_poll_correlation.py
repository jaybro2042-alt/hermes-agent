from __future__ import annotations

import json
import subprocess
from pathlib import Path


_MODULE = Path("plugins/platforms/photon/sidecar/poll-correlation.mjs").resolve()


def test_poll_option_guid_normalization_executes_real_sidecar_helper() -> None:
    harness = (
        f"import {{ pollGuidForOption }} from {json.dumps(_MODULE.as_uri())};\n"
        "const cases = {\n"
        "  pollId: pollGuidForOption({poll: {id: 'poll-a'}}),\n"
        "  pollGuid: pollGuidForOption({poll: {guid: 'poll-b'}}),\n"
        "  parentId: pollGuidForOption({parentId: 'poll-c'}),\n"
        "  encodedVote: pollGuidForOption({}, "
        "'p:0/ABC:+15551234567:option-1:selected:1770000000000'),\n"
        "  missing: pollGuidForOption({title: 'A'}),\n"
        "};\n"
        "process.stdout.write(JSON.stringify(cases));\n"
    )
    run = subprocess.run(
        ["node", "--input-type=module", "-e", harness],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout) == {
        "pollId": "poll-a",
        "pollGuid": "poll-b",
        "parentId": "poll-c",
        "encodedVote": "p:0/ABC",
        "missing": None,
    }
