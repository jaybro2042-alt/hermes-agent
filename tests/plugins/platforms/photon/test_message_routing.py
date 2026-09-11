from __future__ import annotations

import json
import subprocess
from pathlib import Path


_MODULE = Path("plugins/platforms/photon/sidecar/message-routing.mjs").resolve()


def test_sidecar_forwards_only_inbound_messages_and_outbound_poll_echoes() -> None:
    harness = (
        "import { shouldForwardSpectrumMessage } from "
        f"{json.dumps(_MODULE.as_uri())};\n"
        "const cases = {\n"
        "  inboundText: shouldForwardSpectrumMessage({direction: 'inbound', content: {type: 'text'}}),\n"
        "  inboundPoll: shouldForwardSpectrumMessage({direction: 'inbound', content: {type: 'poll'}}),\n"
        "  outboundPoll: shouldForwardSpectrumMessage({direction: 'outbound', content: {type: 'poll'}}),\n"
        "  outboundText: shouldForwardSpectrumMessage({direction: 'outbound', content: {type: 'text'}}),\n"
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
        "inboundText": True,
        "inboundPoll": True,
        "outboundPoll": True,
        "outboundText": False,
    }
