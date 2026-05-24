"""SealedVault: write-once container for `sealed_test_final.json`.

PROPOSAL.html §6.5 line 1268: opened exactly once at the end of a Discovery
run, refuses any second write. No LLM ever sees the contents -- only the
Orchestrator and downstream developer-facing tooling do.

The "seal" is the file's *existence*. Once `sealed_test_final.json` exists
on disk, any further `open_and_write` raises `SealedVaultError`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_VAULT_PATH = Path("artifacts/sealed/sealed_test_final.json")


class SealedVaultError(RuntimeError):
    """Raised on any attempt to write after the vault is sealed."""


class SealedVault:
    def __init__(self, path: str | Path = DEFAULT_VAULT_PATH) -> None:
        self.path = Path(path)

    def is_sealed(self) -> bool:
        return self.path.exists()

    def open_and_write(self, payload: dict[str, Any]) -> Path:
        """Write `payload` to the vault. Raises if already sealed.

        The file is created with mode 0o444 (read-only) and includes a
        `sealed_at_utc` timestamp so the audit trail records when the seal
        was opened.
        """
        if self.is_sealed():
            raise SealedVaultError(
                f"SealedVault at {self.path} is already sealed; refusing second write"
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "sealed_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            **payload,
        }
        self.path.write_text(json.dumps(record, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        # Make it read-only so a casual `echo > file` doesn't silently clobber.
        self.path.chmod(0o444)
        return self.path

    def read(self) -> dict[str, Any]:
        if not self.is_sealed():
            raise SealedVaultError(f"SealedVault at {self.path} has not been sealed")
        return json.loads(self.path.read_text(encoding="utf-8"))
