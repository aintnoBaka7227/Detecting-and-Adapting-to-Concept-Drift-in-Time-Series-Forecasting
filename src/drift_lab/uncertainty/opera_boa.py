"""Bernstein Online Aggregation run by the R package opera.

AgACI's endpoint aggregator (Zaffran et al. 2022, Section 4 / Algorithm 1 /
Appendix C.1). The authors aggregate with

    opera::mixture(model = "BOA", loss.gradient = TRUE,
                   loss.type = list(name = "pinball", tau = ...))

and this module calls that same code instead of re-implementing it: one
background `Rscript` process (opera_boa.R) hosts every model, and each
`OperaBOA` talks to it over line-delimited JSON.

Requirements, on every machine that runs AgACI:

    R            on PATH as `Rscript` (macOS: `brew install r`)
    opera 1.2.2  no longer on CRAN for current R; install the pinned commit:
                 Rscript -e 'install.packages(c("remotes", "jsonlite"));
                   remotes::install_github("Dralliag/opera@24070db")'

Feedback. Weights are read at issuance and stay fixed for every forecast
of that issuance. `update` then hands the fed-back rows to opera, whose BOA
loop processes them in order. With one row per update (the project's
half-hourly issuance) this is exactly opera's online loop; with several,
the state left behind is the same but the issued forecasts shared the
weights from before the block.

With a single expert there is nothing to aggregate: the weight is 1 and R
is not started.
"""

from __future__ import annotations

import atexit
import json
import shutil
import subprocess
import uuid
from pathlib import Path

import numpy as np

OPERA_VERSION = "1.2.2"
OPERA_COMMIT = "24070db124c14df022c9356a3b517599ffd95148"

_SCRIPT = Path(__file__).with_name("opera_boa.R")
_INSTALL_HINT = (
    "AgACI aggregates with the R package opera. Install R (`brew install r`) and then "
    "opera: Rscript -e 'install.packages(c(\"remotes\", \"jsonlite\")); "
    f'remotes::install_github("Dralliag/opera@{OPERA_COMMIT[:7]}")\''
)


class _OperaSession:
    """The one background R process shared by every OperaBOA."""

    def __init__(self) -> None:
        rscript = shutil.which("Rscript")
        if rscript is None:
            raise RuntimeError(f"Rscript not found. {_INSTALL_HINT}")
        self._process = subprocess.Popen(
            [rscript, "--vanilla", str(_SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        try:
            self.info = self._read()
        except RuntimeError as exc:
            raise RuntimeError(f"could not start opera in R ({exc}). {_INSTALL_HINT}") from exc

    def call(self, **request) -> dict:
        self._process.stdin.write(json.dumps(request) + "\n")
        self._process.stdin.flush()
        return self._read()

    def _read(self) -> dict:
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError("the R session ended unexpectedly")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise RuntimeError(f"opera: {reply.get('error')}")
        return reply

    def close(self) -> None:
        if self._process.poll() is None:
            self._process.stdin.close()
            self._process.wait(timeout=10)


_session: _OperaSession | None = None


def _get_session() -> _OperaSession:
    global _session
    if _session is None:
        _session = _OperaSession()
        atexit.register(_session.close)
    return _session


def opera_available() -> bool:
    """True when R and opera can be started on this machine."""
    try:
        _get_session()
    except RuntimeError:
        return False
    return True


def opera_backend() -> str:
    """The R and opera versions in use, for run logging."""
    info = _get_session().info
    return f"opera {info['opera']} ({info['r']})"


class OperaBOA:
    """opera's BOA weights over `n_experts` forecasts of one quantile.

    Parameters
    ----------
    n_experts : int
        Number of experts (AgACI candidates).
    tau : float
        Quantile level of the pinball loss, strictly between 0 and 1.
    """

    def __init__(self, n_experts: int, tau: float) -> None:
        if isinstance(n_experts, bool) or int(n_experts) != n_experts or n_experts < 1:
            raise ValueError("n_experts must be a positive integer.")
        tau = float(tau)
        if not 0 < tau < 1:
            raise ValueError("tau must be strictly between 0 and 1.")

        self.n_experts = int(n_experts)
        self.tau = tau
        self._weights = np.full(self.n_experts, 1 / self.n_experts)
        self._n_updates = 0
        self._key: str | None = None
        if self.n_experts > 1:
            self._key = uuid.uuid4().hex
            _get_session().call(op="init", key=self._key, tau=self.tau)

    @property
    def n_updates(self) -> int:
        return self._n_updates

    @property
    def weights(self) -> np.ndarray:
        """Weights the next aggregation uses (a copy); non-negative, sum to 1."""
        return self._weights.copy()

    def aggregate(self, experts) -> np.ndarray:
        """Weighted average of each row of expert forecasts. No state change."""
        return self._expert_rows(experts) @ self._weights

    def update(self, experts, y) -> None:
        """Feed back one block: `experts` has one row of expert forecasts per
        outcome in `y`, oldest first."""
        experts = self._expert_rows(experts)
        y = np.asarray(y, dtype=float)
        if y.shape != (len(experts),):
            raise ValueError("y must have one outcome per row of experts.")
        if not np.isfinite(y).all():
            raise ValueError("y must be finite.")

        if self._key is not None and len(y):
            reply = _get_session().call(
                op="update", key=self._key, experts=experts.tolist(), y=y.tolist()
            )
            self._weights = np.asarray(reply["weights"], dtype=float)
        self._n_updates += len(y)

    def close(self) -> None:
        """Release the model held in the R session."""
        # `_key` is missing when __init__ rejected its arguments.
        if getattr(self, "_key", None) is not None and _session is not None:
            try:
                _session.call(op="drop", key=self._key)
            except (RuntimeError, OSError, ValueError):
                pass
        self._key = None

    def __del__(self) -> None:
        self.close()

    def _expert_rows(self, experts) -> np.ndarray:
        experts = np.atleast_2d(np.asarray(experts, dtype=float))
        if experts.ndim != 2 or experts.shape[1] != self.n_experts:
            raise ValueError(f"experts must have {self.n_experts} forecasts per row.")
        if not np.isfinite(experts).all():
            raise ValueError("experts must be finite; replace infinite endpoints first.")
        return experts
