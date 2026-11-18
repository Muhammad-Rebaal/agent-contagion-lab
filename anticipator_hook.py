"""Runtime inter-agent message inspection and passive telemetry logging hook."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from config import default_config
from detectors import DetectionResult, scan_agent_output


@dataclass
class InterceptAlert:
    """Security alert emitted when canary tokens or suspicious directives cross agent boundaries."""
    alert_id: str
    timestamp: str
    source_agent: str
    destination_agent: str
    hop_index: int
    match_type: str
    matched_substring: Optional[str]
    message_length: int
    severity: str


class ContagionRuntimeMonitor:
    """Passive runtime observer monitoring agent message exchange and logging security alerts."""

    def __init__(
        self,
        canary_token: str = "CRITICAL_BYPASS",
        alert_log_path: Optional[Path] = None,
        quarantine_on_detection: bool = False
    ):
        """Initialize the monitor with a target canary token and optional quarantine enforcement."""
        self.canary_token = canary_token
        self.alert_log_path = alert_log_path or (default_config.output_dir / "runtime_monitor_alerts.jsonl")
        self.quarantine_on_detection = quarantine_on_detection
        self.alert_history: List[InterceptAlert] = []

    def inspect_message(
        self,
        source_agent: str,
        destination_agent: str,
        hop_index: int,
        content: str
    ) -> DetectionResult:
        """Inspect inter-agent text, log an alert if canary tokens are detected, and optionally quarantine."""
        detection = scan_agent_output(text=content, canary=self.canary_token)

        if detection.is_infected:
            alert = InterceptAlert(
                alert_id=f"alert_{int(time.time() * 1000)}",
                timestamp=datetime.now(timezone.utc).isoformat(),
                source_agent=source_agent,
                destination_agent=destination_agent,
                hop_index=hop_index,
                match_type=detection.match_type,
                matched_substring=detection.matched_substring,
                message_length=len(content),
                severity="HIGH" if detection.exact_match else "WARNING"
            )
            self._record_alert(alert)

            if self.quarantine_on_detection:
                raise PermissionError(
                    f"[CONTAGION MONITORED] Message quarantined from {source_agent} -> {destination_agent}. "
                    f"Contagion match: {detection.match_type}"
                )

        return detection

    def _record_alert(self, alert: InterceptAlert) -> None:
        """Append an interception alert record to the alert history and persist to JSONL."""
        self.alert_history.append(alert)
        self.alert_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.alert_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(alert), ensure_ascii=False) + "\n")


class LangGraphContagionCallback:
    """Callback adapter connecting runtime inspection to graph agent completion events."""

    def __init__(self, monitor: ContagionRuntimeMonitor):
        """Initialize the callback adapter with an active runtime monitor."""
        self.monitor = monitor

    def on_agent_finish(self, finish: Any, **kwargs: Any) -> None:
        """Intercept agent execution output on node completion and forward to monitor inspection."""
        output_text = str(finish.return_values.get("output", ""))
        self.monitor.inspect_message(
            source_agent=kwargs.get("name", "UnknownAgent"),
            destination_agent="NextHop",
            hop_index=kwargs.get("hop_index", -1),
            content=output_text
        )
