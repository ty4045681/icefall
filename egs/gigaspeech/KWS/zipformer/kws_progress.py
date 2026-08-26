#!/usr/bin/env python3
# Copyright 2026 Xiaomi Corporation
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Optional Rich progress display and a private parent/child event protocol."""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, Mapping, Optional, TextIO

PROGRESS_EVENT_ENV = "ICEFALL_KWS_PROGRESS_EVENTS"
PROGRESS_EVENT_PREFIX = "__ICEFALL_KWS_PROGRESS__="


def progress_events_requested(
    environ: Optional[Mapping[str, str]] = None,
) -> bool:
    values = os.environ if environ is None else environ
    return values.get(PROGRESS_EVENT_ENV) == "1"


def emit_progress_event(
    *,
    completed: int,
    total: int,
    stream: Optional[TextIO] = None,
    **fields: Any,
) -> None:
    payload = {"completed": int(completed), "total": int(total), **fields}
    try:
        print(
            PROGRESS_EVENT_PREFIX
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            file=sys.stdout if stream is None else stream,
            flush=True,
        )
    except (OSError, ValueError):
        # Progress is best-effort and must never abort model inference.
        pass


def parse_progress_event(line: str) -> Optional[Dict[str, Any]]:
    if not line.startswith(PROGRESS_EVENT_PREFIX):
        return None
    try:
        payload = json.loads(line[len(PROGRESS_EVENT_PREFIX) :])
        completed = payload["completed"]
        total = payload["total"]
    except (KeyError, TypeError, json.JSONDecodeError):
        return None
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or isinstance(total, bool)
        or not isinstance(total, int)
    ):
        return None
    if completed < 0 or total < 0 or completed > total:
        return None
    return {**payload, "completed": completed, "total": total}


class ConsoleProgress:
    """A small Rich progress wrapper with a plain-text fallback."""

    def __init__(
        self,
        *,
        total: int,
        description: str,
        enabled: Optional[bool] = None,
        stream: Optional[TextIO] = None,
        unit: str = "items",
    ) -> None:
        self.total = max(0, int(total))
        self.description = description
        self.stream = sys.stderr if stream is None else stream
        try:
            is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        except Exception:
            is_tty = False
        self.enabled = is_tty if enabled is None else bool(enabled)
        self.unit = unit
        self.completed = 0
        self.status = ""
        self._is_tty = is_tty
        self._progress = None
        self._task_id = None
        self._plain = False
        self._last_plain_completed = -1
        self._last_plain_time = 0.0
        self._last_rich_refresh_time = 0.0

    def start(self) -> "ConsoleProgress":
        if not self.enabled or self._progress is not None or self._plain:
            return self
        try:
            from rich.console import Console
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                SpinnerColumn,
                TaskProgressColumn,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )

            console = Console(
                file=self.stream,
                force_terminal=self._is_tty,
                color_system="auto" if self._is_tty else None,
            )
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                TextColumn("{task.fields[status]}"),
                console=console,
                auto_refresh=False,
                transient=False,
                redirect_stdout=True,
                redirect_stderr=True,
            )
            self._progress.start()
            self._task_id = self._progress.add_task(
                self.description,
                total=self.total,
                completed=self.completed,
                status=self.status,
            )
            self._progress.refresh()
            self._last_rich_refresh_time = time.monotonic()
        except Exception:
            if self._progress is not None:
                try:
                    self._progress.stop()
                except Exception:
                    pass
                self._progress = None
                self._task_id = None
            self._plain = True
            self._print_plain(force=True)
        return self

    def update(
        self,
        *,
        completed: Optional[int] = None,
        advance: int = 0,
        status: Optional[str] = None,
    ) -> None:
        if completed is None:
            completed = self.completed + int(advance)
        self.completed = min(self.total, max(0, int(completed)))
        if status is not None:
            self.status = str(status)
        if not self.enabled:
            return
        if self._progress is None and not self._plain:
            self.start()
        if self._progress is not None:
            try:
                now = time.monotonic()
                refresh = (
                    self.completed == self.total
                    or now - self._last_rich_refresh_time >= 0.1
                )
                self._progress.update(
                    self._task_id,
                    completed=self.completed,
                    status=self.status,
                    refresh=refresh,
                )
                if refresh:
                    self._last_rich_refresh_time = now
            except Exception:
                try:
                    self._progress.stop()
                except Exception:
                    pass
                self._progress = None
                self._task_id = None
                self._plain = True
                self._print_plain(force=True)
        elif self._plain:
            self._print_plain(force=self.completed == self.total)

    def complete(self, status: Optional[str] = None) -> None:
        self.update(completed=self.total, status=status)

    def stop(self) -> None:
        if self._progress is not None:
            try:
                self._progress.stop()
            except Exception:
                pass
            finally:
                self._progress = None
                self._task_id = None
        elif self._plain and self.completed != self._last_plain_completed:
            self._print_plain(force=True)

    def track(self, iterable: Any) -> Any:
        self.start()
        try:
            for item in iterable:
                yield item
        finally:
            self.stop()

    def _print_plain(self, *, force: bool) -> None:
        now = time.monotonic()
        step = max(1, self.total // 100)
        if not force and self.completed < self._last_plain_completed + step:
            if now - self._last_plain_time < 5.0:
                return
        percent = 100.0 if self.total == 0 else 100.0 * self.completed / self.total
        suffix = " {}".format(self.status) if self.status else ""
        try:
            print(
                "{}: {}/{} {} ({:.1f}%){}".format(
                    self.description,
                    self.completed,
                    self.total,
                    self.unit,
                    percent,
                    suffix,
                ),
                file=self.stream,
                flush=True,
            )
        except Exception:
            self.enabled = False
            self._plain = False
            return
        self._last_plain_completed = self.completed
        self._last_plain_time = now

    def __enter__(self) -> "ConsoleProgress":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()
