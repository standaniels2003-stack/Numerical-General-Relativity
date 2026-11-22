"""
Readable test output: a progress bar plus a one-line description of each test.

Each line shows the progress bar, the test count, the test's docstring
(or its name if it has none) and, once it finishes, the result and run time.
Run with `pytest -v` to get pytest's normal verbose output instead.
"""
import pytest

BAR_WIDTH = 20
RESULT_STYLE = {
    "passed": ("PASSED", {"green": True}),
    "failed": ("FAILED", {"red": True, "bold": True}),
    "skipped": ("SKIPPED", {"yellow": True}),
}


class ProgressReporter:
    def __init__(self, terminal):
        self.tw = terminal
        self.total = 0
        self.done = 0
        self.items = {}

    def bar(self):
        filled = round(BAR_WIDTH * self.done / self.total) if self.total else 0
        return "[" + "#" * filled + "-" * (BAR_WIDTH - filled) + "]"

    @pytest.hookimpl(trylast=True)
    def pytest_collection_finish(self, session):
        self.total = len(session.items)
        self.items = {item.nodeid: item for item in session.items}
        self.tw.write_line("")
        self.tw.write_line(f"Running {self.total} tests", bold=True)

    def pytest_runtest_logstart(self, nodeid, location):
        item = self.items.get(nodeid)
        doc = getattr(getattr(item, "function", None), "__doc__", None)
        desc = doc.strip().splitlines()[0] if doc else location[2]
        slow = " (slow)" if item is not None and item.get_closest_marker("slow") else ""
        self.tw.write(f"{self.bar()} {self.done + 1:>{len(str(self.total))}}/{self.total}  {desc}{slow} ... ")
        self.tw.flush()

    def pytest_runtest_logreport(self, report):
        # Report once per test: on the call phase, or on setup if it failed/skipped there.
        if report.when == "call" or (report.when == "setup" and not report.passed):
            self.done += 1
            word, style = RESULT_STYLE.get(report.outcome, (report.outcome.upper(), {}))
            self.tw.write(word, **style)
            self.tw.write_line(f" ({report.duration:.2f}s)")

    def pytest_report_teststatus(self, report, config):
        # Silence pytest's own per-test dots/letters; we print our own lines.
        if report.when == "call" or (report.when == "setup" and not report.passed):
            return report.outcome, "", ""
        return None


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    terminal = config.pluginmanager.get_plugin("terminalreporter")
    if terminal is None or config.option.verbose != 0 or config.option.collectonly:
        return
    terminal.showfspath = False
    config.pluginmanager.register(ProgressReporter(terminal), "grsim_progress")
