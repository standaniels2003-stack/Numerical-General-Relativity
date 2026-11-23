"""
Readable test output with a global progress bar.

Layout in a terminal:

     60% [=================>            ] 6/10 [00:01]

      ✓ PASSED     0.03s  Flat space (Cartesian): ...
      ✓ PASSED     0.04s  Flat space (spherical): ...
      ⠙ RUNNING    9.40s  Kerr (symbolic M, a): ...   <- running test, live timer

The bar at the top stays in place and keeps updating; each finished test
becomes a result line underneath it (descriptions come from the docstrings).

When output is not a terminal (piped, CI) or with `-s`, nothing is redrawn:
only the result lines are printed.
Run with `pytest -v` to get pytest's normal verbose output instead.
"""
import os
import shutil
import textwrap
import threading
import time

import pytest

BAR_MIN, BAR_MAX = 20, 70       # bar width adapts to the terminal within these bounds
REFRESH_SECONDS = 0.1
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

ESC = "\x1b["
CLEAR_LINE = "\r" + ESC + "2K"
CYAN, GREEN, YELLOW, RED, BOLD, DIM, RESET = (
    ESC + "36m", ESC + "32m", ESC + "33m", ESC + "31m", ESC + "1m", ESC + "2m", ESC + "0m"
)
# word, ANSI color (live mode), TerminalWriter markup (non-live mode)
RESULT_STYLE = {
    "passed": ("✓ PASSED ", GREEN, {"green": True}),
    "failed": ("✗ FAILED ", RED + BOLD, {"red": True, "bold": True}),
    "skipped": ("- SKIPPED", YELLOW, {"yellow": True}),
}


def describe(item, fallback):
    """
    Builds the one-line description shown for a test.

    Input:   the pytest item (a collected test) and a fallback name.
    Returns: the first line of the test function's docstring, or the fallback
             (the function name) if it has none, plus " (slow)" for tests marked slow.
    """
    doc = getattr(getattr(item, "function", None), "__doc__", None)
    desc = doc.strip().splitlines()[0] if doc else fallback
    if item is not None and item.get_closest_marker("slow"):
        desc += " (slow)"
    return desc


def section_title(item):
    """
    First line of the test module's docstring, plus its file name.
    """
    module = getattr(item, "module", None)
    doc = getattr(module, "__doc__", None)
    title = doc.strip().splitlines()[0].rstrip(".") if doc else ""
    path = item.location[0]
    return f"{title}  ({path})" if title else path


HEADER_TITLE = "test session starts"


def banner_lines(width, total, files, slow, kind="test"):
    """
    The start-of-run announcement: a full rule, then two pytest-style titled rules.

    kind is the word used for what runs: "test" (general tests) or
    "diagnostic" (when only diagnostics were selected); it is pluralised as needed.
    """
    tests = f"{total} {kind}{'s' * (total != 1)}"
    files = f"{files} file{'s' * (files != 1)}"

    def titled(text):
        """Returns text centered in a line of '=' signs, like pytest's own section headers."""
        text = f" {text} "
        left = max((width - len(text)) // 2, 3)
        return "=" * left + text + "=" * max(width - left - len(text), 3) + "\n"

    return [
        ("=" * width + "\n", {"bold": True}),
        (titled(HEADER_TITLE), {"bold": True}),
        (titled(f"starting {tests}  ·  {files}  ·  {slow} slow"), {"bold": True}),
    ]


def fit(text, width):
    """
    Cuts text to width characters so a line never wraps (wrapping breaks redraws).
    """
    return text if len(text) <= width else text[: max(width - 1, 0)] + "…"


class ProgressReporter:
    """
    pytest plugin that replaces the normal per-test output.

    In a terminal it draws a live, centered progress bar at the top, a header for
    each test file, one result line per finished test, and a RUNNING line with a
    timer for the test in progress. Without a terminal (piped output, CI) it only
    prints the headers and result lines.

    Parameters:
        terminal: pytest's TerminalReporter (used for normal, non-live output).
        tty: the terminal device opened for writing, or None for non-live mode.
    """
    def __init__(self, terminal, tty):
        """Stores the output streams and sets up the counters and redraw thread state."""
        self.tw = terminal._tw
        # pytest captures stdout while a test runs, so in a terminal the live
        # output is written straight to the terminal device instead.
        self.tty = tty
        self.live = tty is not None
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = None
        self.total = 0
        self.done = 0
        self.items = {}
        self.current = None          # description of the running test
        self.current_start = 0.0
        self.session_start = 0.0
        self.frame = 0
        self.lines_below_bar = 0     # newlines written since the bar line
        self.section = None          # test file of the last printed section header

    # ---- drawing (callers hold the lock) ---------------------------------------------
    def size(self):
        """Returns the terminal size (columns, lines), read from the terminal device itself."""
        # Ask the terminal device itself: stdout may be captured by pytest.
        try:
            size = os.get_terminal_size(self.tty.fileno())
            if size.columns > 0 and size.lines > 0:
                return size
        except (OSError, ValueError):
            pass
        return shutil.get_terminal_size()

    def bar_text(self):
        """
        The progress bar, centered in the terminal width.
        """
        frac = self.done / self.total if self.total else 1.0
        elapsed = int(time.time() - self.session_start)
        percent = f"{frac:4.0%} "
        # Fixed-width count so the bar doesn't shift as the numbers grow.
        count = f" {self.done:>{len(str(self.total))}}/{self.total}  "
        clock = f"[{elapsed // 60:02d}:{elapsed % 60:02d}]"
        columns = self.size().columns
        side = len(percent) + 2 + len(count) + len(clock)      # 2 = the brackets
        width = max(BAR_MIN, min(BAR_MAX, columns - 1 - side - 8))
        filled = int(width * frac)
        if filled >= width:
            inner = "=" * width
        else:
            inner = "=" * filled + ">" + " " * (width - filled - 1)
        pad = " " * max((columns - 1 - side - width) // 2, 0)
        return (f"{pad}{BOLD}{percent}{RESET}[{CYAN}{inner}{RESET}]"
                f"{count}{DIM}{clock}{RESET}")

    def running_text(self):
        """Returns the cyan RUNNING line (spinner, elapsed time, description), cut to fit one line."""
        spin = SPINNER[self.frame % len(SPINNER)]
        line = f"  {spin} RUNNING {time.time() - self.current_start:7.2f}s  {self.current}"
        return CYAN + fit(line, self.size().columns - 1) + RESET

    def bar_reachable(self):
        """True if the bar line is still on screen, so the cursor can move up to redraw it."""
        # The cursor can only move up to lines that are still on screen.
        return 0 < self.lines_below_bar < self.size().lines

    def redraw(self):
        """
        Updates the top bar in place and redraws the running-test line
        ."""
        out = ""
        if self.bar_reachable():
            n = self.lines_below_bar
            out += f"\r{ESC}{n}A{CLEAR_LINE}{self.bar_text()}\r{ESC}{n}B"
        out += CLEAR_LINE
        if self.current is not None:
            out += self.running_text()
        self.tty.write(out)
        self.tty.flush()

    def write_line(self, text):
        """
        Writes one finished line under the bar (replacing the running line).
        """
        self.tty.write(CLEAR_LINE + text + "\n")
        self.lines_below_bar += 1

    def ticker(self):
        """Background thread: redraws the bar and the RUNNING line every REFRESH_SECONDS until stopped."""
        while not self.stop.wait(REFRESH_SECONDS):
            with self.lock:
                self.frame += 1
                self.redraw()

    # ---- pytest hooks ----------------------------------------------------------------
    @pytest.hookimpl(trylast=True)
    def pytest_collection_finish(self, session):
        """
        pytest hook, called once all tests are collected.

        Counts the tests, files and slow tests, prints the start announcement
        (the three '=' lines) and, in a terminal, draws the first bar and starts the
        redraw thread.
        """
        self.total = len(session.items)
        self.items = {item.nodeid: item for item in session.items}
        self.session_start = time.time()
        files = len({item.location[0] for item in session.items})
        slow = sum(1 for item in session.items if item.get_closest_marker("slow"))
        width = self.tw.fullwidth      # same width as pytest's own "=== ... ===" lines
        only_diagnostics = bool(session.items) and all(
            item.get_closest_marker(DIAGNOSTICS) for item in session.items)
        kind = "diagnostic" if only_diagnostics else "test"
        self.tw.line("")
        for text, markup in banner_lines(width, self.total, files, slow, kind):
            self.tw.write(text, **markup)
        if self.live:
            self.tw.line("")      # space between the start line and the bar
        self.tw.flush()
        if self.live and self.total:
            with self.lock:
                self.tty.write(self.bar_text() + "\n")
                self.lines_below_bar = 1
                self.tty.flush()
            self.thread = threading.Thread(target=self.ticker, daemon=True)
            self.thread.start()

    def pytest_runtest_logstart(self, nodeid, location):
        """
        pytest hook, called when a test starts.

        Prints the file header if this test is in a new file, and shows the RUNNING line.
        """
        with self.lock:
            item = self.items.get(nodeid)
            if item is not None and location[0] != self.section:
                self.section = location[0]
                title = section_title(item)
                if self.live:
                    self.write_line("")
                    for line in textwrap.wrap(title, self.size().columns - 3, subsequent_indent="  "):
                        self.write_line(f"  {BOLD}{line}{RESET}")
                else:
                    self.tw.line("")
                    self.tw.line(f"  {title}", bold=True)
            self.current = describe(self.items.get(nodeid), location[2])
            self.current_start = time.time()
            if self.live:
                self.redraw()

    def pytest_runtest_logreport(self, report):
        """
        pytest hook, called after each phase (setup/call/teardown) of a test.

        Once per test (after the call phase, or after setup if that failed or skipped)
        it counts the test as done, prints its PASSED / FAILED / SKIPPED line with run
        time and description (wrapped if too long), and redraws the bar.
        """
        # Report once per test: on the call phase, or on setup if it failed/skipped there.
        if not (report.when == "call" or (report.when == "setup" and not report.passed)):
            return
        with self.lock:
            self.done += 1
            word, color, markup = RESULT_STYLE.get(report.outcome, (report.outcome.upper(), "", {}))
            rest = f" {report.duration:7.2f}s  {self.current}"
            self.current = None
            if self.live:
                # Long descriptions wrap onto indented continuation lines.
                indent = 2 + len(word) + len(f" {0:7.2f}s  ")
                lines = textwrap.wrap(f"  {word}{rest}", self.size().columns - 1,
                                      subsequent_indent=" " * indent)
                self.write_line(f"  {color}{word}{RESET}{lines[0][2 + len(word):]}")
                for line in lines[1:]:
                    self.write_line(line)
                self.redraw()
            else:
                self.tw.write("  ")
                self.tw.write(word, **markup)
                self.tw.write(rest + "\n")
                self.tw.flush()

    def pytest_report_teststatus(self, report, config):
        """
        pytest hook: hides pytest's own per-test letters/dots, since this plugin prints its own lines.

        Returns: (category, short letter, verbose word); the category and word are
                 still used for pytest's final summary (e.g. "FAILED tests/...").
        """
        # Silence pytest's own per-test dots/letters; we print our own lines.
        if report.when == "call" or (report.when == "setup" and not report.passed):
            return report.outcome, "", report.outcome.upper()
        return None

    @pytest.hookimpl(tryfirst=True)
    def pytest_sessionfinish(self, session):
        """pytest hook, called when all tests are done: stops the redraw thread and draws the final bar."""
        if self.thread is None:
            return
        self.stop.set()
        self.thread.join()
        with self.lock:
            self.current = None
            if self.bar_reachable():
                self.redraw()                         # final 100% bar, in place at the top
            else:
                self.write_line(self.bar_text())      # top scrolled away: repeat it here
            self.tty.flush()
        self.tty.close()


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    """
    pytest hook, called once at startup: installs the ProgressReporter.

    Skipped with -v (normal verbose output) or --collect-only. Opens the terminal
    device for live drawing if output goes to a terminal and -s is not used,
    hides pytest's own progress markers, and moves pytest's
    "=== test session starts ===" line into the start announcement.
    """
    terminal = config.pluginmanager.get_plugin("terminalreporter")
    if terminal is None or config.option.verbose != 0 or config.option.collectonly:
        return
    tty = None
    if terminal.isatty and config.option.capture != "no":
        try:
            tty = open("/dev/tty", "w")
        except OSError:
            pass
    terminal.showfspath = False
    # Leave out pytest's own "=== test session starts ===" at the top; it is
    # printed again as part of our announcement, right before the tests run.
    write_sep = terminal.write_sep

    def write_sep_without_header(sep, title=None, *args, **kwargs):
        """Like pytest's write_sep, but skips the "test session starts" line (we print it later)."""
        if title != HEADER_TITLE:
            write_sep(sep, title, *args, **kwargs)

    terminal.write_sep = write_sep_without_header
    # pytest's own "[100%]" marker is padded to the full terminal width and
    # would break the layout.
    terminal._show_progress_info = False
    config.pluginmanager.register(ProgressReporter(terminal, tty), "grsim_progress")


DIAGNOSTICS = "diagnostics"


def pytest_collection_modifyitems(config, items):
    """
    pytest hook, called after collection: keeps the diagnostics out of a normal run.

    What happens:
        Tests marked @pytest.mark.diagnostics are deselected, unless
          - the -m expression mentions "diagnostics" (e.g. pytest -m diagnostics,
            or pytest -m "diagnostics and not slow"), or
          - only diagnostics were collected (e.g. pytest tests/test_diagnostics.py).
        So `pytest` runs the general tests, `pytest -m diagnostics` the diagnostics.
        Deselected tests are reported by pytest as "N deselected".
    """
    if DIAGNOSTICS in (config.option.markexpr or ""):
        return
    diagnostics = [item for item in items if item.get_closest_marker(DIAGNOSTICS)]
    if not diagnostics or len(diagnostics) == len(items):
        return
    keep = [item for item in items if not item.get_closest_marker(DIAGNOSTICS)]
    config.hook.pytest_deselected(items=diagnostics)
    items[:] = keep
