"""Pytest session-wide checks."""

from time import perf_counter

import pytest


_MAX_TEST_SUITE_SECONDS = 10.0
_NUM_SLOWEST_TESTS_TO_REPORT = 10
_test_suite_start = 0.0
_test_durations: dict[str, float] = {}


def pytest_sessionstart(session: pytest.Session) -> None:
    global _test_suite_start, _test_durations
    _test_suite_start = perf_counter()
    _test_durations = {}


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.when != "call":
        return
    _test_durations[report.nodeid] = report.duration


def pytest_sessionfinish(
    session: pytest.Session, exitstatus: int
) -> None:
    elapsed = perf_counter() - _test_suite_start
    if elapsed <= _MAX_TEST_SUITE_SECONDS:
        return

    terminalreporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if terminalreporter is not None:
        terminalreporter.write_sep("=", "test suite too slow")
        terminalreporter.write_line(
            f"The test suite took {elapsed:.2f}s, exceeding the "
            f"{_MAX_TEST_SUITE_SECONDS:.2f}s limit."
        )
        terminalreporter.write_line("Slowest tests:")
        for nodeid, duration in sorted(
            _test_durations.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:_NUM_SLOWEST_TESTS_TO_REPORT]:
            terminalreporter.write_line(f"  {duration:.2f}s {nodeid}")

    session.exitstatus = pytest.ExitCode.TESTS_FAILED
