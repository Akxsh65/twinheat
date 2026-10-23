"""Deliberately failing test: proves CI goes red. Do not merge."""


def test_deliberate_failure() -> None:
    assert False, "deliberate failure to check that CI shows red"
