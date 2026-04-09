"""Tests for extended tools (code executor, file ops)."""

from cosmos_agi.tools.code_executor import execute_python
from cosmos_agi.tools.file_ops import (
    WORKSPACE_ROOT,
    _resolve_safe_path,
    list_files,
    read_file,
    write_file,
)


class TestCodeExecutor:
    def test_simple_execution(self):
        result = execute_python("print('hello world')")
        assert result["returncode"] == "0"
        assert "hello world" in result["stdout"]

    def test_blocked_import(self):
        result = execute_python("import subprocess; subprocess.run(['ls'])")
        assert result["returncode"] == "-1"
        assert "BLOCKED" in result["stderr"]

    def test_timeout(self):
        result = execute_python("import time; time.sleep(10)", timeout=2)
        assert "timed out" in result["stderr"].lower()

    def test_syntax_error(self):
        result = execute_python("def broken(")
        assert result["returncode"] != "0"

    def test_computation(self):
        result = execute_python("print(sum(range(100)))")
        assert result["returncode"] == "0"
        assert "4950" in result["stdout"]


class TestFileOps:
    def test_path_safety(self):
        assert _resolve_safe_path("../../../etc/passwd") is None

    def test_normal_path(self):
        result = _resolve_safe_path("test.txt")
        assert result is not None
        assert str(WORKSPACE_ROOT.resolve()) in str(result)

    def test_write_and_read(self):
        write_file("test_output.txt", "Hello, CosmosAGI!")
        content = read_file("test_output.txt")
        assert content == "Hello, CosmosAGI!"

    def test_read_missing_file(self):
        result = read_file("nonexistent_file_xyz.txt")
        assert "ERROR" in result

    def test_list_files(self):
        write_file("list_test/a.txt", "a")
        write_file("list_test/b.txt", "b")
        result = list_files("list_test")
        assert "a.txt" in result
        assert "b.txt" in result
