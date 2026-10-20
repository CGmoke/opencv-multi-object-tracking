"""工具层文件名清洗与父目录创建（步骤 6a 验收测试）。"""

from __future__ import annotations

from mot.utils.paths import ensure_parent_dir, sanitize_filename


class TestSanitizeFilename:
    def test_replaces_unsafe_characters(self):
        assert sanitize_filename("../../etc/passwd") == "etc_passwd"

    def test_keeps_allowed_characters(self):
        assert sanitize_filename("demo-01.mp4") == "demo-01.mp4"

    def test_empty_falls_back(self):
        assert sanitize_filename("", fallback="source") == "source"

    def test_is_bounded(self):
        assert len(sanitize_filename("a" * 300)) == 80


def test_ensure_parent_dir_creates_it(tmp_path):
    target = ensure_parent_dir(tmp_path / "a" / "b" / "file.txt")
    assert target.parent.is_dir()
    assert target.name == "file.txt"
