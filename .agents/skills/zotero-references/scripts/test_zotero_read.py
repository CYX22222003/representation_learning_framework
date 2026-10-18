"""Bounded checks of source preservation, WAL visibility and path handling."""

import hashlib
from pathlib import Path
import sqlite3
import tempfile
import unittest

from zotero_read import attachment_path, snapshot, subtree


class ReadOnlyAccessTests(unittest.TestCase):
    def test_wal_visibility_and_source_preservation(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            source = root / "zotero.sqlite"
            writer = sqlite3.connect(source)
            try:
                writer.execute("PRAGMA journal_mode=WAL")
                writer.execute("PRAGMA wal_autocheckpoint=0")
                writer.execute("CREATE TABLE sample(value TEXT)")
                writer.commit()
                writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                writer.execute("INSERT INTO sample VALUES ('committed only in WAL')")
                writer.commit()
                paths = [source, Path(str(source) + "-wal"), Path(str(source) + "-shm")]

                def fingerprints():
                    return [(p.stat().st_size, p.stat().st_mtime_ns,
                             hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]

                before = fingerprints()
                with snapshot(root) as reader:
                    self.assertEqual(reader.execute("SELECT value FROM sample").fetchone()[0],
                                     "committed only in WAL")
                    with self.assertRaises(sqlite3.OperationalError):
                        reader.execute("INSERT INTO sample VALUES ('forbidden')")
                    copied_path = Path(reader.execute("PRAGMA database_list").fetchone()[2])
                self.assertEqual(before, fingerprints())
                self.assertFalse(copied_path.exists())
            finally:
                writer.close()

    def test_missing_database_not_created(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                with snapshot(root):
                    pass
            self.assertEqual(list(root.iterdir()), [])

    def test_active_rollback_journal_rejected(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            (root / "zotero.sqlite").write_bytes(b"sample")
            (root / "zotero.sqlite-journal").write_bytes(b"active")
            with self.assertRaisesRegex(ValueError, "rollback journal"):
                with snapshot(root):
                    pass

    def test_paths(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as directory:
            root = Path(directory)
            folder = root / "storage" / "ABCD1234"
            folder.mkdir(parents=True)
            pdf = folder / "a paper.pdf"
            pdf.write_bytes(b"%PDF-")
            self.assertEqual(attachment_path(root, "ABCD1234", "storage:a paper.pdf"),
                             (str(pdf), None))
            self.assertIsNone(attachment_path(root, "ABCD1234", "storage:../../outside.pdf")[0])
            self.assertIn("Unresolved", attachment_path(root, "ABCD1234", "attachments:a.pdf")[1])
            self.assertEqual(attachment_path(root, "ABCD1234", r"E:\papers\a.pdf")[0],
                             "/mnt/e/papers/a.pdf")

    def test_collection_subtree_uses_keys(self):
        rows = [{"key": "ROOT", "collectionID": 1, "parentCollectionID": None},
                {"key": "CHILD", "collectionID": 2, "parentCollectionID": 1},
                {"key": "OTHER", "collectionID": 3, "parentCollectionID": None}]
        self.assertEqual(subtree(rows, "ROOT"), [1, 2])
        self.assertEqual(subtree(rows, "2"), [2])
        with self.assertRaises(ValueError):
            subtree(rows, "missing")


if __name__ == "__main__":
    unittest.main()
