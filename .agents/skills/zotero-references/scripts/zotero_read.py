#!/usr/bin/env python3
"""On-demand Zotero access; source files are opened only for binary reading."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile


def signature(path):
    try:
        stat = path.stat()
        return (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    except FileNotFoundError:
        return None


@contextmanager
def snapshot(data_dir):
    """Copy DB + WAL during a stable interval; SQLite touches only temp files.

    immutable=1 on the live database would ignore pending WAL transactions.
    SHM is not copied: SQLite rebuilds it in the temporary directory.
    """
    source = data_dir / "zotero.sqlite"
    if not source.is_file():
        raise ValueError(f"Database not found: {source}")
    paths = [source, Path(str(source) + "-wal"), Path(str(source) + "-journal")]
    with tempfile.TemporaryDirectory(prefix="zotero-read-", dir="/tmp") as temp:
        target = Path(temp) / source.name
        for _ in range(3):
            before = [signature(path) for path in paths]
            if before[2] and before[2][1]:
                raise ValueError("Active rollback journal; retry after Zotero is idle/closed.")
            for path, state in zip(paths[:2], before[:2]):
                dest = Path(temp) / path.name
                dest.unlink(missing_ok=True)
                if state is not None:
                    try:
                        with path.open("rb") as src, dest.open("wb") as dst:
                            shutil.copyfileobj(src, dst)
                    except FileNotFoundError:
                        break
            else:
                if before == [signature(path) for path in paths]:
                    break
        else:
            raise ValueError("Zotero changed while copying; retry when idle/closed.")
        connection = sqlite3.connect(target.as_uri() + "?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            yield connection
        finally:
            connection.close()


def collections(db, library_id):
    rows = [dict(row) for row in db.execute("""
        SELECT c.collectionID, c.key, c.collectionName, c.parentCollectionID,
          (SELECT count(*) FROM collectionItems ci JOIN items i USING(itemID)
           JOIN itemTypesCombined t USING(itemTypeID)
           WHERE ci.collectionID=c.collectionID
           AND t.typeName NOT IN ('attachment','note','annotation')
           AND NOT EXISTS(SELECT 1 FROM deletedItems d WHERE d.itemID=i.itemID))
          AS direct_items
        FROM collections c WHERE c.libraryID=? AND NOT EXISTS
          (SELECT 1 FROM deletedCollections d WHERE d.collectionID=c.collectionID)
    """, (library_id,))]
    by_id = {row["collectionID"]: row for row in rows}

    def path(row, seen):
        ident = row["collectionID"]
        if ident in seen:
            raise ValueError("Cycle in collection hierarchy")
        parent = by_id.get(row["parentCollectionID"])
        return (path(parent, seen | {ident}) + "/" if parent else "") + row["collectionName"]

    for row in rows:
        row["path"] = path(row, set())
    return rows


def subtree(rows, selector):
    matches = [r for r in rows if r["key"] == selector or str(r["collectionID"]) == selector]
    if len(matches) != 1:
        raise ValueError(f"Unknown/ambiguous collection key or ID: {selector}")
    ids = {matches[0]["collectionID"]}
    while True:
        expanded = ids | {r["collectionID"] for r in rows if r["parentCollectionID"] in ids}
        if ids == expanded:
            return sorted(ids)
        ids = expanded


def fields(db, item_id):
    return dict(db.execute("""
        SELECT f.fieldName, v.value FROM itemData d
        JOIN fieldsCombined f USING(fieldID) JOIN itemDataValues v USING(valueID)
        WHERE d.itemID=?
    """, (item_id,)))


def creators(db, item_id):
    return [dict(row) for row in db.execute("""
        SELECT c.firstName, c.lastName, c.fieldMode, t.creatorType, ic.orderIndex
        FROM itemCreators ic JOIN creators c USING(creatorID)
        JOIN creatorTypes t USING(creatorTypeID) WHERE ic.itemID=? ORDER BY ic.orderIndex
    """, (item_id,))]


def lookup(db, key, library_id):
    row = db.execute("""
        SELECT i.itemID, i.key, i.libraryID, t.typeName FROM items i
        JOIN itemTypesCombined t USING(itemTypeID) WHERE i.key=? AND i.libraryID=?
        AND NOT EXISTS(SELECT 1 FROM deletedItems d WHERE d.itemID=i.itemID)
    """, (key, library_id)).fetchone()
    if row is None:
        raise ValueError(f"Live item not found: library {library_id}, key {key}")
    return dict(row)


def attachment_path(data_dir, key, stored):
    if not stored:
        return None, "No local path (possibly a linked URL)"
    if stored.startswith("storage:"):
        root = (data_dir / "storage" / key).resolve()
        candidate = (root / stored.removeprefix("storage:")).resolve()
        if not candidate.is_relative_to(root):
            return None, "Invalid storage-relative path"
    elif re.match(r"^[A-Za-z]:[\\/]", stored):
        candidate = Path("/mnt") / stored[0].lower() / stored[3:].replace("\\", "/")
    elif stored.startswith("/"):
        candidate = Path(stored)
    else:
        return None, "Unresolved relative linked file; verify Zotero's attachment base directory"
    return str(candidate), None if candidate.is_file() else "File missing or inaccessible"


def attachments(db, data_dir, item_id):
    result = []
    for row in db.execute("""
        SELECT i.key, a.parentItemID, a.linkMode, a.contentType, a.path AS stored_path
        FROM itemAttachments a JOIN items i USING(itemID)
        WHERE (a.parentItemID=? OR a.itemID=?) AND NOT EXISTS
          (SELECT 1 FROM deletedItems d WHERE d.itemID=a.itemID) ORDER BY i.key
    """, (item_id, item_id)):
        entry = dict(row)
        entry["local_path"], entry["path_issue"] = attachment_path(
            data_dir, entry["key"], entry["stored_path"])
        result.append(entry)
    return result


def search(db, args, rows):
    ids = subtree(rows, args.collection)
    clauses = ["i.libraryID=?", "t.typeName NOT IN ('attachment','note','annotation')",
               "NOT EXISTS(SELECT 1 FROM deletedItems d WHERE d.itemID=i.itemID)",
               "EXISTS(SELECT 1 FROM collectionItems ci WHERE ci.itemID=i.itemID "
               f"AND ci.collectionID IN ({','.join('?' for _ in ids)}))"]
    params = [args.library_id, *ids]
    for term in args.query.split():
        clauses.append("""(EXISTS(SELECT 1 FROM itemData d
            JOIN itemDataValues v USING(valueID) JOIN fieldsCombined f USING(fieldID)
            WHERE d.itemID=i.itemID AND f.fieldName IN
              ('title','shortTitle','abstractNote','DOI','publicationTitle','extra','citationKey')
              AND instr(lower(v.value),lower(?))>0)
            OR EXISTS(SELECT 1 FROM itemCreators ic JOIN creators c USING(creatorID)
              WHERE ic.itemID=i.itemID AND instr(lower(c.firstName || ' ' || c.lastName),lower(?))>0)
            OR EXISTS(SELECT 1 FROM itemTags it JOIN tags tg USING(tagID)
              WHERE it.itemID=i.itemID AND instr(lower(tg.name),lower(?))>0))""")
        params.extend([term] * 3)
    query = ("SELECT i.itemID, i.key, i.libraryID, t.typeName FROM items i "
             "JOIN itemTypesCombined t USING(itemTypeID) WHERE " + " AND ".join(clauses)
             + " ORDER BY i.key LIMIT ? OFFSET ?")
    result = []
    for row in db.execute(query, [*params, args.limit, args.offset]):
        entry = dict(row)
        metadata = fields(db, row["itemID"])
        entry.update({k: metadata.get(k) for k in ("title", "date", "DOI")})
        entry["creators"] = creators(db, row["itemID"])
        result.append(entry)
    return {"collection": args.collection, "includes_descendants": True,
            "limit": args.limit, "offset": args.offset, "items": result}


def item(db, args, rows):
    result = lookup(db, args.key, args.library_id)
    ident = result["itemID"]
    result["fields"] = fields(db, ident)
    result["creators"] = creators(db, ident)
    memberships = {r[0] for r in db.execute(
        "SELECT collectionID FROM collectionItems WHERE itemID=?", (ident,))}
    result["collections"] = [r for r in rows if r["collectionID"] in memberships]
    result["tags"] = [r[0] for r in db.execute(
        "SELECT name FROM itemTags JOIN tags USING(tagID) WHERE itemID=? ORDER BY name", (ident,))]
    result["attachments"] = attachments(db, args.data_dir, ident)
    metadata = result["fields"]
    pinned = re.findall(r"(?im)^\s*(?:BibTeX\s+)?Citation Key:\s*(\S+)\s*$", metadata.get("extra", ""))
    result["citation_key_in_metadata"] = metadata.get("citationKey") or (pinned[0] if pinned else None)
    result["report_key_fallback"] = f"zotero-L{args.library_id}-{args.key}"
    result["citation_key_note"] = (
        "Reuse the report's existing bibliography key first, otherwise use the stored metadata key. "
        "Fallback is a report-local identifier, not an existing Zotero/Better BibTeX key.")
    return result


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/e/zotero"))
    parser.add_argument("--library-id", type=positive, default=1)
    commands = parser.add_subparsers(dest="command", required=True)
    tree = commands.add_parser("collections", help="Collection hierarchy and direct item counts")
    tree.add_argument("--root", default="IXACZ4EG", help="Collection key/ID; default FYP")
    find = commands.add_parser("search", help="Bounded metadata search within a collection subtree")
    find.add_argument("query", nargs="?", default="", help="AND of whitespace-separated literal terms")
    find.add_argument("--collection", default="D8C8SZHT", help="Collection key/ID; default CA/final report")
    find.add_argument("--limit", type=positive, default=10)
    find.add_argument("--offset", type=int, default=0)
    detail = commands.add_parser("item", help="Metadata, authors, memberships, attachment paths")
    detail.add_argument("key")
    text = commands.add_parser("text", help="Read selected pages of one PDF attachment")
    text.add_argument("key", help="Attachment item key from item output")
    text.add_argument("--first", type=positive, default=1)
    text.add_argument("--last", type=positive, default=3)
    args = parser.parse_args()
    args.data_dir = args.data_dir.resolve()
    if args.command == "search" and (args.limit > 100 or args.offset < 0):
        parser.error("--limit must be 1..100 and --offset nonnegative")
    if args.command == "text" and args.last < args.first:
        parser.error("--last must be >= --first")
    try:
        with snapshot(args.data_dir) as db:
            rows = collections(db, args.library_id)
            if args.command == "collections":
                ids = subtree(rows, args.root)
                result = sorted((r for r in rows if r["collectionID"] in ids), key=lambda r: r["path"])
            elif args.command == "search":
                result = search(db, args, rows)
            elif args.command == "item":
                result = item(db, args, rows)
            else:
                entry = lookup(db, args.key, args.library_id)
                if entry["typeName"] != "attachment":
                    raise ValueError("text requires an attachment key, not its parent paper key")
                found = attachments(db, args.data_dir, entry["itemID"])
                attachment = next(a for a in found if a["key"] == args.key)
                if attachment["path_issue"]:
                    raise ValueError(attachment["path_issue"])
                pdf = Path(attachment["local_path"])
                with pdf.open("rb") as handle:
                    if handle.read(5) != b"%PDF-":
                        raise ValueError("Attachment does not have a PDF header")
                if not shutil.which("pdftotext"):
                    raise ValueError("pdftotext is unavailable; use a verified PDF reader")
                extracted = subprocess.run(
                    ["pdftotext", "-layout", "-f", str(args.first), "-l", str(args.last), str(pdf), "-"],
                    check=True, capture_output=True, text=True)
                print(extracted.stdout, end="")
                if extracted.stderr:
                    print(extracted.stderr, file=sys.stderr, end="")
                if not extracted.stdout.strip():
                    raise ValueError("No text extracted; render/OCR these pages outside Zotero")
                return
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, sqlite3.Error, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Zotero read failed: {error}\n")


if __name__ == "__main__":
    main()
