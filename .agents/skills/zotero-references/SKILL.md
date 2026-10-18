---
name: zotero-references
description: Search the user's local Zotero collections, inspect citation metadata, and read original papers, sample reports and the submitted CA report on demand for FYP report writing. Access is read-only; do not inventory or summarize the whole library.
---

# Zotero References

Use the existing Zotero library as the starting reference source for the FYP
report on financial time-series representation learning. The verified data
directory is `/mnt/e/zotero` (Windows `E:\zotero`), with `zotero.sqlite` and
`storage/`. This is a local filesystem workflow; no Zotero API, plugin service,
network access or credentials are needed.

## Access and search

Run commands from the repository root using `.venv/bin/python3`, following the
project's `python-env` skill. All Zotero-access scripts belong inside this skill,
under `scripts/`; do not add them to project `src/` or experiment script folders.

```bash
# Collection hierarchy, stable keys, and direct membership counts under FYP.
.venv/bin/python3 .agents/skills/zotero-references/scripts/zotero_read.py collections

# Default: report collection and all descendants. Terms are ANDed, case-insensitive
# literal substrings of metadata/authors/tags; this does not search PDF contents.
.venv/bin/python3 .agents/skills/zotero-references/scripts/zotero_read.py search 'representation learning' --limit 10

# Narrow to a subcollection, or broaden to all FYP after a targeted search.
.venv/bin/python3 .agents/skills/zotero-references/scripts/zotero_read.py search --collection I27CFASA --limit 5
.venv/bin/python3 .agents/skills/zotero-references/scripts/zotero_read.py search 'prediction market' --collection IXACZ4EG --limit 10

# Full metadata and attachment relationships for a selected parent paper.
.venv/bin/python3 .agents/skills/zotero-references/scripts/zotero_read.py item 7HJZZELR

# Use the PDF attachment key returned by item, not the parent paper key.
.venv/bin/python3 .agents/skills/zotero-references/scripts/zotero_read.py text 9S9NLK6G --first 1 --last 3
```

`search --offset N` paginates a bounded result set; `--limit` is capped at 100.
Use pagination for a specific writing question, not to dump the library.
Selectors are collection keys or numeric IDs, not slash-separated paths.
Global options `--data-dir` and `--library-id` go before the subcommand; verified
defaults are `/mnt/e/zotero` and personal library `1`.

Prioritize the report collection `D8C8SZHT`, whose **actual stored name** is
`CA/Final Report Refefence`. The slash is part of one collection name, not a
separate `CA` parent. Do not rename or correct it in Zotero.
Read [references/verified-access.md](references/verified-access.md) for verified
subcollection keys, sample PDF checks, and database relationships when needed.

The owner describes the report collection as the principal source, incorporating
useful references from older groups. Broaden to `FYP/prediction market` and other
FYP collections when the question warrants it. Older finance, deep-learning,
lab, literature-review, awesome-finance-paper and NLP groups have lower priority,
not automatic exclusion. Verify actual items; a collection name is no evidence
of a paper's content, quality or relevance.

## Sample reports and submitted CA report

For report organization, presentation examples or continuity with last semester,
consult `FYP/past-work-and-guidelines`, collection key `RDTKNYX6` (ID 48).
Both currently listed reports have accessible PDFs, verified on 2026-10-18:

- Sample dissertation: **Indentation-Sensitive Parser for Rhombus**, Ahmad
  Mashfi Ishtiaque; parent `ASBL8ABH`, PDF attachment `2BYFITPI`, 22 pages.
  Its contents page is physical PDF page 4.
- Owner's submitted **Final Year Project CA Report**: **A Unified Representation
  Learning Framework for Multi-Task Time-Series Modeling in Event Prediction
  Markets**, Chen Yixun; parent `MXK42RBD`, PDF attachment `3ITWXQDD`, 26 pages.
  Covers are physical pages 1–2; introduction/objectives start on page 3 and
  literature review on page 5.

```bash
.venv/bin/python3 .agents/skills/zotero-references/scripts/zotero_read.py search --collection RDTKNYX6 --limit 10
.venv/bin/python3 .agents/skills/zotero-references/scripts/zotero_read.py item MXK42RBD
.venv/bin/python3 .agents/skills/zotero-references/scripts/zotero_read.py text 3ITWXQDD --first 1 --last 6
.venv/bin/python3 .agents/skills/zotero-references/scripts/zotero_read.py text 2BYFITPI --first 1 --last 4
```

Use the sample to inform structure, front matter, level of detail and presentation
when relevant. It is another student's dissertation on a different topic, not an
official template or evidence for financial/ML claims. The collection name alone
does not establish submission rules; inspect any actual guideline document before
treating it as authoritative. Write original prose and adapt structure to this FYP.

Use the CA report to recover the owner's earlier framing, terminology, literature
choices and submitted scope. Reconcile its proposed methods and objectives with
current authoritative project plans and verified experiment artifacts before
reusing them in the final report. Earlier proposals are not proof of implemented
work or current results. Follow references back to the original works in Zotero
and verify their supporting passages before carrying citations forward.

The PDFs identify both documents as student reports, although Zotero currently
classifies both parents as `journalArticle` and supplies no publication date.
The CA creator fields also split the cover name differently (`firstName=Chen`,
`lastName=Yixun`). Preserve source metadata unchanged; use the verified cover name
and document type if describing the report. Do not invent publication details
or use the sample/CA report as a substitute for primary academic references.
See [verified report access](references/verified-access.md#sample-and-ca-reports)
for exact attachment paths, stored keys and verification scope.

## Read before citing

1. Search for the argument or method needed by the current report section.
   Inspect candidate metadata and resolve the original attachment.
2. Match the PDF's title, authors and version to the selected record. Read the
   passages that support the intended claim, including method details, experiment
   conditions and limitations where relevant. For a full paper analysis, use the
   available `paper-reading` skill; routine citation checks do not require a full
   reading note.
3. Record the parent item key, citation key, PDF/version and supporting page,
   section, table or equation in the report's existing notes or writing context.
   Distinguish author claims from demonstrated results and your own inference.
4. Incorporate citations where they support the argument, using the report's
   existing citation style. Verify quantitative claims against the original
   table/text. Never infer a finding from a title, collection, abstract alone,
   cached summary, annotation or another paper's citation of it.

`text` reads only the selected PDF pages and prints to stdout. `pdftotext` and
`pdfinfo` are installed in the verified WSL environment. For larger passages,
redirect selected text to `/tmp/` and inspect it there. For scanned documents or
illegible tables/equations, render or OCR selected pages into `/tmp/` using
available tools. Do not place extracted text, OCR output or reader state inside
Zotero. Report inaccessible or unreadable evidence; do not guess it.

## Bibliography and keys

- Preserve creator order, corporate authors (`fieldMode=1`), title, date, venue,
  volume/issue/pages, DOI/URL and version as actually verified. Metadata may have
  malformed dates or describe a preprint while the PDF includes proceedings
  details. Resolve discrepancies for the report; never repair Zotero itself.
- Check the report's bibliography first and reuse its established key for the
  same work. The verified database exposes `citationKey` through
  `itemData -> fieldsCombined -> itemDataValues`; `item` returns that exact key
  as `citation_key_in_metadata`. It also recognizes pinned `Citation Key:` or
  `BibTeX Citation Key:` lines in `extra` if the field is absent.
- Do not generate an author/year guess when a stored key exists. If neither
  report nor metadata has a key, use the deterministic report-local fallback
  `zotero-L<libraryID>-<parentItemKey>`, checking for conflicts. This fallback is
  not an existing Zotero/Better BibTeX key. Keep a consistent mapping in the
  report bibliography; do not write it back to Zotero.
- Collection membership can repeat the same item. Deduplicate by library/item
  key, then DOI/title/version when separate Zotero records describe one work.
  Keep distinct editions or substantively different versions explicit.
- Generate or update report bibliography entries only for works actually used,
  in the report's established destination and format. No bulk export or library
  pre-summarization. Do not invent missing authors, publication details or DOIs.

## Source preservation and failures

Never modify the Zotero database, collections, metadata, attachments, sidecars,
plugin files or preferences. The helper opens source database/WAL files only as
binary readers, copies them to a temporary `/tmp/zotero-read-*` directory, then
queries the copies with SQLite `mode=ro` and `PRAGMA query_only=ON`. It copies
the WAL so committed changes are visible, rebuilds SHM only in `/tmp/`, and
removes the temporary snapshot when finished. This mechanical metadata copy is
not an extraction or summary of the library.

The helper checks source size/timestamps/inode before and after copying and
retries up to three times if files change. If it still fails, retry when Zotero
is idle, or ask the owner to close Zotero. These checks reduce live-copy races;
they do not provide a cross-process atomic backup guarantee. Never bypass a
failure with `immutable=1` on the live DB: it ignores pending WAL transactions.
Never checkpoint, recover or change journal mode on the source database.

Stored attachments resolve through their **attachment** key:
`storage:<filename>` means `/mnt/e/zotero/storage/<attachmentKey>/<filename>`.
Windows absolute linked paths are translated to `/mnt/<drive>/...`. Relative
`attachments:` paths need a separately verified attachment base directory;
linked URLs have no local PDF. Report missing paths rather than substituting a
similarly named file or editing attachment relationships. If the library moves,
rediscover the exact location and update verified instructions after checking it.

Validate helper preservation behavior with:

```bash
.venv/bin/python3 .agents/skills/zotero-references/scripts/test_zotero_read.py
```
