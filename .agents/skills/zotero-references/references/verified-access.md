# Verified Zotero access — 2026-10-18

WSL directory `/mnt/e/zotero` maps to Windows `E:\zotero`.
Database: `/mnt/e/zotero/zotero.sqlite`; attachment root:
`/mnt/e/zotero/storage/`. Personal `libraryID=1`.
The main database, active `-wal`, `-shm` and attachment files were readable.
Only temporary copies were opened with SQLite; sources were binary-read only.

## Collection keys

These are verified live, non-deleted collections. Recheck with `collections`
if membership or names change. Counts are intentionally not maintained here.

| Collection | Key |
| --- | --- |
| FYP | `IXACZ4EG` |
| FYP / CA/Final Report Refefence | `D8C8SZHT` |
| ↳ alpha research | `IVK4DFGW` |
| ↳ deep learning | `WEJLFRLH` |
| ↳ prediction market | `JQ4LPMHN` |
| ↳ price volume modeling | `P8UCZYGL` |
| ↳ representation learning | `I27CFASA` |
| ↳ representation_learning + finance | `EXC5YSW3` |
| ↳ stronger baselines | `5HVHUJPR` |
| ↳ textbook | `P9JSMD6Z` |
| FYP / prediction market | `6YIPRHDQ` |
| FYP / awesome-finance-paper | `EJWJY456` |
| FYP / deep learning | `N5J7XXCK` |
| FYP / finance | `IHGCZWGQ` |
| FYP / lab-articles | `ZAVNHTB4` |
| FYP / literature reviews | `EGKXWMHM` |
| FYP / NLP | `D6AC7PKK` |
| FYP / past-work-and-guidelines | `RDTKNYX6` |

`CA/Final Report Refefence` is one name containing a slash and a spelling error.
It is a direct child of FYP, numeric collection ID 22. FYP is ID 1.
Do not infer additional hierarchy from that slash. A deleted older collection
also has the name `prediction market`; the helper excludes `deletedCollections`
and `deletedItems` rather than merging duplicate names.

## Verified PDF and key examples

Only first-page extraction and identity were checked for access verification;
these checks are not full methodology/findings reviews.

| Parent item key | Paper | Stored citation key | PDF attachment key |
| --- | --- | --- | --- |
| `7HJZZELR` | A Simple Framework for Contrastive Learning of Visual Representations | `chenSimpleFrameworkContrastive2020` | `9S9NLK6G` |
| `3R6QFSUD` | Divide and Contrast: Learning Robust Temporal Features without Augmentation | `shambaDivideContrastLearning2026` | `YBQ23E4V` |

First PDF:
`/mnt/e/zotero/storage/9S9NLK6G/Chen et al. - 2020 - A Simple Framework for Contrastive Learning of Visual Representations.pdf`.
Its first page matches Chen, Kornblith, Norouzi and Hinton, with
`arXiv:2002.05709v3`, dated 1 July 2020.

Second PDF:
`/mnt/e/zotero/storage/YBQ23E4V/Shamba et al. - 2026 - Divide and Contrast Learning Robust Temporal Features without Augmentation.pdf`.
Its first page matches Shamba, Bach and Taylor, with
`arXiv:2605.21241v1`, dated 20 May 2026.

Both parent records also have HTML attachments. Select the actual PDF attachment
using metadata/content and the file header, not the first attachment in a list.
The stored `citationKey` field was available for both examples. This workflow
does not depend on a Better BibTeX cache or live plugin API.

## Sample and CA reports

Verified on 2026-10-18 in `FYP/past-work-and-guidelines`, collection ID 48,
key `RDTKNYX6`. The collection currently lists these two parent records.
Metadata lookup, attachment resolution, `pdfinfo` and text extraction from
physical pages 1–6 succeeded for each. These were access/identity checks, not
complete reviews of either report. PDF page numbers below count from 1 and
can differ from the printed page labels.

| Role | Parent key | Stored citation key | PDF attachment | Pages |
| --- | --- | --- | --- | --- |
| Sample dissertation | `ASBL8ABH` | `ishtiaqueIndentationSensitiveParserRhombus` | `2BYFITPI` | 22 |
| Owner's submitted CA report | `MXK42RBD` | `yixunUnifiedRepresentationLearning` | `3ITWXQDD` | 26 |

**Sample dissertation:** *Indentation-Sensitive Parser for Rhombus*, Ahmad
Mashfi Ishtiaque. The cover identifies a B.Comp. Dissertation, Department of
Computer Science, School of Computing, National University of Singapore,
2025/2026. Physical pages 1–2 are covers, page 3 is the abstract, page 4 the
contents, and pages 5–6 start the introduction. Use as a structural example;
its subject is unrelated to this FYP's financial representation-learning claims.

Exact PDF path:
`/mnt/e/zotero/storage/2BYFITPI/Ishtiaque - Indentation-Sensitive Parser for Rhombus.pdf`.

**Submitted CA report:** *A Unified Representation Learning Framework for
Multi-Task Time-Series Modeling in Event Prediction Markets*, Chen Yixun.
The cover explicitly says “Final Year Project CA Report”, NUS School of
Computing, 2025/2026, project H398220, advisor Asst. Prof. Lu Yao. Physical
pages 1–2 are covers, pages 3–4 introduce the project and objectives, and
pages 5–6 begin the literature review. Earlier proposed scope must be reconciled
with current project plans/results before reuse.

Exact PDF path:
`/mnt/e/zotero/storage/3ITWXQDD/Yixun - A Unified Representation Learning Framework for Multi-Task Time-Series Modeling in Event Prediction.pdf`.

Both parents are stored as `journalArticle`, but their covers establish student
report types. Neither has a Zotero date field. The CA record stores creator
`firstName=Chen`, `lastName=Yixun`; the cover reads **Chen Yixun**. Do not silently
treat the imported record as a journal publication or infer a publication date
from a PDF creation timestamp. No Zotero metadata was corrected. The stored
citation keys above identify these records; their availability does not make
them mandatory academic citations in the final report.

## Verified database relationships

- `collections.parentCollectionID` builds the hierarchy; `libraryID` scopes keys.
- `collectionItems` links a collection to a bibliographic parent item.
- `items` plus `itemTypesCombined` identifies papers, books, webpages,
  attachments, notes and annotations. Exclude deleted records.
- `itemData` joins `fieldsCombined` by `fieldID` and `itemDataValues` by
  `valueID` for title, abstract, date, DOI, citationKey and other fields.
- `itemCreators` joins `creators` and `creatorTypes`; preserve `orderIndex`.
- `itemTags` joins `tags` for scoped search and item inspection.
- `itemAttachments.parentItemID` links the attachment to its parent. The
  attachment's own `items.key`, together with `itemAttachments.path`, resolves
  stored files. The parent's key is not the storage directory key.

## Validation

The helper successfully listed the FYP hierarchy, searched the report subtree,
retrieved both example records and their attachments, and extracted page 1 from
both original PDFs. Five focused helper checks passed, including visibility of
committed WAL-only data, rejection of writes, unchanged source DB/WAL/SHM
fingerprints in a controlled fixture, temporary snapshot cleanup, rollback-journal
refusal, missing-database handling, collection selection and path resolution.
