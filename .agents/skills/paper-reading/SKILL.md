---
name: paper-reading
description: Systematically read and analyze an academic or technical research paper from a local file, upload, DOI, arXiv link, PDF URL, or paper webpage, then produce an evidence-aware technical reading note. Use for paper summaries, method or experiment deep-dives, novelty and claim evaluation, research-oriented critiques, and follow-up questions about a paper; do not use for merely formatting an existing summary.
---

# Paper Reading

Turn the primary paper into a concise, technically useful reading note centered on its actual contribution. Prefer understanding and verification over section-by-section paraphrase.

## Inputs and mode

Use the supplied `title` and `source` when available. A source may be a local or uploaded PDF, arXiv or DOI URL, direct PDF URL, or paper webpage. If only one is supplied, use it to identify the paper. Ask for missing input only when the paper cannot be identified or accessed safely without it.

Infer the requested mode from the user's wording:

- `quick`: problem, main idea, actual contributions, headline evidence, and takeaways.
- `deep`: complete workflow and output below; this is the default, while keeping routine sections concise.
- `method`: emphasize architecture or algorithm, equations, objectives, training, inference, and data flow.
- `experiment`: emphasize datasets, splits, baselines, metrics, protocol, tables, ablations, and claim validity.
- `research`: emphasize novelty, predecessors, assumptions, limitations, open gaps, and relevance to the user's work.

Honor any requested format or emphasis over these defaults.

## Evidence discipline

Keep three kinds of statements visibly distinct:

- **Paper states:** content explicitly stated, shown, or claimed by the paper. Treat an author claim as a claim, not an independently established fact.
- **Inference:** a conclusion reasonably derived from the paper but not stated directly.
- **Assessment:** the agent's interpretation, comparison, or critique.

Use these labels inline whenever categories could be confused; do not clutter obvious factual metadata with labels. Attach page, section, figure, table, theorem, or equation references to important details when the source supports stable references. Mark uncertainty and extraction problems. Never invent metadata, values, equations, citations, or claims.

## Workflow

### 1. Acquire and verify the primary paper

1. Open the supplied local/uploaded file or retrieve the supplied URL. For a landing page, DOI, or arXiv abstract page, locate the corresponding paper PDF when possible. Prefer the actual paper and its supplement over summaries, blogs, or citation pages.
2. Verify the retrieved title against the user's title, allowing only harmless punctuation, subtitle, or capitalization differences. Cross-check authors and version when ambiguity remains. Do not silently substitute a similarly named paper.
3. Record the source actually read. Distinguish a preprint from a published version and note a meaningful version mismatch.
4. If the PDF text layer is incomplete, use page rendering or OCR where available and state any sections, equations, or tables that remain unreadable.
5. If the paper cannot be accessed, stop and report what failed, what was verified, and what source or upload would unblock the task. Do not reconstruct the paper from secondary summaries unless the user explicitly accepts that limitation.

### 2. First pass: establish the whole paper

Read the title, abstract, introduction, contribution statements, method/model, experiments or main results, and conclusion. Read related work at this stage only as needed to understand the claimed gap. Identify without drafting the final note yet:

- research problem, motivation, and difficulty;
- central idea and claimed contributions;
- evaluation or proof setup;
- headline results and conclusions.

### 3. Build a paper map

Before deep reading, form a working map:

- What problem is solved, and why does it matter?
- What existed before this work?
- What is the key new idea?
- What are the method's components and how do they interact?
- What assumptions and resources does it rely on?
- What evidence is offered for each major claim?

Use this map to allocate attention. Spend the most effort on novel, difficult, or claim-critical material rather than reading every section equally.

### 4. Deep-read the important material

For methods and systems:

- Trace the complete data flow from inputs through intermediate representations to outputs.
- Identify architecture or algorithm components, objectives, losses, training procedure, inference procedure, and computational requirements.
- Define important symbols, shapes, domains, and variables. For each important equation, explain every material term, its purpose, and its connection to the overall method in plain but precise language.
- Separate standard components, adaptations of prior work, and genuinely new mechanisms.

For theoretical work:

- Identify key definitions, assumptions, lemmas, theorems, guarantees, and counterexamples.
- Explain the proof strategy and why its major steps work; distinguish intuition from the formal result.
- Check the scope of quantifiers, conditions, and guarantees. Do not claim a proof is understood when a required step or notation is unavailable.

For empirical work:

- Identify datasets, collection or preprocessing, train/validation/test construction, baselines, metrics, hyperparameter-selection protocol, compute, and repeated-run or uncertainty reporting.
- Extract major quantitative results with table/figure references and preserve units, directionality, and comparison conditions.
- Read ablations and diagnostics as tests of specific design claims, not merely as extra results.
- Check for leakage, unfair baseline treatment, metric mismatch, cherry-picked subsets, and unsupported generalization when relevant.

Read appendices or supplements when they contain details necessary to understand the method, proof, experimental validity, or reproducibility. Skip routine implementation details unless they affect the claims.

### 5. Reconstruct the contribution

After understanding both method and evidence, determine:

- exactly what is novel;
- which elements are standard or inherited;
- which elements modify prior work;
- what capability, result, or artifact would be missing without this paper;
- the simplest accurate description of the contribution.

Do not repeat the authors' marketing language without analysis. When novelty depends on prior work not available in the paper, qualify the assessment or consult primary prior sources if the task and available access permit.

### 6. Test claims against evidence

List the paper's major claims and pair each with the relevant experiment, ablation, theorem, proof, or analysis. Rate support as:

- **Strongly supports:** evidence directly tests the claim under credible conditions.
- **Partially supports:** evidence is relevant but limited in scope, controls, statistical strength, or generality.
- **Does not clearly establish:** evidence is absent, indirect, confounded, or mismatched to the claim.

Explain the rating briefly. Do not equate empirical improvement on selected benchmarks with universal superiority, or a theorem under assumptions with performance outside those assumptions.

### 7. Critique proportionally

Assess genuine strengths, weaknesses, hidden assumptions, confounders, missing baselines or ablations, external-validity limits, computational cost, and reproducibility. Ground criticism in the paper's goals and evidence. Do not manufacture weaknesses for balance, and do not demand experiments irrelevant to the stated scope.

### 8. Connect to the user's research when context exists

Use provided user context and relevant project context to identify reusable ideas, unsuitable components, plausible adaptations, useful baselines, datasets, metrics, and follow-up concepts. Make the connection concrete and label speculative proposals as assessment. Omit this section when no meaningful context is available.

For follow-up questions, reuse the established paper map and prior findings. Reopen only the sections or external sources needed for the new question rather than restarting the complete workflow.

## Reading note format

Use the following structure for `deep` mode. Adapt its depth to the paper type and user's request; omit inapplicable subsections rather than filling them with generic prose. In `quick` mode, normally retain sections 1–4, the key-result portion of 6, and section 11. In focused modes, keep enough summary and context to make the focused analysis intelligible.

```markdown
# Paper

**Title:**
**Authors:**
**Year / Venue:**
**Source:**

## 1. One-paragraph summary

## 2. Problem

- What problem does the paper solve?
- Why does it matter?
- What makes it difficult?

## 3. Main idea

## 4. Contributions

- Contribution — type: methodological | theoretical | empirical | system/design | dataset/benchmark

## 5. Method

## 6. Experiments

- Datasets
- Baselines
- Metrics
- Experimental setup
- Key results
- Ablations and what each tests

## 7. Evidence vs claims

## 8. Strengths

## 9. Limitations

## 10. Relation to prior work

## 11. Key takeaways

## 12. Questions after reading

## 13. Relevance to my work
```

In section 10, compare only the most important predecessors and explain the delta; do not produce a generic related-work survey. Give 3–7 memorable points in section 11. In section 12, list unresolved questions, unclear choices, or worthwhile investigations rather than questions already answered by the paper.

## Failure handling

- **Identity mismatch:** report the retrieved title/authors and request confirmation or a correct source.
- **Inaccessible source:** explain whether the failure is permissions, network access, missing file, paywall, malformed PDF, or unsupported format, and state the minimal remedy.
- **Partial paper:** analyze only what is available, name the missing material, and narrow conclusions accordingly.
- **Illegible equations or tables:** identify them precisely and do not guess their contents.
- **Insufficient prior-work evidence:** qualify novelty comparisons instead of presenting them as settled.
- **Ambiguous result:** preserve the ambiguity and explain what additional information would resolve it.

## Quality check before delivery

Confirm that:

- the title and version match the requested paper;
- the note is based primarily on the paper itself;
- major technical statements and numbers are traceable to the paper;
- symbols and important equations are interpreted, not merely copied;
- actual novelty is separated from standard machinery;
- each major claim is paired with and rated against evidence;
- explicit statements, inferences, and assessments are distinguishable;
- criticism is specific, proportionate, and supported;
- uncertainty and inaccessible material are disclosed;
- long passages are paraphrased rather than copied;
- the selected mode and any available research context are reflected in the note.
