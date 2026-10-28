# Design decisions

The decisions behind this prototype and the reason for each. Numbers are stable identifiers:
a new decision takes the next free number and goes into the section it belongs to.

## Scope and stack

1. **Local only, at zero cost.** Everything runs on one laptop (MacBook Air M2, 8 GB):
   Ollama for the models, chromadb on disk for the vectors and pypdf for PDF text. No cloud
   services, API keys or paid tools. No LangChain or LlamaIndex either, so that every
   pipeline stage is written, and measured, in this repository.
2. **Python 3.12 in a virtual environment.** macOS ships Python 3.9, which is too old for
   current libraries. Python 3.12 (Homebrew, native arm64) is supported by chromadb 1.5,
   pypdf 6 and the ollama client.
3. **Models: nomic-embed-text and llama3.2:3b.** The embedding model expects task prefixes:
   `search_document:` for chunks and `search_query:` for questions. llama3.2:3b is a 3B
   chat model that fits in 8 GB next to an editor. Both run fully on the GPU. With little
   free memory Ollama can unload one model to load the other, and one such reload took
   35 s, so Ollama is asked to keep models loaded for 10 minutes (`keep_alive`).
4. **chromadb telemetry is off.** chromadb sends anonymous usage data by default, which
   conflicts with running fully locally.

## Code structure

5. **Clean layers with an enforced dependency rule.** `domain` (data types, ports and errors)
   uses only the standard library. `chunking` and `services` depend only on `domain`.
   `adapters` wrap Ollama, chromadb and pypdf, and the CLI wires everything together. A unit
   test fails if a module breaks the rule.
6. **Fakes held to the same contracts as the real adapters.** In-memory fakes keep unit tests
   fast and independent of Ollama. Shared contract tests run against both the fakes and the
   real adapters, so the fakes cannot drift from real behaviour.
7. **Tooling:** ruff for linting and formatting, mypy in strict mode, and pytest.
   `make check` runs all three. `make coverage` runs the tests again with a branch-coverage
   report; it is kept separate because measuring coverage while the PDF libraries load
   turns a one-second test run into a 15-second one.
8. **Strict configuration.** Every key in `config.toml` is required, unknown keys are
   rejected with a did-you-mean hint, and values are type- and range-checked, so a typo
   fails loudly instead of falling back to a default.
9. **No silent failures.** Expected failures raise a specific error with a hint for the
   user; anything else crashes with a traceback. Text that is too long for the embedding
   model raises an error rather than being cut off, and so does a prompt too long for the
   chat model's context window, or an answer that runs out of room (decision 41).

## Documents and metadata

10. **Chunks never cross a page or section boundary.** Loaders produce segments: one PDF
    page (or the part of a page under one bookmark), or one markdown section. Chunks are cut
    from a single segment, so each chunk has exactly one page and one section. The cost is
    that a sentence running over a page break is cut at the break.
11. **Sources show the document name plus the page (PDF) or the heading (markdown).**
    Markdown has no pages, so success criterion 1 reads "document name plus page (PDF) or
    heading (markdown)". Labels are plain ASCII, for example
    `refund-policy.pdf, p. 3 (Refund window)` and `handbook.md (Time off > Vacation)`.
12. **PDF sections come only from bookmarks, never from guesses.** A page that lies entirely
    inside one bookmarked section gets that section's title, and so does text at the top of
    a page that continues the previous page's section. A page on which a new section starts
    is split only at a line whose whole text equals the bookmark title (ignoring spacing);
    a title that appears only inside a sentence does not count. The part of the page in
    which an unfound section begins gets an empty section. Font sizes and heading-like lines
    are never used.
13. **Created dates record where they came from.** PDFs use their `/CreationDate` metadata,
    markdown files a `created:` line in their front matter, and the file's creation time on
    disk is the last resort. Every chunk records which of the three was used.
14. **Documents are identified by file name, which must be unique,** because the test set
    names the expected document by its file name.
23. **Page headers and footers stay in the extracted text.** Detecting and removing them would
    take heuristics. The cost is visible in the chunks: a footer can land in the middle of a
    sentence that continues on the next page.
24. **Markdown sections are heading paths below the document title,** such as
    `Vacation > How much vacation do I earn?`. A single level-1 heading at the top is the
    title: it is left out of deeper paths and labels only the text directly under it. Each
    segment starts with its heading, which often is the question an FAQ answers. A heading
    with only subheadings under it makes no segment, and code fences are never read as
    headings.
25. **What the loaders refuse.** Hidden files are ignored. Other file types are reported as
    skipped, with the reason. Encrypted PDFs, PDFs with no extractable text (scans) and files
    that cannot be read stop the ingest with an error that names the file.
26. **The disk is the last resort for created dates, and only its creation time counts.**
    When a file has no date of its own, its creation (birth) time on disk is used, where
    the operating system records one. The modification time is never used, because it
    changes with every edit; without a birth time, the date stays empty.

## Indexing

15. **Every ingest rebuilds the strategy's collection from scratch,** so chunks from edited
    or deleted documents cannot linger. Within one run, a repeated chunk id is an error
    (it means a chunking bug), never an overwrite.
16. **An index is searchable only after a finished rebuild.** The manifest, which records
    the chunking strategy and embedding model, is written when a rebuild finishes. A run
    that fails halfway leaves an index that refuses searches instead of one that silently
    misses documents. Searching with a different embedding model from the one the index
    was built with is an error.
17. **One index folder per chunking strategy,** such as `index/fixed-300-50/`, so
    strategies can be compared on the same questions without re-ingesting.
27. **Strategy 1 is `fixed-300-50`:** windows of whole words of at most 300 characters,
    overlapping by up to 50, cut inside each segment after collapsing whitespace. The
    corpus's 168 segments have a median length of 251 characters, so a common 800-character
    window would leave almost every section whole (only 4 are longer) and strategy 1 would
    be the same as section-based chunking. At 300 characters, 48 segments are split. On the
    sample corpus this gives 241 chunks.
28. **Embedding requests are explicit.** The model's task prefixes are set in `config.toml`
    next to the model name, because they belong to that model. Ollama would silently cut off
    text that is too long, so truncation is turned off and such text raises an error. Blank
    text is rejected before any request, and chunks are embedded 32 per request.
29. **How the chromadb index is stored.** Each strategy's folder holds `current.json` and one
    chromadb database per build, in a folder such as `build-20261028T104512123456-a3f9c2/`.
    `current.json` names the live build and records the manifest.
    - A rebuild first deletes `current.json`, so the index stops answering searches (decision
      16). It then deletes the earlier build folders, and fills a new one.
    - When the rebuild finishes, `current.json` is written to a temporary file in the same
      folder, flushed to disk, and renamed into place, so it is never half-written. A
      leftover temporary file from a crash is ignored, and the next rebuild removes it.
    - Every build gets a folder name chromadb has never opened. chromadb keeps each database
      it opens cached for the life of the process, so a client opened on a deleted and
      recreated folder would be handed the stale copy.
    - Only chromadb's public API is used: nothing reads its internal records. Folders other
      than build folders are never touched.
    - Collections use cosine distance, and a result's score is 1 minus the distance.
      chromadb ignores an add for an id it already stores, without any error, so the adapter
      checks ids first; it also rejects empty metadata values, so fields without a value
      are left out.
34. **Strategy 2 is `sentences-300`:** whole sentences packed into chunks of at most 300
    characters, inside each segment. It has the same size budget as strategy 1, so the two
    differ only in where the cuts fall: a sentence is never cut, which is why it needs no
    overlap. Blank lines, list items and table rows always start a new sentence; other line
    breaks count as spaces, because PDF text breaks lines mid-sentence. A sentence ends at
    ".", "!" or "?" followed by a space, except after abbreviations such as "a.m." or "Jr.".
    On the sample corpus this gives 240 chunks.
35. **Strategy 3 is `sections-4000`:** one chunk per segment, that is, per markdown section,
    per bookmarked part of a PDF page, or per whole page of a PDF without bookmarks. A
    segment longer than 4,000 characters would be split at sentence boundaries; that limit
    keeps chunks within what the embedding model accepts and is not a tuning knob (no
    segment of the sample corpus reaches it). On the sample corpus this gives 168 chunks,
    from about 250 characters for a typical section to about 2,200 for a page without
    bookmarks.

## Command line

30. **`rag ingest`, `search`, `ask`, `inspect`, `eval` and `answer-all`.** `ingest`
    rebuilds a strategy's index and reports chunk counts and embedding speed. `search` shows
    the top chunks for a question, with sources and scores. `ask` answers a question from
    the top chunks and shows the sources the answer cites (decision 43). `inspect` shows
    what the index stores, filtered by document and page. `eval` measures retrieval on the
    test set, and `answer-all` answers every test question into a file for grading
    (decision 45). Each command uses the strategy set in `config.toml` unless `--strategy`
    names another; `ingest` and `eval` also accept `--strategy all`, and `eval` then prints
    the comparison. Expected failures print one line, `error: ...`, plus a `hint: ...`
    line, and exit with status 1; results go to stdout and logs to stderr.

## Evaluation

18. **Retrieval is measured separately from generation.** A question counts as a hit when a
    chunk from the expected document is among the top 3 retrieved chunks. Whether answers
    are supported by their sources is graded separately (decision 44).
19. **Test set format:** `data/eval/questions.csv` with the columns
    `id,question,expected_doc,expected_answer`. `expected_answer` is optional and used only
    for grading answers. A row with an empty `expected_doc` has no answer in any document.
    These rows are left out of the 50 questions and the hit rate, and test whether the chat
    model says it doesn't know.
31. **The test questions were generated alongside the corpus, not written independently.**
    The 50 answerable questions, their reference answers and the 5 unanswerable questions
    were written together with the 20 documents, by the same author, with every document's
    wording known. To limit the bias this brings, the questions are phrased the way an
    employee or customer would ask, avoid the documents' section titles and distinctive
    phrases (no question shares more than four consecutive words with any document), and
    were written without running any retrieval, so they are not tuned to the index. The
    hit rate may still be higher than a set written by someone who had not read the
    documents would give, so such a set remains the stronger test.
    - The set covers every document with 2 or 3 questions each.
    - It deliberately includes the corpus's edge cases: retirement savings asked without
      naming the 401(k) plan, a travel-safety detail that runs across a page break, facts
      that only appear in FAQ tables, and the numbers 30 days, 14 days and $75 in their
      different contexts.
    - Four questions target facts that one document only points to; their expected
      document is the one that actually contains the fact.
32. **The test set is checked before anything is searched, and it is frozen.**
    `questions.csv` must have exactly the columns `id`, `question` and `expected_doc`, plus
    optionally `expected_answer`, and the ids must be unique; a byte-order mark is accepted.
    Every `expected_doc` must be a document in the index, or the evaluation stops and names
    the questions, because a misspelled file name would otherwise count as a retrieval miss.
    Questions are not changed after results have been seen.
33. **How retrieval results are reported.** All answerable questions are embedded and
    searched in one batch, keeping each question's top 5 chunks. The measure is the hit
    rate at 3 (decision 18); the hit rates at 1 and 5 are reported next to it as context.
    A hit is document-level: the chunk that hits does not have to be the one with the
    answer; answer coverage (decision 39) checks that. `results/retrieval-<strategy>.csv`
    holds one row per retrieved chunk (question, expected document, rank, retrieved
    document and its source label, score, and the question's hit and coverage) and is
    replaced by each run of that strategy.
36. **Strategies are compared on the same questions and adopted by a fixed rule.** Each
    strategy is evaluated on the same 50 answerable questions, reporting the hit rates at
    1, 3 and 5 and the mean reciprocal rank (the average of 1/rank of the first chunk from
    the expected document, 0 when it is not in the top 5). The strategy to adopt is the one
    with the highest hit rate at 3, as the charter specifies; ties are broken by the hit
    rate at 1, then by the mean reciprocal rank. `results/comparison.csv` lists the
    strategies in that order (`rule_rank`) and marks the one in use. With 50 questions, a
    difference of one or two questions is within noise, so the per-question results are
    kept for inspection. For the sample corpus, decision 40 overrides the tie-break.
37. **A single-chunk "answer hit" is not used to compare strategies.** Asking whether one of
    the top 3 chunks contains every fact of the expected answer cannot be compared fairly:
    answers whose facts span several sentences can never fit in one small chunk, so the
    best possible score differs by strategy (44, 41 and 49 of 50 for strategies 1, 2 and
    3) before retrieval plays any part, and q44's fact is split by a page break for all of
    them. Answer coverage, whether the top 3 chunks together contain every fact intact,
    can be compared fairly, because every strategy can reach 49 of 50. It was first
    measured once, for the strategy comparison, and is now part of every `rag eval` run
    (decision 39).
38. **The fact list is frozen like the questions.** `data/eval/facts.csv` has the columns
    `id,fact` and one row per fact: for each of the 50 answerable questions, the facts its
    expected answer relies on, worded exactly as the expected document states them. There
    are 78 facts, 1 to 4 per question. They were written with the questions (decision 31),
    before any answer was generated, and are not changed after results have been seen.
    Every answerable question needs at least one fact and the others none, and a fact may
    not be listed twice for a question; `rag eval` stops otherwise.
    - A test checks that every fact appears whole, ignoring case and spacing, inside a
      single segment of its expected document's extracted text, the most a chunk can hold.
    - The one allowed exception is q44's "Tell your manager about any change that keeps you
      away longer than planned": the page break, with the page 1 footer in between, splits
      it (decisions 10 and 23). The test fails if any other fact breaks, and also if this
      one stops breaking, so the exception cannot outlive its cause.
39. **Answer coverage is reported by every `rag eval` run.** A question is covered at n
    when each of its facts appears whole, ignoring case and spacing, in one of the top n
    chunks that come from the expected document; different facts may be in different
    chunks.
    - It is reported at 3, next to the hit rate it complements, and at 5, the depth kept
      per question. It is not reported at 1, which would be the single-chunk measure that
      decision 37 rejects.
    - `comparison.csv` has `covered_at_3` and `covered_at_5`. `retrieval-<strategy>.csv`
      has, per question, `covered_at_3` and `covered_rank`, the smallest n at which every
      fact is in.
    - The command line lists each question whose top 3 chunks miss a fact, with the facts
      that are missing.
    - q44 can never be covered (decision 38), so 49 of 50 is the most any strategy can reach.
40. **Strategy 3, `sections-4000`, is adopted, overriding the tie-break of decision 36.**
    `rag eval --strategy all` on the 50 answerable questions gives:

    | Strategy | Chunks | Hits at 1 | Hits at 3 | Hits at 5 | MRR | Covered at 3 | Covered at 5 |
    |---|---|---|---|---|---|---|---|
    | `fixed-300-50` | 241 | 43 | 48 | 50 | 0.917 | 38 | 42 |
    | `sentences-300` | 240 | 44 | 48 | 49 | 0.925 | 39 | 41 |
    | `sections-4000` | 168 | 41 | 48 | 50 | 0.888 | 44 | 48 |

    Misses at 3, with the rank of the first chunk from the expected document:

    | Strategy | Misses at 3 |
    |---|---|
    | `fixed-300-50` | q02 (rank 4), q15 (rank 4) |
    | `sentences-300` | q02 (rank 4), q50 (not in the top 5) |
    | `sections-4000` | q02 (rank 5), q04 (rank 5) |

    Questions whose top 3 chunks miss a fact, with the rank at which every fact is in:

    | Strategy | Not covered at 3 |
    |---|---|
    | `fixed-300-50` (12) | q01 (4), q02 (5), q05 (-), q15 (4), q21 (-), q22 (-), q31 (-), q32 (-), q33 (4), q39 (-), q44 (-), q50 (-) |
    | `sentences-300` (11) | q01 (-), q02 (-), q05 (-), q21 (-), q22 (-), q31 (-), q32 (-), q33 (4), q38 (5), q44 (-), q50 (-) |
    | `sections-4000` (6) | q02 (5), q04 (5), q22 (5), q32 (-), q33 (4), q44 (-) |

    "-" means not within the top 5. Only 9 questions rank their expected document
    differently between strategies:

    | Question | Expected document | `fixed-300-50` | `sentences-300` | `sections-4000` |
    |---|---|---|---|---|
    | q02 | `benefits-overview.pdf` | 4 | 4 | 5 |
    | q04 | `code-of-conduct.pdf` | 1 | 1 | 5 |
    | q05 | `code-of-conduct.pdf` | 1 | 1 | 2 |
    | q06 | `code-of-conduct.pdf` | 1 | 1 | 3 |
    | q11 | `data-retention-policy.pdf` | 1 | 1 | 2 |
    | q13 | `employee-handbook-time-off.md` | 2 | 2 | 1 |
    | q15 | `gear-rental-faq.md` | 4 | 1 | 3 |
    | q45 | `travel-and-expense-policy.pdf` | 2 | 2 | 1 |
    | q50 | `workplace-safety-and-incident-reporting.pdf` | 3 | - | 3 |

    The rule would adopt `sentences-300`. It is overridden because:
    - All three strategies tie at 48 of 50 hits at 3, the charter's measure.
    - `sentences-300` leads at 1 by one question, which is within noise on 50 questions
      (decision 36), and it is the only strategy that loses a question from the top 5
      (q50).
    - `sections-4000` has every fact of the answer in the top 3 chunks for 44 questions,
      against 38 and 39 (48 at 5, against 42 and 41). That matters most for answering: the
      chat model can only state a fact it is given.

    The rule itself is unchanged: `rag eval --strategy all` still prints the order it gives
    and then the strategy in use, and `comparison.csv` marks it. `sections-4000` ranks
    lower than both others only on q04, q05, q06 and q11, all answered on PDF pages
    without bookmarks, each indexed as one chunk of up to about 2,200 characters. A fourth
    strategy that splits such long pages into sentence windows might recover those; it is
    listed in the README as possible future work and not built.
44. **Answers are graded with a rubric fixed before any answer was generated.**
    Each of the 55 test questions gets two grades:
    - `supported` (yes/no): every statement in the answer, including its conclusions,
      follows from the sources it was given. A wrong conclusion drawn from the right source
      is not supported: an answer to q47 that treats 1.5 years as more than 2 years, as
      one sample answer did, is not supported.
    - `correct` (yes/partial/no): the answer compared with `expected_answer`. For u01 to
      u05, which no document answers, yes means the answer says "I don't know".

    How the rubric applies to two cases it does not name:
    - "The sources it was given" are all the chunks in the prompt, cited or not. Whether a
      citation number points at the right chunk is not graded.
    - An answer that only says "I don't know" makes no statement, so it is supported. For
      an answerable question it is not correct.

    This decision was recorded before any of the 55 answers was generated, so the rubric
    cannot bend to them. `results/grading-notes.md` says how the grades were made, how the
    rubric was applied, and which answers needed a judgment call.
45. **`rag answer-all` writes the answers for grading.** It answers every test question,
    the 50 answerable and the 5 unanswerable, exactly as `rag ask` would (same retrieval,
    prompt, top_k and temperature), and writes `results/answers.csv`, one row per question.
    - `answer` is the model's reply exactly as it came back. `supported` and `correct` are
      left empty for grading with the rubric of decision 44.
    - Next to them are `cited` (the cited sources with their labels), `prompt` (the
      numbered sources and the question exactly as the model got them, which `supported`
      is graded against), `all_facts_given` (whether the prompt held every fact of the
      expected answer), and the token counts, time, model, strategy and top_k.
    - A question the model cannot answer within its limits (decision 41) gets its error in
      an `error` column instead of an answer, the run goes on, and the command exits with
      status 1. Any other error, such as Ollama becoming unreachable, stops the run without
      writing the file.
    - The file is never replaced, because it may hold grades: the command refuses to start
      while `answers.csv` exists, and the file is created in a way that fails if one
      appeared during the run.
    - Its summary counts the replies that are only "I don't know", ignoring citations,
      case, spacing and a final period. That is a count of wording, not of refusals
      (decision 46).
46. **`rag grades` computes the results from the graded answers.** It reads `id`,
    `expected_doc`, `supported`, `correct` and `all_facts_given` from
    `results/answers.csv` and ignores the other columns. It checks every grade against the
    rubric's options (case does not matter) and every id against the test set. It then
    reports:
    - for the 50 answerable questions, how many answers are supported, and how many are
      correct, partly correct and not correct;
    - how many of the questions that no document answers were graded correct, that is,
      refused;
    - the answerable questions split by whether the prompt held every fact of the answer;
    - the answers not graded yes on both.

    Refusals are counted from the graded `correct` column, not from the wording of the
    reply: u05's reply is "I don't know." followed by a sentence explaining why, which is
    a refusal that a wording match misses.

## Answering

41. **Generation never cuts text silently.** Probing Ollama 0.34.2 with llama3.2:3b and a
    4,096-token context showed two silent failures:
    - A prompt longer than the context is cut from the front. A 6,046-token prompt came
      back as an ordinary answer (HTTP 200, `done_reason` "stop") built on its last 2,050
      tokens; the fact at the start was lost and the answer was wrong. The only trace was a
      warning in the server log.
    - When the prompt fits but prompt plus answer do not, Ollama drops part of the prompt
      to keep generating ("context shift": 2,045 prompt tokens in the probe).

    The chat adapter therefore sends `truncate: false` and `shift: false`. A prompt that
    does not fit is refused by Ollama (HTTP 400) and raises `PromptTooLongError`. An answer
    that reaches the end of the context, or the answer limit `chat_max_answer_tokens`
    (512), raises `AnswerTooLongError` rather than coming back cut off. As a last check, a
    reply whose prompt and answer together exceed the context raises an error. The context
    size is set on every request (`chat_context_tokens`, 4,096) rather than left to
    Ollama's default, which has changed between versions.
    - Both fields are in Ollama's API types since 0.12.6 but not in its API documentation,
      and older servers silently ignore them. The adapter reads the server version before
      its first request and refuses anything older.
    - The official Python client (0.6.3, the latest release) cannot send them for chat, so
      the adapter calls Ollama's REST API with httpx, now a declared dependency.
42. **The chat model gets the top 5 chunks.** Before choosing, the real prompt (system
    prompt, numbered sources and question, as `rag ask` builds it) was measured for every
    test question with `sections-4000` and top_k 3, 4 and 5, counted by Ollama itself,
    chat template included:

    | top_k | Largest prompt, 50 answerable questions | Median | Largest, 5 unanswerable |
    |---|---|---|---|
    | 3 | 810 tokens (q47) | 378 | 728 (u04) |
    | 4 | 911 tokens (q47) | 458 | 790 (u04) |
    | 5 | 1,040 tokens (q10) | 550 | 866 (u04) |

    Every prompt fits with room to spare: the context is 4,096 tokens, and leaving room
    for the longest answer allowed (512) leaves 3,584 for the prompt. So the context does
    not limit the choice. Top 5 gives the model a chunk from the right document for all
    50 questions (48 at 3) and every fact of the answer for 48 (44 at 3, 45 at 4), at the
    cost of longer prompts for a small model to read. Whether the extra chunks distract
    llama3.2:3b shows only in the graded answers. The prompts averaged 4.2
    characters per token, so 5 chunks at the 4,000-character limit of `sections-4000`
    would not fit; a corpus with sections that long stops with `PromptTooLongError`
    (decision 41), and `-k` or `[retrieval] top_k` lowers the count.
43. **How answers are generated.** The system prompt tells the model to use only the
    numbered sources, to cite each sentence's sources by number in square brackets, to
    reply with exactly "I don't know." when the sources do not answer the question, and
    to answer in one to three sentences. The user message lists each chunk under its number
    and source label, with the question last. Temperature is 0, so a question gets the same
    answer every time it is asked, which grading needs.
    - `rag ask` shows the answer, then the sources it cites with their labels, then the
      chunks the model was given but did not cite. An answer that cites nothing shows
      "Sources: none cited"; the retrieved chunks are never presented as its sources.
      Numbers that match no source are flagged with a warning.
    - An answer that is only "I don't know", ignoring citations, case, spacing and a final
      period, shows "Sources: none (the model says it doesn't know)", even when it cites a
      number, as the sample answer to u02 did. The answers file keeps the reply unchanged
      (decision 45).
    - The prompt was written before any test question was answered, and was checked only
      on questions written for that purpose, so the test set does not tune it.

## Sample corpus

20. **20 fictional documents for a made-up company,** Tallowmere Outfitters: 10 markdown
    files and 10 PDFs of customer policies, FAQs and handbook pages. Topics overlap as they
    would in a real company, but each fact appears in only one document, so every
    answerable question has one correct source.
21. **The PDFs are generated from text sources** in `data/pdf-sources/` with fpdf2, a
    development-only dependency. The build is deterministic, and a test fails if a PDF is
    out of date. 7 of the 10 PDFs have bookmarks. They include a bookmark whose title
    differs from the printed heading and a section that continues across a page break,
    because the section logic in decision 12 must handle both.
22. **Created dates are varied and realistic** (2021 to 2026, all different). They are set in
    the PDF metadata and the markdown front matter, not taken from the date the files were
    generated.
