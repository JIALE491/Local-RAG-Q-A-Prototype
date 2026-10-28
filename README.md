# Local RAG Q&A prototype

Question answering over a folder of PDF and markdown documents, running entirely on one
laptop with local models (Ollama) and a local vector store (chromadb). Every pipeline stage
is written in this repository, without LangChain or LlamaIndex, and measured on its own:
retrieval is scored separately from the answers.

**This is a local, single-user prototype.** It runs on one machine for one person, from the
command line. It has no server, no user accounts or access control, and no cloud services.
It is not built or tested for production use.

## What it does

- **Reads PDFs and markdown files**, keeping each passage's document name, page (PDF),
  section (a PDF bookmark or a markdown heading) and the document's created date.
- **Splits them into chunks, embeds them** with `nomic-embed-text` and stores them in
  chromadb on disk. Three chunking strategies were compared on the same test questions;
  `sections-4000`, one chunk per section, is in use.
- **Answers questions** with `llama3.2:3b`, given the 5 closest chunks. The answer cites its
  sources by document name plus page (PDF) or heading (markdown), and says "I don't know"
  when the sources do not answer.
- **Measures itself** on a test set of 55 questions: retrieval hit rates and answer
  coverage for every chunking strategy, and every answer for grading.

```
$ rag ask "How often do forklift drivers need to renew their certification?"
Forklift drivers need to renew their certification every 3 years. [1]

Sources
  [1] workplace-safety-and-incident-reporting.pdf, p. 1 (Forklifts and powered equipment)

Also given to the model, not cited
  [2] loyalty-program-terms.pdf, p. 2 (Changes to the program)
  [3] onboarding-checklist.md (First 30 days)
  [4] loyalty-program-terms.pdf, p. 1 (Reward certificates)
  [5] code-of-conduct.pdf, p. 2

llama3.2:3b with 5 chunks from sections-4000: prompt 645 of 4,096 tokens, answer 18 tokens, 4.6 s
```

The sample documents describe Tallowmere Outfitters, an outdoor gear retailer invented for
this project; everything in [`data/`](data/README.md) is fictional.

## Setup

Tested on a MacBook Air M2 with 8 GB of memory, running macOS.

1. Install Python 3.12, for example with `brew install python@3.12`.
2. Install [Ollama](https://ollama.com), 0.12.6 or newer (tested with 0.34.2), start it,
   and download the two models (274 MB and 2.0 GB):

   ```
   ollama pull nomic-embed-text
   ollama pull llama3.2:3b
   ```

3. Clone this repository, then create the virtual environment and install the package with
   its development tools:

   ```
   git clone https://github.com/JIALE491/Local-RAG-Q-A-Prototype.git
   cd Local-RAG-Q-A-Prototype
   python3.12 -m venv .venv
   make install
   source .venv/bin/activate
   ```

## Usage

Run every command from the repository folder, with Ollama running:

```
rag ingest                       # build the index from data/docs (--strategy all: all three)
rag ask "Can I ship a kayak?"    # answer a question, with the sources it cites
rag search "Can I ship a kayak?" # show the closest chunks and their scores
rag inspect --doc shipping-faq.md
rag eval                         # retrieval on the test set (--strategy all: compare the three)
rag answer-all                   # answer all 55 test questions into results/answers.csv
rag grades                       # count the grades once answers.csv has been graded
```

The index is not stored in the repository, so run `rag ingest` first, or
`rag ingest --strategy all` before comparing strategies. Each ingest rebuilds the index from
scratch. `rag eval` replaces its result files on every run; `rag answer-all` refuses to
start while `results/answers.csv` exists, because that file may hold grades.

Settings are in [`config.toml`](config.toml): folders, models, chunking strategies and how
many chunks the model gets. Every key is required and unknown keys are rejected, so a typo
fails loudly. Expected failures print an `error:` line and a `hint:` line, never a stack
trace.

## Use your own documents

Nothing in the pipeline is specific to the sample company: it answers from any folder of PDF
and markdown files. Keep your settings, index and results in a folder of your own, outside
the repository, so the sample results stay as published and your documents' text never
ends up in a commit.

1. Copy the settings into a new folder:

   ```
   mkdir -p ~/rag-data
   cp config.toml ~/rag-data/config.toml
   ```

2. In `~/rag-data/config.toml`, point `docs_dir` at your documents, for example
   `docs_dir = "~/my-documents"`. Relative paths are resolved from the folder that holds
   the config file, so the index and results go to `~/rag-data/index` and
   `~/rag-data/results`.

3. Build the index from your documents, then ask:

   ```
   rag --config ~/rag-data/config.toml ingest
   rag --config ~/rag-data/config.toml ask "your question"
   ```

What your documents need:

- **PDF or markdown files.** Other files are skipped and listed. A scanned PDF, which has no
  text layer, or an encrypted PDF stops the ingest with an error naming the file: there is
  no OCR.
- **Unique file names**, even in different subfolders, because sources are shown by file
  name.
- **Bookmarks or headings, for section labels.** Markdown headings become sections. A PDF
  needs bookmarks; Word adds them when you export to PDF with "Create bookmarks using
  headings". Without them, a PDF's sources show the page only.
- **Sections of moderate length.** If the chunks given to the model are too long for its
  context, `rag ask` stops with an error instead of cutting the prompt; `-k 3` gives the
  model fewer chunks.

To measure retrieval and answers on your documents too, write a `questions.csv` and a
`facts.csv` in the format described in [`data/README.md`](data/README.md), and set
`questions_csv` and `facts_csv` in your config to their paths. `rag eval`, `rag answer-all`
and `rag grades` then work as they do for the sample set.

## Architecture

```
src/rag/
  domain/      data types, ports (interfaces) and errors; standard library only
  chunking/    the three chunking strategies
  services/    ingest, retrieve, evaluate and answer, written against the ports
  adapters/    PDF and markdown loaders, Ollama embedder and chat model, chromadb, CSV files
  cli.py       the composition root: builds the adapters from config.toml
```

Inner layers never import outer ones: domain is used by chunking and services, which are
used by adapters and the CLI. A unit test enforces this rule. The pipeline:

```
rag ingest   PDF or markdown -> segments (page, section) -> chunks -> nomic-embed-text -> chromadb
rag ask      question -> nomic-embed-text -> top 5 chunks -> numbered sources -> llama3.2:3b
             -> answer, with the sources it cites
```

A few properties worth knowing:

- **Nothing is cut off silently.** By default Ollama shortens text that is too long without
  saying so. Here, a chunk too long for the embedding model, a prompt too long for the chat
  model's context, and an answer that runs out of room all raise an error instead. Integration
  tests check this against the installed Ollama, because the chat settings that do it are
  not in Ollama's documentation (decision 41).
- **An index is either complete or refuses to answer.** Every ingest rebuilds the
  strategy's index from scratch, and a search is refused unless the last rebuild finished,
  so an interrupted ingest cannot leave an index that silently lacks documents.
- **Chunks never cross a page or section**, so every source label is exact.
- **Fakes stand in for Ollama and chromadb** in the unit tests, and shared contract tests
  run against the fakes and the real adapters alike, so the fakes cannot drift.

Every design decision, with its reason, is in [`docs/decisions.md`](docs/decisions.md).

## Evaluation results

The test set in [`data/eval/`](data/eval) has 50 questions that one document answers (2 or 3
per document), 5 that no document answers, and the 78 facts the expected answers rely on.
The questions were written together with the documents, by the same author (decision 31),
so they are an easier test than questions written independently.

All results were measured on a MacBook Air M2 (8 GB) with Python 3.12.12, Ollama 0.34.2,
chromadb 1.5.9 and pypdf 6.21.3, using `nomic-embed-text` (ID `0a109f422b47` in
`ollama list`) and `llama3.2:3b` (ID `a80c4f17acd5`). A model tag can be updated upstream;
the IDs identify the exact models used.

### Retrieval: three chunking strategies

Hits are questions with a chunk from the right document among the top n; covered means the
top n chunks contain every fact of the answer, each one whole.

| Strategy | Chunks | Hits at 1 | Hits at 3 | Hits at 5 | MRR | Covered at 3 | Covered at 5 |
|---|---|---|---|---|---|---|---|
| `fixed-300-50`: word windows of 300 characters | 241 | 43 | 48 | 50 | 0.917 | 38 | 42 |
| `sentences-300`: whole sentences, up to 300 characters | 240 | 44 | 48 | 49 | 0.925 | 39 | 41 |
| `sections-4000`: one chunk per section, in use | 168 | 41 | 48 | 50 | 0.888 | 44 | 48 |

All three put the right document in the top 3 for 48 of 50 questions. `sections-4000` was
adopted because it gives the chat model every fact of the answer far more often (decision
40). Embedding its 168 chunks takes about 3 seconds.

### Answers

With `sections-4000`, the top 5 chunks and `llama3.2:3b` at temperature 0, as counted by
`rag grades` from [`results/answers.csv`](results/answers.csv):

| | Result |
|---|---|
| Supported: every statement follows from the chunks given | 44 of 50 (88%) |
| Correct, compared with the expected answer: yes / partial / no | 43 / 4 / 3 |
| Questions no document answers, answered "I don't know" | 5 of 5 |

The rubric was fixed before any answer was generated (decision 44). The grades were made by
an AI model following that rubric, not by hand; [`results/grading-notes.md`](results/grading-notes.md)
explains how the rubric was applied and lists every answer not graded yes on both.

All 9 answers not graded yes on both had every fact of the answer in their prompt. The
errors come from the chat model reading its sources, not from retrieval: one answer says
that a headlamp a year and a half old is more than 2 years old.

Answers take 1.4 to 4.8 seconds (median 2.6). Prompts use 552 tokens at the median and
1,040 at most, of a 4,096-token context.

### The charter's success criteria

| # | Criterion | Result |
|---|---|---|
| 1 | A question returns an answer with the document name and page of each source | Yes. Markdown has no pages, so its sources show the heading instead (decision 11). |
| 2 | The right document is in the top 3 for at least 40 of 50 questions | 48 of 50, with each strategy |
| 3 | The chunking choice is backed by data | Three strategies compared on the same 50 questions; they tie at 3, so answer coverage decided (decision 40) |
| 4 | At least 40 of 50 answers fully supported by their sources, judged by hand | 44 of 50 supported, but graded by an AI model rather than by hand |
| 5 | Delivered on time | No deadline was set |

### Limitations

- The test questions were written alongside the documents, and 50 questions are few: a
  difference of one or two questions is within noise.
- The answer grades come from an AI model following a fixed rubric, not from a person.
- The corpus is small: 20 fictional documents. Behaviour at the charter's stretch goal of
  500 documents has not been measured.
- `llama3.2:3b` sometimes misreads chunks it was given, as the errors above show.
- Page footers stay in the extracted text, so a sentence that runs across a page break is
  split by the footer (decisions 10 and 23).

## Out of scope

- Scanned PDFs and OCR
- File types other than PDF and markdown, such as Word documents, slides and spreadsheets
- Google Docs integration or live syncing
- Re-indexing automatically when documents change: run `rag ingest` again
- Access permissions, user accounts and use by more than one person
- Public web hosting or deployment
- Hybrid keyword search and reranking
- Chat models of 7B parameters or more
- Paid APIs and cloud models
- Automated LLM-as-judge grading: the pipeline never grades its own answers
- Follow-up questions with conversation memory: each question stands alone
- LangChain and LlamaIndex

## Possible future work

- **A fourth chunking strategy.** `sections-4000`, the strategy in use, keeps each section
  whole, which gave the best answer coverage of the three strategies compared. But a PDF
  page without bookmarks becomes a single chunk of up to about 2,200 characters, and the
  four questions on which `sections-4000` ranks the right document lower than the other
  strategies are all answered on such pages. Splitting long sections into sentence
  windows, while keeping short ones whole, might recover them. On 50 test questions the
  gain would be hard to tell from noise, so it was not built (`docs/decisions.md`,
  decision 40).
- **Scaling toward 500 documents**, the charter's stretch goal, measuring ingest time and
  retrieval at that size.
- **A local chat page**, for example with Streamlit or Gradio, which the charter lists as
  optional.

## Repository layout

| Path | Contents |
|---|---|
| `src/rag/` | The package |
| `tests/` | Unit, contract, corpus and integration tests |
| `data/docs/` | The 20 sample documents |
| `data/pdf-sources/` | The text the sample PDFs are generated from |
| `data/eval/` | The test set: `questions.csv` and `facts.csv` |
| `results/` | Retrieval results per strategy, `comparison.csv`, the graded `answers.csv` and `grading-notes.md` |
| `docs/decisions.md` | The design decisions and their reasons |
| `scripts/build_pdfs.py` | Builds the sample PDFs from their text |
| `config.toml` | The settings |

## Development

```
make check             # ruff lint and format check, mypy --strict, unit tests
make test-integration  # tests against the real Ollama; needs both models
make coverage          # unit tests with a branch-coverage report
make pdfs              # rebuild data/docs/*.pdf from data/pdf-sources/
```

## License

MIT; see [LICENSE](LICENSE).
