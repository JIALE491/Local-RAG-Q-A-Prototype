"""Command-line interface: `rag ingest`, `search`, `ask`, `inspect`, `eval`, `answer-all` and
`grades`.

This is the composition root, the only module that builds real adapters from the settings
and hands them to the services. Results go to stdout; logs and errors go to stderr.
"""

import argparse
import sys
import textwrap
import time
from collections import Counter
from collections.abc import Callable, Sequence
from contextlib import closing
from pathlib import Path

from rag.adapters.chroma_index import ChromaIndex
from rag.adapters.filesystem import DirectorySource
from rag.adapters.ollama_chat import OllamaChat
from rag.adapters.ollama_embedder import OllamaEmbedder
from rag.adapters.questions_csv import load_questions, load_test_set
from rag.adapters.results_csv import (
    ANSWERS_FILE,
    new_answers_path,
    read_grades,
    write_answers,
    write_comparison,
    write_retrieval_results,
)
from rag.chunking.fixed import FixedSizeChunker
from rag.chunking.sections import SectionChunker
from rag.chunking.sentences import SentenceChunker
from rag.config import STRATEGIES, AppConfig, load_config
from rag.domain.errors import GradesError, IndexNotBuiltError, InvalidDataError, RagError
from rag.domain.models import (
    Answer,
    AnsweredQuestion,
    Chunk,
    Correctness,
    EvalReport,
    Grade,
    GradeCounts,
    RetrievedChunk,
    adoption_order,
    count_grades,
)
from rag.domain.ports import ChunkingStrategy
from rag.log import configure_logging
from rag.services.answer import Answerer, answer_questions
from rag.services.evaluate import evaluate_retrieval
from rag.services.ingest import IngestReport, ingest
from rag.services.retrieve import Retriever

type Command = Callable[[AppConfig, argparse.Namespace], int]

EVAL_DEPTH = 5
"""Chunks kept per test question, enough to report hit rates at 1, 3 and 5."""


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_logging(verbose=args.verbose)
    command: str = args.command
    try:
        return _COMMANDS[command](load_config(args.config), args)
    except RagError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        if exc.hint:
            print(f"hint: {exc.hint}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag", description="Local question answering over PDF and markdown documents."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.toml"),
        help="settings file (default: ./config.toml)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="show debug logging")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    ingest_command = commands.add_parser("ingest", help="rebuild the index from the documents")
    _strategy_option(ingest_command, allow_all=True)
    search = commands.add_parser("search", help="show the chunks closest to a question")
    search.add_argument("question")
    search.add_argument("-k", type=int, help="how many chunks (default: [retrieval] top_k)")
    _strategy_option(search, allow_all=False)
    ask = commands.add_parser("ask", help="answer a question from the documents, with sources")
    ask.add_argument("question")
    ask.add_argument(
        "-k", type=int, help="how many chunks to give the model (default: [retrieval] top_k)"
    )
    _strategy_option(ask, allow_all=False)
    inspect = commands.add_parser("inspect", help="show what the index stores")
    inspect.add_argument("--doc", help="only chunks from this document (file name)")
    inspect.add_argument("--page", type=int, help="only chunks from this page")
    _strategy_option(inspect, allow_all=False)
    evaluate = commands.add_parser("eval", help="measure retrieval on the test set")
    _strategy_option(evaluate, allow_all=True)
    answer_all = commands.add_parser(
        "answer-all", help="answer every test question into a CSV for grading"
    )
    answer_all.add_argument(
        "-k", type=int, help="how many chunks to give the model (default: [retrieval] top_k)"
    )
    _strategy_option(answer_all, allow_all=False)
    commands.add_parser("grades", help="count the grades in the graded answers CSV")
    return parser


def _strategy_option(parser: argparse.ArgumentParser, *, allow_all: bool) -> None:
    if allow_all:
        parser.add_argument(
            "--strategy",
            choices=(*STRATEGIES, "all"),
            help="chunking strategy, or all of them (default: [chunking] strategy)",
        )
    else:
        parser.add_argument(
            "--strategy",
            choices=STRATEGIES,
            help="chunking strategy (default: [chunking] strategy)",
        )


def _ingest(config: AppConfig, args: argparse.Namespace) -> int:
    with closing(_embedder(config)) as embedder:
        for position, strategy in enumerate(_strategies(config, args.strategy)):
            report = ingest(
                source=DirectorySource(config.paths.docs_dir),
                strategy=strategy,
                embedder=embedder,
                index=_index(config, strategy),
                batch_size=config.ingest.batch_size,
            )
            if position:
                print()
            _print_ingest(report, config)
    return 0


def _search(config: AppConfig, args: argparse.Namespace) -> int:
    k = _top_k(config, args)
    [strategy] = _strategies(config, args.strategy)
    with closing(_embedder(config)) as embedder:
        retriever = Retriever(
            embedder=embedder, index=_index(config, strategy), strategy=strategy.name
        )
        [results] = retriever.search([args.question], k=k)
    print(f'Top {_count(len(results), "chunk")} for "{args.question}" ({strategy.name}):\n')
    for result in results:
        _print_result(result)
    return 0


def _ask(config: AppConfig, args: argparse.Namespace) -> int:
    k = _top_k(config, args)
    [strategy] = _strategies(config, args.strategy)
    with closing(_embedder(config)) as embedder, closing(_chat_model(config)) as chat_model:
        answerer = Answerer(
            retriever=Retriever(
                embedder=embedder, index=_index(config, strategy), strategy=strategy.name
            ),
            chat_model=chat_model,
            top_k=k,
        )
        started = time.perf_counter()
        answer = answerer.answer(args.question)
        seconds = time.perf_counter() - started
    _print_answer(answer, strategy=strategy.name, seconds=seconds, config=config)
    return 0


def _inspect(config: AppConfig, args: argparse.Namespace) -> int:
    [strategy] = _strategies(config, args.strategy)
    index = _index(config, strategy)
    chunks = index.chunks()
    manifest = index.manifest()
    if manifest is None:
        raise IndexNotBuiltError(f"the index for {strategy.name!r} is being rebuilt")
    per_document = Counter(chunk.doc.name for chunk in chunks)
    print(f"Index {config.paths.index_dir}")
    print(f"  collection   {strategy.name} (built with {manifest.embed_model})")
    print(f"  chunks       {len(chunks)} from {_count(len(per_document), 'document')}")
    doc: str | None = args.doc
    page: int | None = args.page
    if doc is None and page is None:
        print("\nChunks per document")
        for name, count in sorted(per_document.items()):
            print(f"  {name:<46} {count:>4}")
        return 0
    selected = [
        chunk
        for chunk in chunks
        if (doc is None or chunk.doc.name == doc) and (page is None or chunk.page == page)
    ]
    if not selected:
        raise InvalidDataError("no stored chunks match --doc and --page")
    print(f"\n{_count(len(selected), 'stored chunk')}\n")
    for chunk in selected:
        _print_chunk(chunk)
    return 0


def _eval(config: AppConfig, args: argparse.Namespace) -> int:
    questions = load_test_set(config.paths.questions_csv, config.paths.facts_csv)
    k = config.retrieval.eval_k
    reports: list[EvalReport] = []
    chunk_counts: dict[str, int] = {}
    with closing(_embedder(config)) as embedder:
        for strategy in _strategies(config, args.strategy):
            index = _index(config, strategy)
            chunks = index.chunks()
            reports.append(
                evaluate_retrieval(
                    questions=questions,
                    retriever=Retriever(embedder=embedder, index=index, strategy=strategy.name),
                    indexed_documents={chunk.doc.name for chunk in chunks},
                    strategy=strategy.name,
                    k=k,
                    depth=max(k, EVAL_DEPTH),
                )
            )
            chunk_counts[strategy.name] = len(chunks)
    paths = [write_retrieval_results(report, config.paths.results_dir) for report in reports]
    left_out = len(questions) - len(reports[0].results)
    if len(reports) == 1:
        _print_eval(reports[0], left_out=left_out)
    else:
        [in_use] = _strategies(config, None)
        hit_cutoffs, coverage_cutoffs = sorted({1, k, EVAL_DEPTH}), sorted({k, EVAL_DEPTH})
        paths.append(
            write_comparison(
                reports,
                chunk_counts,
                config.paths.results_dir,
                hit_cutoffs=hit_cutoffs,
                coverage_cutoffs=coverage_cutoffs,
                in_use=in_use.name,
            )
        )
        _print_comparison(
            reports,
            chunk_counts,
            left_out=left_out,
            hit_cutoffs=hit_cutoffs,
            coverage_cutoffs=coverage_cutoffs,
            in_use=in_use.name,
        )
    print("\nResults:")
    for path in paths:
        print(f"  {path}")
    return 0


def _answer_all(config: AppConfig, args: argparse.Namespace) -> int:
    folder = config.paths.results_dir
    new_answers_path(folder)  # fail now, not after every question has been answered
    questions = load_test_set(config.paths.questions_csv, config.paths.facts_csv)
    k = _top_k(config, args)
    [strategy] = _strategies(config, args.strategy)
    answered: list[AnsweredQuestion] = []
    with closing(_embedder(config)) as embedder, closing(_chat_model(config)) as chat_model:
        answerer = Answerer(
            retriever=Retriever(
                embedder=embedder, index=_index(config, strategy), strategy=strategy.name
            ),
            chat_model=chat_model,
            top_k=k,
        )
        model = chat_model.model_name
        print(
            f"Answering {_count(len(questions), 'question')} with {model}, "
            f"{_count(k, 'chunk')} each from {strategy.name}\n"
        )
        for item in answer_questions(questions, answerer):
            answered.append(item)
            print(f"  {item.question.qid:<5}{item.seconds:5.1f} s   {_outcome(item)}")
    path = write_answers(answered, folder, model=model, strategy=strategy.name, top_k=k)
    _print_answer_summary(answered)
    print(f"\nResults: {path}\n(supported and correct are left empty, for grading)")
    return 1 if any(item.error for item in answered) else 0


def _grades(config: AppConfig, _args: argparse.Namespace) -> int:
    path = config.paths.results_dir / ANSWERS_FILE
    grades = read_grades(config.paths.results_dir)
    expected = {q.qid: q.answerable for q in load_questions(config.paths.questions_csv)}
    graded = {grade.qid: grade.answerable for grade in grades}
    if graded != expected:
        missing = [qid for qid in expected if qid not in graded]
        extra = [qid for qid in graded if qid not in expected]
        changed = [qid for qid in expected if qid in graded and graded[qid] != expected[qid]]
        problems = [f"missing {', '.join(missing)}"] if missing else []
        problems += [f"not in the test set: {', '.join(extra)}"] if extra else []
        problems += [f"answerable or not differs for {', '.join(changed)}"] if changed else []
        raise GradesError(
            f"{path} does not match the test set ({'; '.join(problems)})",
            hint="answer the test set again with `rag answer-all`, then grade the new file",
        )
    _print_grades(grades, path)
    return 0


_COMMANDS: dict[str, Command] = {
    "ingest": _ingest,
    "search": _search,
    "ask": _ask,
    "inspect": _inspect,
    "eval": _eval,
    "answer-all": _answer_all,
    "grades": _grades,
}


def _top_k(config: AppConfig, args: argparse.Namespace) -> int:
    k: int = config.retrieval.top_k if args.k is None else args.k
    if k < 1:
        raise InvalidDataError(f"-k must be 1 or more, got {k}")
    return k


def _strategies(config: AppConfig, choice: str | None) -> list[ChunkingStrategy]:
    """The strategy named on the command line, all of them, or the configured default."""
    chunking = config.chunking
    available: dict[str, ChunkingStrategy] = {
        "fixed": FixedSizeChunker(size=chunking.fixed.size, overlap=chunking.fixed.overlap),
        "sentences": SentenceChunker(size=chunking.sentences.size),
        "sections": SectionChunker(max_size=chunking.sections.max_size),
    }
    if choice == "all":
        return list(available.values())
    return [available[choice or chunking.strategy]]


def _index(config: AppConfig, strategy: ChunkingStrategy) -> ChromaIndex:
    return ChromaIndex(path=config.paths.index_dir, collection=strategy.name)


def _embedder(config: AppConfig) -> OllamaEmbedder:
    ollama = config.ollama
    return OllamaEmbedder(
        host=ollama.host,
        model=ollama.embed_model,
        document_prefix=ollama.embed_document_prefix,
        query_prefix=ollama.embed_query_prefix,
        timeout_s=ollama.timeout_s,
        keep_alive=ollama.keep_alive,
    )


def _chat_model(config: AppConfig) -> OllamaChat:
    ollama = config.ollama
    return OllamaChat(
        host=ollama.host,
        model=ollama.chat_model,
        context_tokens=ollama.chat_context_tokens,
        max_answer_tokens=ollama.chat_max_answer_tokens,
        temperature=ollama.chat_temperature,
        timeout_s=ollama.timeout_s,
        keep_alive=ollama.keep_alive,
    )


def _print_ingest(report: IngestReport, config: AppConfig) -> None:
    documents = _count(len(report.chunks_per_document), "document")
    print(f'Indexed {documents} as "{report.strategy}"')
    print(f"  chunks       {report.chunks}")
    print(f"  model        {report.embed_model}")
    first = report.batch_seconds[0]
    speed = report.steady_chunks_per_second
    print(
        f"  embedding    {report.embed_seconds:.1f} s in "
        f"{_count(len(report.batch_sizes), 'batch', 'batches')}; "
        f"first batch {first:.1f} s (includes loading the model if it was not loaded)"
        + (f", then {speed:.0f} chunks/s" if speed is not None else "")
    )
    print(f"  total time   {report.total_seconds:.1f} s")
    print(f"  stored in    {config.paths.index_dir / report.strategy}")
    print("\nChunks per document")
    for name, count in report.chunks_per_document:
        print(f"  {name:<46} {count:>4}")
    if report.skipped:
        print("\nSkipped files")
        for skipped in report.skipped:
            print(f"  {skipped.path}: {skipped.reason}")


def _print_answer(answer: Answer, *, strategy: str, seconds: float, config: AppConfig) -> None:
    for line in answer.text.strip().splitlines():
        print(textwrap.fill(line, width=92) if line.strip() else "")
    cited = () if answer.says_dont_know else answer.cited
    if answer.says_dont_know:
        print("\nSources: none (the model says it doesn't know)")
    else:
        print("\nSources" if cited else "\nSources: none cited")
    for number in cited:
        print(f"  [{number}] {answer.source(number).chunk.source.label()}")
    others = [number for number in range(1, len(answer.sources) + 1) if number not in cited]
    if others:
        print("\nAlso given to the model, not cited" if cited else "\nGiven to the model")
        for number in others:
            print(f"  [{number}] {answer.source(number).chunk.source.label()}")
    if answer.invalid_citations:
        listed = ", ".join(f"[{number}]" for number in answer.invalid_citations)
        given = _count(len(answer.sources), "source")
        print(f"\nwarning: the answer cites {listed}, but the model was given {given}")
    context = config.ollama.chat_context_tokens
    print(
        f"\n{answer.model} with {_count(len(answer.sources), 'chunk')} from {strategy}: "
        f"prompt {answer.prompt_tokens:,} of {context:,} tokens, "
        f"answer {_count(answer.answer_tokens, 'token')}, {seconds:.1f} s"
    )


def _outcome(item: AnsweredQuestion) -> str:
    """One line on what came back: an error, "I don't know", or the citations."""
    if item.answer is None:
        return f"failed: {item.error}"
    if item.answer.says_dont_know:
        return 'replies only "I don\'t know"'
    cited = ", ".join(f"[{number}]" for number in item.answer.citations)
    return f"answered, citing {cited}" if cited else "answered, citing nothing"


def _print_answer_summary(answered: Sequence[AnsweredQuestion]) -> None:
    answers = [item.answer for item in answered if item.answer is not None]
    seconds = sum(item.seconds for item in answered)
    unanswerable = [item for item in answered if not item.question.answerable]
    dont_know = [item for item in answered if item.answer and item.answer.says_dont_know]
    dont_know_unanswerable = [item for item in dont_know if not item.question.answerable]
    print(f"\n{_count(len(answers), 'answer')} in {seconds / 60:.1f} minutes")
    exactly = 'replies only "I don\'t know"'
    print(
        f"  {exactly:<29}{len(dont_know) - len(dont_know_unanswerable)} of "
        f"{len(answered) - len(unanswerable)} answerable, {len(dont_know_unanswerable)} of "
        f"{len(unanswerable)} unanswerable"
    )
    uncited = [a for a in answers if not a.says_dont_know and not a.citations]
    print(f"  {'cites nothing':<29}{len(uncited)}")
    print(f"  {'cites a made-up number':<29}{sum(bool(a.invalid_citations) for a in answers)}")
    failed = [item for item in answered if item.answer is None]
    print(f"  {'failed':<29}{len(failed)}")
    for item in failed:
        print(f"    {item.question.qid}: {item.error}")


def _print_grades(grades: Sequence[Grade], path: Path) -> None:
    answerable = [grade for grade in grades if grade.answerable]
    unanswerable = [grade for grade in grades if not grade.answerable]
    counts = count_grades(answerable)
    print(f"Grades of {_count(len(grades), 'answer')} in {path}\n")
    print(f"  {'answerable questions':<36}{counts.answers}")
    print(f"  {'supported':<36}{_share(counts.supported, counts.answers)}")
    print(f"  {'correct: yes / partial / no':<36}{_levels(counts)}")
    if unanswerable:
        refused = count_grades(unanswerable).correct[Correctness.YES]
        label = 'unanswerable, said "I don\'t know"'
        print(f"  {label:<36}{refused}/{len(unanswerable)}")
    print("\nAnswerable questions, by whether the prompt held every fact of the answer")
    for value, label in (
        (True, "every fact given"),
        (False, "a fact missing"),
        (None, "not known"),
    ):
        group = [grade for grade in answerable if grade.all_facts_given is value]
        if group:
            part = count_grades(group)
            print(
                f"  {f'{label} ({part.answers})':<24}supported {part.supported}/{part.answers}, "
                f"correct {_levels(part)}"
            )
    weak = [grade for grade in grades if not (grade.supported and grade.correct is Correctness.YES)]
    print(f"\n{_count(len(weak), 'answer')} not graded yes on both")
    for grade in weak:
        supported = "yes" if grade.supported else "no"
        print(f"  {grade.qid:<5}supported {supported:<4}correct {grade.correct}")


def _levels(counts: GradeCounts) -> str:
    """Answers graded correct yes, partial and no, as "43 / 4 / 3"."""
    return " / ".join(str(counts.correct[level]) for level in Correctness)


def _print_eval(report: EvalReport, *, left_out: int) -> None:
    total = len(report.results)
    without = f"; {_count(left_out, 'question')} with no answer left out" if left_out else ""
    print(f'Retrieval evaluation of "{report.strategy}": {total} answerable questions{without}\n')
    for n in sorted({1, report.k, EVAL_DEPTH}):
        measure = "   <- the measure" if n == report.k else ""
        print(f"  {f'hit rate at {n}':<23}{_share(report.hits_at(n), total)}{measure}")
    print(f"  {'mean reciprocal rank':<23}{report.mean_reciprocal_rank:.3f}")
    for n in sorted({report.k, EVAL_DEPTH}):
        print(f"  {f'answer coverage at {n}':<23}{_share(report.covered_at(n), total)}")
    _print_misses(report)
    _print_uncovered(report)


def _print_comparison(
    reports: Sequence[EvalReport],
    chunk_counts: dict[str, int],
    *,
    left_out: int,
    hit_cutoffs: Sequence[int],
    coverage_cutoffs: Sequence[int],
    in_use: str,
) -> None:
    total = len(reports[0].results)
    without = f"; {_count(left_out, 'question')} with no answer left out" if left_out else ""
    print(f"Retrieval comparison on {total} answerable questions{without}\n")
    columns = [f"hit@{n}" for n in hit_cutoffs] + ["MRR"]
    columns += [f"cover@{n}" for n in coverage_cutoffs]
    print(f"  {'strategy':<16}{'chunks':>8}" + "".join(f"{column:>9}" for column in columns))
    for report in reports:
        cells = [f"{report.hits_at(n)}/{total}" for n in hit_cutoffs]
        cells.append(f"{report.mean_reciprocal_rank:.3f}")
        cells += [f"{report.covered_at(n)}/{total}" for n in coverage_cutoffs]
        print(
            f"  {report.strategy:<16}{chunk_counts[report.strategy]:>8}"
            + "".join(f"{cell:>9}" for cell in cells)
        )
    k = reports[0].k
    print(f"\nOrder by the adoption rule (top-{k}, then top-1, then mean reciprocal rank):")
    for position, report in enumerate(adoption_order(reports), start=1):
        print(f"  {position}. {report.strategy}")
    print(f"In use ([chunking] strategy): {in_use}")
    for report in reports:
        print(f"\n== {report.strategy}")
        _print_misses(report)
        _print_uncovered(report)


def _print_misses(report: EvalReport) -> None:
    misses = report.misses
    print(f"\n{_count(len(misses), 'miss', 'misses')} at {report.k}")
    for result in misses:
        question = result.question
        rank = result.hit_rank
        found = f"first at rank {rank}" if rank else f"not in the top {len(result.retrieved)}"
        print(f"\n{question.qid}  {question.question}")
        print(f"     expected  {question.expected_doc} ({found})")
        for retrieved in result.retrieved[: report.k]:
            label = retrieved.chunk.source.label()
            print(f"     {retrieved.rank}. {label}  (score {retrieved.score:.3f})")


def _print_uncovered(report: EvalReport) -> None:
    uncovered = report.uncovered
    print(f"\n{_count(len(uncovered), 'question')} whose top {report.k} chunks miss a fact")
    for result in uncovered:
        question = result.question
        rank = result.coverage_rank
        covered = f"by rank {rank}" if rank else f"not in the top {len(result.retrieved)}"
        print(f"\n{question.qid}  {question.question}")
        for fact in result.missing_facts(report.k):
            print(f"     missing   {fact}")
        print(f"     covered   {covered}")


def _share(count: int, total: int) -> str:
    return f"{count}/{total} ({count / total:.0%})"


def _count(number: int, singular: str, plural: str | None = None) -> str:
    noun = singular if number == 1 else (plural or f"{singular}s")
    return f"{number} {noun}"


def _print_result(result: RetrievedChunk) -> None:
    print(f"{result.rank}. {result.chunk.source.label()}  (score {result.score:.3f})")
    print(textwrap.indent(textwrap.fill(result.chunk.text, width=92), "   "), end="\n\n")


def _print_chunk(chunk: Chunk) -> None:
    info = chunk.doc
    created = f"{info.created} from {info.created_from}" if info.created else "unknown"
    print(f"{chunk.chunk_id}")
    print(f"  document  {info.name} ({info.kind})")
    print(f"  page      {chunk.page if chunk.page is not None else '-'}")
    print(f"  section   {chunk.section if chunk.section is not None else '(none)'}")
    print(f"  created   {created}")
    print(f"  source    {chunk.source.label()}")
    print(textwrap.indent(textwrap.fill(chunk.text, width=88), "  | "), end="\n\n")
