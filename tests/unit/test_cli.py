"""The command line: errors become one-line messages with hints, never tracebacks."""

import csv
import datetime as dt
import re
from pathlib import Path

import pytest

from rag.adapters.chroma_index import ChromaIndex
from rag.cli import main
from rag.domain.models import IndexManifest
from tests.builders import chunk
from tests.fakes import FakeChatModel, FakeEmbedder

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config.toml"


def config_in(tmp_path: Path) -> Path:
    """A copy of the repository config whose relative paths point inside tmp_path."""
    path = tmp_path / "config.toml"
    path.write_text(REPO_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    return path


def test_help_lists_the_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0
    out = capsys.readouterr().out
    assert all(command in out for command in ("ingest", "search", "inspect"))


def test_a_missing_config_file_is_an_error_with_a_hint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--config", str(tmp_path / "missing.toml"), "inspect"]) == 1
    err = capsys.readouterr().err
    assert "error: config file not found" in err
    assert "hint:" in err


def test_inspect_before_ingest_says_what_to_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--config", str(config_in(tmp_path)), "inspect"]) == 1
    err = capsys.readouterr().err
    assert "error: there is no index for 'sections-4000' yet" in err
    assert "hint: run `rag ingest`" in err


def test_search_rejects_k_below_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--config", str(config_in(tmp_path)), "search", "refunds?", "-k", "0"]) == 1
    assert "-k must be 1 or more" in capsys.readouterr().err


def build_index(tmp_path: Path, *, finish: bool = True) -> None:
    """Fill the index the copied config points at, with fake vectors (no Ollama needed)."""
    chunks = [
        chunk(
            "Refunds take 5 days.",
            doc="refunds.pdf",
            page=1,
            section="Refund window",
            created=dt.date(2024, 2, 12),
        ),
        chunk(
            "Store credit without a receipt.",
            doc="refunds.pdf",
            ordinal=1,
            page=2,
            created=dt.date(2024, 2, 12),
        ),
        chunk(
            "Vacation is 15 days.",
            doc="handbook.md",
            section="Vacation",
            created=dt.date(2022, 1, 10),
        ),
    ]
    index = ChromaIndex(path=tmp_path / "index", collection="sections-4000")
    index.begin_rebuild()
    index.add(chunks, FakeEmbedder().embed_documents([c.text for c in chunks]))
    if finish:
        index.finish_rebuild(
            IndexManifest(strategy="sections-4000", embed_model="nomic-embed-text")
        )


def inspect(tmp_path: Path, *options: str) -> int:
    return main(["--config", str(config_in(tmp_path)), "inspect", *options])


def test_inspect_summarises_the_index(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    build_index(tmp_path)
    assert inspect(tmp_path) == 0
    out = capsys.readouterr().out
    assert "collection   sections-4000 (built with nomic-embed-text)" in out
    assert "chunks       3 from 2 documents" in out
    assert re.search(r"handbook\.md +1\n", out)
    assert re.search(r"refunds\.pdf +2\n", out)


def test_inspect_shows_every_stored_field_of_a_page(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    build_index(tmp_path)
    assert inspect(tmp_path, "--doc", "refunds.pdf", "--page", "1") == 0
    out = capsys.readouterr().out
    for line in (
        "1 stored chunk\n",
        "refunds.pdf#0000",
        "document  refunds.pdf (pdf)",
        "page      1",
        "section   Refund window",
        "created   2024-02-12 from pdf_metadata",
        "source    refunds.pdf, p. 1 (Refund window)",
        "| Refunds take 5 days.",
    ):
        assert line in out


def test_inspect_shows_missing_page_and_section_plainly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    build_index(tmp_path)
    assert inspect(tmp_path, "--doc", "refunds.pdf", "--page", "2") == 0
    assert "section   (none)" in capsys.readouterr().out
    assert inspect(tmp_path, "--doc", "handbook.md") == 0
    out = capsys.readouterr().out
    assert "page      -" in out
    assert "created   2022-01-10 from front_matter" in out


def test_inspect_filters_that_match_nothing_are_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    build_index(tmp_path)
    assert inspect(tmp_path, "--doc", "missing.md") == 1
    assert "error: no stored chunks match --doc and --page" in capsys.readouterr().err


def test_inspect_refuses_an_index_whose_rebuild_did_not_finish(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    build_index(tmp_path, finish=False)
    assert inspect(tmp_path) == 1
    assert "did not finish" in capsys.readouterr().err


def write_questions(tmp_path: Path, rows: str, facts: str = "") -> None:
    """Write the test set the copied config points at: questions, and facts if given."""
    folder = tmp_path / "data" / "eval"
    folder.mkdir(parents=True, exist_ok=True)
    header = "id,question,expected_doc,expected_answer\n"
    (folder / "questions.csv").write_text(header + rows, encoding="utf-8")
    if facts:
        (folder / "facts.csv").write_text("id,fact\n" + facts, encoding="utf-8")


def test_eval_without_a_test_set_says_where_to_look(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--config", str(config_in(tmp_path)), "eval"]) == 1
    err = capsys.readouterr().err
    assert "error: the test set was not found" in err
    assert "hint: check [paths] questions_csv in config.toml" in err


def test_eval_refuses_expected_documents_that_are_not_indexed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    build_index(tmp_path)
    write_questions(
        tmp_path,
        "q1,How long do refunds take?,refunds.pdf,5 days\n"
        "q2,Is there a dress code?,dress-code.md,\n",
        "q1,5 days\nq2,jeans are fine\n",
    )
    assert main(["--config", str(config_in(tmp_path)), "eval"]) == 1
    assert "not in the index: q2 (dress-code.md)" in capsys.readouterr().err


def test_eval_without_a_fact_list_says_where_to_look(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    build_index(tmp_path)
    write_questions(tmp_path, "q1,How long do refunds take?,refunds.pdf,5 days\n")
    assert main(["--config", str(config_in(tmp_path)), "eval"]) == 1
    err = capsys.readouterr().err
    assert "error: the fact list was not found" in err
    assert "hint: check [paths] facts_csv in config.toml" in err


def test_inspect_can_name_another_strategy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    build_index(tmp_path)
    assert inspect(tmp_path, "--strategy", "sentences") == 1
    assert "no index for 'sentences-300' yet" in capsys.readouterr().err


def test_an_unknown_strategy_is_rejected_by_the_parser(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--config", str(config_in(tmp_path)), "inspect", "--strategy", "semantic"])
    assert caught.value.code == 2


def ask(tmp_path: Path, *options: str) -> int:
    return main(["--config", str(config_in(tmp_path)), "ask", *options])


def use_fakes(monkeypatch: pytest.MonkeyPatch, reply: str) -> None:
    """Swap the Ollama adapters for fakes, so `rag ask` runs without Ollama."""
    monkeypatch.setattr("rag.cli._embedder", lambda _: FakeEmbedder(model_name="nomic-embed-text"))
    monkeypatch.setattr("rag.cli._chat_model", lambda _: FakeChatModel(reply))


def test_ask_before_ingest_says_what_to_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert ask(tmp_path, "How long do refunds take?") == 1
    err = capsys.readouterr().err
    assert "error: there is no finished index for 'sections-4000'" in err
    assert "hint: run `rag ingest`" in err


def test_ask_rejects_k_below_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert ask(tmp_path, "How long do refunds take?", "-k", "0") == 1
    assert "-k must be 1 or more" in capsys.readouterr().err


def test_ask_prints_the_answer_and_the_sources_it_cites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    build_index(tmp_path)
    use_fakes(monkeypatch, "Refunds take 5 days [1].")
    assert ask(tmp_path, "How long do refunds take?", "-k", "2") == 0
    out = capsys.readouterr().out
    assert out.startswith("Refunds take 5 days [1].\n\nSources\n")
    assert "  [1] refunds.pdf, p. 1 (Refund window)\n" in out
    assert re.search(r"Also given to the model, not cited\n  \[2\] \S+", out)
    assert re.search(r"fake-chat with 2 chunks from sections-4000: prompt \d+ of 4,096 tokens", out)
    assert "warning" not in out


def test_ask_shows_no_sources_for_i_dont_know_even_when_it_cites_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    build_index(tmp_path)
    use_fakes(monkeypatch, "I don't know. [1]")
    assert ask(tmp_path, "Is there a dress code?") == 0
    out = capsys.readouterr().out
    assert out.startswith("I don't know. [1]\n\nSources: none (the model says it doesn't know)\n")
    assert re.search(r"\nGiven to the model\n  \[1\] \S+", out)


def test_ask_shows_an_answer_without_citations_and_made_up_ones_plainly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    build_index(tmp_path)
    use_fakes(monkeypatch, "Five days.")
    assert ask(tmp_path, "How long do refunds take?") == 0
    assert "Sources: none cited" in capsys.readouterr().out
    use_fakes(monkeypatch, "Five days [1][7].")
    assert ask(tmp_path, "How long do refunds take?") == 0
    assert "warning: the answer cites [7], but the model was given 3 sources" in (
        capsys.readouterr().out
    )


def test_ask_reports_a_prompt_too_long_for_the_model_as_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    build_index(tmp_path)
    monkeypatch.setattr("rag.cli._embedder", lambda _: FakeEmbedder(model_name="nomic-embed-text"))
    monkeypatch.setattr("rag.cli._chat_model", lambda _: FakeChatModel("Yes.", context_tokens=20))
    assert ask(tmp_path, "How long do refunds take?") == 1
    assert "does not fit in the 20-token context" in capsys.readouterr().err


def answer_all(tmp_path: Path) -> int:
    return main(["--config", str(config_in(tmp_path)), "answer-all"])


def read_answers(tmp_path: Path) -> list[dict[str, str]]:
    with (tmp_path / "results" / "answers.csv").open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def test_answer_all_refuses_to_start_while_an_answers_file_exists(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    graded = tmp_path / "results" / "answers.csv"
    graded.parent.mkdir()
    graded.write_text("id,supported\nq1,yes\n", encoding="utf-8")
    assert answer_all(tmp_path) == 1
    err = capsys.readouterr().err
    assert "answers.csv already exists and may hold your grades" in err
    assert "hint: rename or move it to answer the test set again" in err
    assert graded.read_text(encoding="utf-8") == "id,supported\nq1,yes\n"


def test_answer_all_writes_every_answer_with_empty_grades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    build_index(tmp_path)
    write_questions(
        tmp_path,
        "q1,How long do refunds take?,refunds.pdf,5 days\nu1,Is there a dress code?,,\n",
        "q1,Refunds take 5 days\n",
    )
    use_fakes(monkeypatch, "Refunds take 5 days [1].")
    assert answer_all(tmp_path) == 0
    out = capsys.readouterr().out
    assert out.startswith("Answering 2 questions with fake-chat, 5 chunks each from sections-4000")
    assert re.search(r"q1 +\d+\.\d s   answered, citing \[1\]", out)
    assert 'replies only "I don\'t know"  0 of 1 answerable, 0 of 1 unanswerable' in out
    assert re.search(r"failed +0\n", out)
    first, second = read_answers(tmp_path)
    assert (first["id"], first["answer"], first["supported"], first["correct"]) == (
        "q1",
        "Refunds take 5 days [1].",
        "",
        "",
    )
    assert (first["all_facts_given"], second["all_facts_given"]) == ("yes", "")
    assert first["prompt"].endswith("Question: How long do refunds take?")


def test_answer_all_records_a_failed_question_and_exits_with_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    build_index(tmp_path)
    write_questions(tmp_path, "q1,How long do refunds take?,refunds.pdf,\n", "q1,5 days\n")
    monkeypatch.setattr("rag.cli._embedder", lambda _: FakeEmbedder(model_name="nomic-embed-text"))
    monkeypatch.setattr(
        "rag.cli._chat_model", lambda _: FakeChatModel("word " * 600, max_answer_tokens=512)
    )
    assert answer_all(tmp_path) == 1
    assert "q1: the answer of 600 tokens was cut off" in capsys.readouterr().out
    [row] = read_answers(tmp_path)
    assert (row["answer"], row["error"]) == ("", "the answer of 600 tokens was cut off")


def test_grades_counts_the_graded_answers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_questions(tmp_path, "q1,How long?,refunds.pdf,\nq2,How much?,prices.md,\nu1,Wi-Fi?,,\n")
    results = tmp_path / "results"
    results.mkdir()
    (results / "answers.csv").write_text(
        "id,expected_doc,supported,correct,all_facts_given\n"
        "q1,refunds.pdf,yes,yes,yes\nq2,prices.md,no,partial,no\nu1,,yes,yes,\n",
        encoding="utf-8",
    )
    assert main(["--config", str(config_in(tmp_path)), "grades"]) == 0
    out = capsys.readouterr().out
    assert re.search(r"supported +1/2 \(50%\)", out)
    assert re.search(r"correct: yes / partial / no +1 / 1 / 0", out)
    assert re.search(r"unanswerable, said \"I don't know\" +1/1", out)
    assert re.search(r"every fact given \(1\) +supported 1/1, correct 1 / 0 / 0", out)
    assert re.search(r"a fact missing \(1\) +supported 0/1, correct 0 / 1 / 0", out)
    assert "1 answer not graded yes on both\n  q2   supported no  correct partial" in out


def test_grades_must_cover_exactly_the_test_set(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_questions(tmp_path, "q1,How long?,refunds.pdf,\nq2,How much?,prices.md,\n")
    results = tmp_path / "results"
    results.mkdir()
    (results / "answers.csv").write_text(
        "id,expected_doc,supported,correct,all_facts_given\n"
        "q1,refunds.pdf,yes,yes,yes\nq9,,yes,yes,\n",
        encoding="utf-8",
    )
    assert main(["--config", str(config_in(tmp_path)), "grades"]) == 1
    err = capsys.readouterr().err
    assert "does not match the test set (missing q2; not in the test set: q9)" in err
