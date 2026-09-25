"""The CLI's parsing and file formats. Nothing here loads a model."""

import json

import pytest

from jul import Choice, Noul, Score
from jul_cli.main import build_parser, load_questions, read_examples


QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team?",
             "criteria": {"billing": "payments", "technical": "bugs"}},
    "is_bug": {"type": "noul", "instructions": "Is this a bug?"},
    "anger": {"type": "score", "instructions": "How angry?", "criteria": ["Calm", "Mad"]},
}


@pytest.fixture
def questions_file(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(QUESTIONS))
    return path


def test_a_json_question_file_builds_the_three_types(questions_file):
    questions = load_questions(questions_file)
    assert isinstance(questions["team"], Choice)
    assert isinstance(questions["is_bug"], Noul)
    assert isinstance(questions["anger"], Score)
    assert questions["team"].criteria == {"billing": "payments", "technical": "bugs"}
    assert questions["anger"].criteria == ["Calm", "Mad"]


def test_yaml_and_json_question_files_agree(tmp_path, questions_file):
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "questions.yaml"
    path.write_text(yaml.safe_dump(QUESTIONS))
    assert load_questions(path).keys() == load_questions(questions_file).keys()
    assert load_questions(path)["team"].criteria == load_questions(questions_file)["team"].criteria


def test_noul_criteria_are_read_as_true_and_false(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps({"b": {"type": "noul", "instructions": "i",
                                      "criteria": {"true": "yes it is", "false": "no"}}}))
    assert load_questions(path)["b"].criteria.true == "yes it is"


def test_an_unknown_question_type_is_refused(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps({"x": {"type": "guess", "instructions": "i"}}))
    with pytest.raises(SystemExit):
        load_questions(path)


def test_a_question_file_must_be_a_mapping(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps(["not", "a", "mapping"]))
    with pytest.raises(SystemExit):
        load_questions(path)


def test_examples_are_read_from_txt_or_jsonl(tmp_path):
    txt = tmp_path / "e.txt"
    txt.write_text("first\nsecond\n\n")
    assert read_examples(txt) == ["first", "second"]
    jsonl = tmp_path / "e.jsonl"
    jsonl.write_text('{"text": "first"}\n{"text": "second"}\n')
    assert read_examples(jsonl) == ["first", "second"]


def test_ask_parses_options_and_repeated_states():
    a = build_parser().parse_args(["ask", "choice", "Which team?", "-o", "billing:payments",
                                   "-o", "technical:bugs", "--state", "one", "--state", "two"])
    assert a.kind == "choice" and a.state == ["one", "two"]
    assert a.option == ["billing:payments", "technical:bugs"]


def test_run_needs_an_input():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "questions.yaml"])


def test_lab_arguments_are_passed_through():
    a = build_parser().parse_args(["lab", "bench", "--task", "tasks/emotion.json"])
    assert a.rest == ["bench", "--task", "tasks/emotion.json"]


def test_context_actions_are_constrained():
    assert build_parser().parse_args(["context", "list"]).action == "list"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["context", "frobnicate", "x"])


# --- jul setup ----------------------------------------------------------------------------------

REQUIRES = ['numpy>=1.24',
            'mlx>=0.32; (sys_platform == "darwin" and platform_machine == "arm64") and extra == "mlx"',
            'torch>=2.4; extra == "torch"', 'transformers>=5.0; extra == "torch"',
            'pyyaml>=6; extra == "yaml"']


def test_setup_reads_an_extra_with_its_platform_markers_but_without_the_extra(monkeypatch):
    from jul_cli import setup
    monkeypatch.setattr(setup.importlib.metadata, "requires", lambda _: REQUIRES)
    assert setup.extra_requirements("torch") == ["torch>=2.4", "transformers>=5.0"]
    assert setup.extra_requirements("mlx") == [
        'mlx>=0.32; (sys_platform == "darwin" and platform_machine == "arm64")']


def test_setup_without_install_names_the_missing_backend_and_the_command(monkeypatch):
    from jul_cli import setup
    monkeypatch.setattr(setup.importlib.util, "find_spec", lambda _: None)
    with pytest.raises(SystemExit, match=r'torch is not installed.*pip install -e "\.\[torch\]"'):
        setup.ensure_backend("torch", install=False)


def test_setup_is_a_command():
    a = build_parser().parse_args(["setup", "--backend", "torch", "--no-check"])
    assert a.backend == "torch" and a.no_check and not a.no_install


def test_ask_says_to_run_setup_when_the_weights_are_not_downloaded(monkeypatch):
    from jul_cli import setup
    from jul_cli.main import main
    monkeypatch.setattr(setup, "missing_modules", lambda _: [])
    monkeypatch.setattr(setup, "weights_cached", lambda *_: False)
    with pytest.raises(SystemExit, match=r"not downloaded for torch.*Run: jul setup --backend torch"):
        main(["ask", "choice", "q", "-o", "a", "-o", "b", "--state", "s", "--backend", "torch"])


def test_ask_says_to_run_setup_when_no_backend_is_installed(monkeypatch):
    from jul_cli.main import main
    monkeypatch.delenv("JUL_BACKEND", raising=False)
    monkeypatch.setattr("jul.backbone.resolve_backend",
                        lambda _: (_ for _ in ()).throw(ImportError("none")))
    with pytest.raises(SystemExit, match=r"No backend installed. Run: jul setup --model minicpm5-2b"):
        main(["ask", "choice", "q", "-o", "a", "-o", "b", "--state", "s", "--model", "minicpm5-2b"])
