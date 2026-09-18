from classical_llm.evaluation.generate import _format_prompt
from classical_llm.evaluation.score import _reference_text, make_blinded_review
from classical_llm.utils.io import read_jsonl, write_jsonl


class RecordingTokenizer:
    chat_template = "present"

    def __init__(self):
        self.messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        return "rendered"


def test_format_prompt_uses_messages_without_gold_answer() -> None:
    tokenizer = RecordingTokenizer()
    row = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "gold answer"},
        ]
    }

    assert _format_prompt(tokenizer, row) == "rendered"
    assert tokenizer.messages == row["messages"][:2]


def test_format_prompt_keeps_prompt_style_rows() -> None:
    tokenizer = RecordingTokenizer()
    prompt = [{"role": "user", "content": "question"}]

    assert _format_prompt(tokenizer, {"prompt": prompt}) == "rendered"
    assert tokenizer.messages == prompt


def test_reference_text_uses_gold_assistant_message() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "gold answer"},
        ]
    }

    assert _reference_text(row) == "gold answer"


def test_blind_review_retains_question_without_gold(tmp_path) -> None:
    source = tmp_path / "generation.jsonl"
    output = tmp_path / "blind.jsonl"
    write_jsonl(source, [{"id": "one", "task": "old_to_modern", "response": "reply",
                         "messages": [{"role": "user", "content": "question"},
                                      {"role": "assistant", "content": "gold"}]}])
    make_blinded_review({"base": source}, output)
    row = next(iter(read_jsonl(output)))
    assert row["prompt"] == [{"role": "user", "content": "question"}]
    assert row["reference"] == "gold"
