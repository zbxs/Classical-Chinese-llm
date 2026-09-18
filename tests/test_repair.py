from jinja2 import Environment

from classical_llm.data.acquire import _task_from_instruction
from classical_llm.data.sft import build_sft_dataset
from classical_llm.training.common import set_assistant_end_token
from classical_llm.utils.io import read_jsonl, write_jsonl


def test_translation_task_not_appreciation():
    assert _task_from_instruction("文言文翻译：隆庆二年进士。", "") == "old_to_modern"
    assert _task_from_instruction("翻译成文言文：他去了。", "") == "modern_to_old"
    assert _task_from_instruction("解释成语：一叶知秋", "") is None
    assert _task_from_instruction("赏析古诗的表达手法", "") == "appreciation"


def test_eos_template_changes_only_assistant_end():
    class Tokenizer:
        def get_vocab(self):
            return {"<|endoftext|>": 151643}

    tok = Tokenizer()
    set_assistant_end_token(tok, "<|endoftext|>")
    template = Environment().from_string(tok.chat_template)
    messages = [{"role": "system", "content": "规则"}, {"role": "user", "content": "问题"}]
    prompt = template.render(messages=messages, add_generation_prompt=True)
    full = template.render(messages=messages + [{"role": "assistant", "content": "答案"}], add_generation_prompt=False)
    assert prompt == "<|im_start|>system\n规则<|im_end|>\n<|im_start|>user\n问题<|im_end|>\n<|im_start|>assistant\n"
    assert full == prompt + "答案<|endoftext|>\n"


def test_existing_instruction_is_not_broadened(tmp_path):
    source = tmp_path / "source.jsonl"
    prompt = "请判断这首诗的题材。\n故人西辞黄鹤楼，烟花三月下扬州。"
    write_jsonl(source, [{"task": "appreciation", "instruction": "请判断这首诗的题材。",
                          "input": prompt, "output": "这是一首送别诗。", "split": "train"}])
    build_sft_dataset([source], tmp_path / "sft")
    rows = list(read_jsonl(tmp_path / "sft/train.jsonl"))
    assert len(rows) == 1
    assert rows[0]["messages"][1]["content"] == prompt
