"""第3回 宿題（実装）: save_learning_log と承認フローのテスト（LLMもAPIキーも使わない）。

保存先は一時フォルダに差し替えるので、本物の learning_log/ には書き込まない。
"""

import pytest
from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.test import TestModel

from app import agent as agent_module
from app import tools


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "ROOT", tmp_path)
    monkeypatch.setattr(tools, "LOG_DIR", tmp_path / "learning_log")
    return tmp_path / "learning_log"


def test_save_writes_markdown_without_overwriting(log_dir):
    first = tools.save_learning_log("LoRA / QLoRA", "- LoRA は差分だけ学習する")
    second = tools.save_learning_log("LoRA / QLoRA", "- 2回目")
    files = sorted(log_dir.glob("*.md"))
    assert "保存しました" in first and "保存しました" in second
    assert len(files) == 2  # 同じ見出しでも上書きしない
    assert all("/" not in f.name.split("_", 1)[1] for f in files)  # 見出しの「/」はファイル名に入らない
    text = files[0].read_text(encoding="utf-8")
    assert text.startswith("# LoRA / QLoRA") and "差分だけ学習" in text


def test_save_rejects_empty_input(log_dir):
    assert "保存はしていません" in tools.save_learning_log("", "本文")
    assert not log_dir.exists()


def _run(answer: str, monkeypatch):
    """TestModel に save_learning_log を呼ばせ、承認の質問に answer で答える。"""
    monkeypatch.setattr("builtins.input", lambda _prompt: answer)
    model = TestModel(call_tools=["save_learning_log"])
    return agent_module.run_with_approval(agent_module.agent, "今日の学びを記録して", model)


def test_save_waits_for_approval(log_dir):
    model = TestModel(call_tools=["save_learning_log"])
    result = agent_module.agent.run_sync("今日の学びを記録して", model=model)
    assert isinstance(result.output, DeferredToolRequests)  # 承認待ちで止まる
    assert result.output.approvals[0].tool_name == "save_learning_log"
    assert not log_dir.exists()  # 承認されるまでは書き込まない


def test_approved_call_is_executed(log_dir, monkeypatch):
    result = _run("y", monkeypatch)
    returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
    assert any("保存しました" in str(p.content) for p in returns)
    assert len(list(log_dir.glob("*.md"))) == 1
    assert isinstance(result.output, str)
    assert result.usage.requests >= 2  # 承認の前と後の実行を足し上げている


def test_duplicated_heading_is_removed(log_dir):
    tools.save_learning_log("PEFT", "# PEFT\n\n本文")
    text = next(log_dir.glob("*.md")).read_text(encoding="utf-8")
    assert text.count("# PEFT") == 1


def test_denied_call_is_not_executed(log_dir, monkeypatch):
    result = _run("n", monkeypatch)
    assert not log_dir.exists()
    assert isinstance(result.output, str)
