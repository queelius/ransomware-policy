"""Tests for system prompt and training format builders."""

import pytest

from training.prompts import (
    DECIDE_DESCRIPTION,
    TOOL_DESCRIPTIONS,
    build_chat_messages,
    build_system_prompt,
    format_episode_input,
)


class TestBuildSystemPrompt:
    def test_default_includes_all_tools(self):
        prompt = build_system_prompt()
        for tool_name in TOOL_DESCRIPTIONS:
            assert tool_name in prompt

    def test_default_includes_decide(self):
        prompt = build_system_prompt()
        assert "DECIDE" in prompt
        assert "verdict" in prompt

    def test_default_includes_budget(self):
        prompt = build_system_prompt(k_max=5)
        assert "5" in prompt

    def test_custom_budget(self):
        prompt = build_system_prompt(k_max=10)
        assert "10" in prompt

    def test_tool_ablation_excludes_recall(self):
        tools = ["inspect_file", "check_process", "scan_directory"]
        prompt = build_system_prompt(available_tools=tools)
        assert "inspect_file" in prompt
        assert "recall_memory" not in prompt

    def test_tool_ablation_single_tool(self):
        prompt = build_system_prompt(available_tools=["inspect_file"])
        assert "inspect_file" in prompt
        assert "check_process" not in prompt
        # DECIDE should always be present
        assert "DECIDE" in prompt

    def test_includes_all_verdicts(self):
        prompt = build_system_prompt()
        for verdict in ["ignore", "monitor", "alert", "quarantine", "block"]:
            assert verdict in prompt

    def test_includes_format_instructions(self):
        prompt = build_system_prompt()
        assert "<think>" in prompt
        assert "<tool_call>" in prompt
        assert "<tool_result>" in prompt


class TestFormatEpisodeInput:
    def test_combines_prompt_and_telemetry(self):
        prompt = "You are a detection agent."
        telemetry = "[t+5s] FILE path=C:/test.docx"
        result = format_episode_input(prompt, telemetry)
        assert prompt in result
        assert telemetry in result
        assert "Telemetry window" in result

    def test_empty_telemetry(self):
        result = format_episode_input("prompt", "")
        assert "prompt" in result


class TestBuildChatMessages:
    def test_basic_structure(self):
        msgs = build_chat_messages("system prompt", "telemetry text")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_system_message_content(self):
        msgs = build_chat_messages("You are an agent", "events")
        assert msgs[0]["content"] == "You are an agent"

    def test_user_message_contains_telemetry(self):
        msgs = build_chat_messages("sys", "[t+5s] FILE test")
        assert "[t+5s] FILE test" in msgs[1]["content"]

    def test_with_history(self):
        history = [
            {"role": "assistant", "content": "<tool_call>inspect_file(\"x\")</tool_call>"},
            {"role": "user", "content": "<tool_result>{}</tool_result>"},
        ]
        msgs = build_chat_messages("sys", "telemetry", history=history)
        assert len(msgs) == 4  # system + 2 history + user
        assert msgs[1]["role"] == "assistant"
        assert msgs[2]["role"] == "user"
        assert msgs[3]["role"] == "user"  # telemetry

    def test_without_history(self):
        msgs = build_chat_messages("sys", "tel", history=None)
        assert len(msgs) == 2
