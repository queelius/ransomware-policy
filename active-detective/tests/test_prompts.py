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

    def test_recall_memory_not_in_tool_descriptions(self):
        assert "recall_memory" not in TOOL_DESCRIPTIONS

    def test_default_prompt_excludes_recall_memory(self):
        prompt = build_system_prompt()
        assert "recall_memory" not in prompt

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

    def test_tool_ablation_subset(self):
        tools = ["inspect_file", "check_process", "scan_directory"]
        prompt = build_system_prompt(available_tools=tools)
        assert "inspect_file" in prompt
        assert "list_connections" not in prompt

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

    def test_mentions_prior_telemetry_context(self):
        prompt = build_system_prompt()
        assert "prior" in prompt.lower()


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


class TestHistoryInPrompt:
    def test_history_windows_in_user_message(self):
        system = build_system_prompt()
        history_windows = [
            "[+0.0s] FILE path=C:/test.docx size_delta=+100",
            "[+0.0s] PROC pid=5 name=svchost.exe parent=1",
        ]
        current = "[+0.0s] FILE path=C:/test2.docx size_delta=+200 ext_change=.docx->.locked"
        messages = build_chat_messages(system, current, history_windows=history_windows)
        user_msg = messages[-1]["content"]
        assert "Window t-2" in user_msg or "window t-2" in user_msg.lower()
        assert "Window t-1" in user_msg or "window t-1" in user_msg.lower()
        assert "CURRENT" in user_msg or "current" in user_msg.lower()
        assert history_windows[0] in user_msg
        assert current in user_msg

    def test_no_history_just_current(self):
        system = build_system_prompt()
        current = "[+0.0s] FILE path=C:/test.docx size_delta=+100"
        messages = build_chat_messages(system, current)
        user_msg = messages[-1]["content"]
        assert current in user_msg
        # Should not have window labels when there's no history
        assert "t-2" not in user_msg

    def test_single_history_window(self):
        system = build_system_prompt()
        history_windows = ["[+0.0s] PROC pid=5 name=svchost.exe parent=1"]
        current = "[+0.0s] FILE path=C:/test.docx size_delta=+100"
        messages = build_chat_messages(system, current, history_windows=history_windows)
        user_msg = messages[-1]["content"]
        assert history_windows[0] in user_msg
        assert current in user_msg

    def test_empty_history_windows_same_as_none(self):
        system = build_system_prompt()
        current = "[+0.0s] FILE path=C:/test.docx size_delta=+100"
        msgs_none = build_chat_messages(system, current, history_windows=None)
        msgs_empty = build_chat_messages(system, current, history_windows=[])
        assert msgs_none[-1]["content"] == msgs_empty[-1]["content"]

    def test_history_windows_with_tool_history(self):
        """history_windows (telemetry context) and history (tool turns) are independent."""
        system = build_system_prompt()
        tool_history = [
            {"role": "assistant", "content": "<tool_call>inspect_file(\"x\")</tool_call>"},
            {"role": "user", "content": "<tool_result>{}</tool_result>"},
        ]
        history_windows = ["[+0.0s] FILE path=C:/old.docx size_delta=+50"]
        current = "[+0.0s] FILE path=C:/new.docx size_delta=+200"
        messages = build_chat_messages(
            system, current, history=tool_history, history_windows=history_windows,
        )
        # system + 2 tool turns + user
        assert len(messages) == 4
        user_msg = messages[-1]["content"]
        assert "Window t-1" in user_msg
        assert "Current window" in user_msg
        assert history_windows[0] in user_msg
        assert current in user_msg

    def test_three_history_windows_labels(self):
        system = build_system_prompt()
        history_windows = ["win0", "win1", "win2"]
        current = "current_telemetry"
        messages = build_chat_messages(system, current, history_windows=history_windows)
        user_msg = messages[-1]["content"]
        assert "Window t-3 (prior)" in user_msg
        assert "Window t-2 (prior)" in user_msg
        assert "Window t-1 (prior)" in user_msg
        assert "Current window" in user_msg
