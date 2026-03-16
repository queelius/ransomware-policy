# Minimal Ransomware Detection Prompt (Qwen Baseline)

## Task

You are analyzing a Windows host for ransomware. Call `get_telemetry()` to retrieve the current host telemetry snapshot, investigate using the available tools, then call `DECIDE(verdict, explanation)` with your assessment. You have up to 5 investigation steps.

## Verdict Definitions

- **ignore**: No indicators of compromise
- **monitor**: Suspicious activity but inconclusive
- **alert**: Likely malicious, requires human review
- **quarantine**: High confidence ransomware detected
- **block**: Active encryption in progress

## Investigation Guidance

Focus on these ransomware indicators:
- **File indicators**: Rapid entropy increases, mass file extensions changes (e.g., `.enc`, `.locked`, `.crypt`)
- **Process indicators**: Suspicious executables, unusual parent-child relationships, system process impersonation
- **Network indicators**: Unusual outbound connections, command-and-control patterns

Use the available tools strategically: `inspect_file()`, `check_process()`, `scan_directory()`, `list_connections()`, `query_registry()`, `list_process_handles()`, `query_event_log()`, `read_file_sample()`.

## Procedure

1. Call `get_telemetry()` to see the current state
2. Investigate suspicious findings with targeted tool calls (up to 5 steps)
3. Call `DECIDE(verdict, explanation)` with your final assessment
