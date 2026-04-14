#!/usr/bin/env python3
"""Generate SFT training data from MITRE ATT&CK STIX JSON.

Extracts ransomware-relevant techniques and generates QA pairs and
instruction-following examples for fine-tuning.

Usage:
    python scripts/generate_mitre_sft.py
"""

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

STIX_PATH = Path(__file__).parent.parent / "data" / "raw_content" / "mitre_attack_stix" / "enterprise-attack.json"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "raw" / "mitre_attack_stix.jsonl"

SYSTEM_PROMPT = "You are a ransomware forensic analyst with deep expertise in Windows internals and incident response."

TARGET_TACTICS = {
    "impact", "defense-evasion", "execution", "persistence",
    "lateral-movement", "credential-access", "initial-access",
    "collection", "exfiltration", "command-and-control",
}

PRIORITY_TECHNIQUE_IDS = {
    "T1486", "T1490", "T1562", "T1059", "T1547", "T1021", "T1003", "T1071",
}

GENERATED_AT = "2026-03-17T11:00:00Z"
GENERATOR_MODEL = "claude-opus-4-6"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_citations(text: str) -> str:
    """Remove MITRE-style (Citation: ...) references."""
    return re.sub(r'\(Citation:\s*[^)]+\)', '', text).strip()


def strip_markdown_links(text: str) -> str:
    """Convert [text](url) to just text."""
    return re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)


def clean_description(text: str) -> str:
    """Clean up STIX description for use in training data."""
    text = strip_citations(text)
    text = strip_markdown_links(text)
    # Collapse multiple spaces/newlines
    text = re.sub(r'\n{2,}', '\n\n', text)
    text = re.sub(r'  +', ' ', text)
    return text.strip()


def make_example(user_content, assistant_content, technique_id, example_type, prompt_id, topics=None):
    """Create a single training example in the expected format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "source": "mitre_attack_stix",
            "source_url": "https://github.com/mitre-attack/attack-stix-data",
            "source_section": technique_id,
            "example_type": example_type,
            "topics": topics or ["ransomware_ttps"],
            "generation_prompt_id": prompt_id,
            "generated_at": GENERATED_AT,
            "generator_model": GENERATOR_MODEL,
            "quality_score": None,
        }
    }


# ---------------------------------------------------------------------------
# STIX data extraction
# ---------------------------------------------------------------------------

def load_stix_data():
    """Load and index STIX objects."""
    with open(STIX_PATH, 'r') as f:
        data = json.load(f)

    objects_by_type = defaultdict(list)
    objects_by_id = {}
    for obj in data['objects']:
        objects_by_type[obj['type']].append(obj)
        objects_by_id[obj['id']] = obj

    return objects_by_type, objects_by_id


def extract_techniques(objects_by_type, objects_by_id):
    """Extract relevant techniques with their mitigations."""
    patterns = objects_by_type['attack-pattern']
    relationships = objects_by_type['relationship']

    # Build mitigation map: technique_stix_id -> list of mitigation descriptions
    mitigation_map = defaultdict(list)
    for rel in relationships:
        if rel.get('relationship_type') == 'mitigates' and not rel.get('revoked'):
            source = objects_by_id.get(rel.get('source_ref'))
            if source and source.get('type') == 'course-of-action':
                mitigation_map[rel['target_ref']].append({
                    'name': source.get('name', ''),
                    'description': clean_description(source.get('description', '')),
                })

    # Build uses map: technique_stix_id -> list of malware/group using it
    uses_map = defaultdict(list)
    for rel in relationships:
        if rel.get('relationship_type') == 'uses' and not rel.get('revoked'):
            source = objects_by_id.get(rel.get('source_ref'))
            target = objects_by_id.get(rel.get('target_ref'))
            if source and target and target.get('type') == 'attack-pattern':
                src_type = source.get('type')
                if src_type in ('malware', 'intrusion-set'):
                    uses_map[target['id']].append({
                        'name': source.get('name', ''),
                        'type': src_type,
                        'description': clean_description(rel.get('description', '')),
                    })

    techniques = []
    for p in patterns:
        if p.get('revoked') or p.get('x_mitre_deprecated'):
            continue

        # Get external ID
        refs = p.get('external_references', [])
        if not refs:
            continue
        ext_id = refs[0].get('external_id', '')
        if not ext_id.startswith('T'):
            continue

        # Get tactics
        phases = p.get('kill_chain_phases', [])
        tactics = {ph['phase_name'] for ph in phases if ph.get('kill_chain_name') == 'mitre-attack'}

        # Check relevance: must be in target tactics OR be a priority technique
        base_id = ext_id.split('.')[0]
        if not (tactics & TARGET_TACTICS) and base_id not in PRIORITY_TECHNIQUE_IDS:
            continue

        # Skip non-enterprise
        domains = set(p.get('x_mitre_domains', []))
        if domains and 'enterprise-attack' not in domains:
            continue

        # Skip mobile-only
        platforms = set(p.get('x_mitre_platforms', []))
        if platforms and platforms <= {'Android', 'iOS'}:
            continue

        techniques.append({
            'id': ext_id,
            'stix_id': p['id'],
            'name': p.get('name', ''),
            'description': clean_description(p.get('description', '')),
            'detection': clean_description(p.get('x_mitre_detection', '')),
            'tactics': sorted(tactics),
            'platforms': sorted(platforms),
            'is_subtechnique': p.get('x_mitre_is_subtechnique', False),
            'mitigations': mitigation_map.get(p['id'], []),
            'used_by': uses_map.get(p['id'], []),
        })

    return techniques


# ---------------------------------------------------------------------------
# Example generation - domain knowledge answers
# ---------------------------------------------------------------------------

# Detailed knowledge for priority techniques
PRIORITY_KNOWLEDGE = {
    "T1486": {
        "ransomware_usage": (
            "Ransomware groups use data encryption as their primary extortion mechanism. "
            "The attack typically begins after initial access and lateral movement are complete, "
            "with the ransomware binary iterating through file system directories to encrypt documents, "
            "databases, images, and other valuable files using symmetric encryption (often AES-256) "
            "with the key itself encrypted by an RSA public key. Groups like Conti, LockBit, and "
            "BlackCat enumerate local and network drives, skip system-critical files to keep the host "
            "bootable, append custom extensions (e.g., .lockbit, .conti), and drop ransom notes in "
            "each directory. Modern variants use intermittent encryption (encrypting only portions of "
            "each file) to speed up the process and evade entropy-based detection."
        ),
        "detection": (
            "Detection of ransomware encryption activity focuses on several telemetry sources. "
            "Monitor for rapid file modification events across many directories, especially when "
            "file extensions change to unknown values. Track Shannon entropy of written files - "
            "encrypted content approaches 8.0 bits per byte. Watch for processes performing bulk "
            "file I/O operations with WriteFile and MoveFile API calls at high rates. Monitor "
            "Windows Event Log for Event ID 4663 (object access) showing mass file access patterns. "
            "Check for deletion of Volume Shadow Copies (vssadmin.exe, wmic.exe) which typically "
            "precedes encryption. EDR telemetry should flag processes that enumerate and modify files "
            "across multiple directory trees in rapid succession."
        ),
        "artifacts": (
            "Ransomware encryption leaves several forensic artifacts. Ransom notes (often README.txt "
            "or similar) are dropped in encrypted directories. File system timestamps show a burst of "
            "modification times within a narrow window. The MFT ($MFT) records show mass RENAME "
            "operations as files get new extensions. Prefetch files (C:\\Windows\\Prefetch) record "
            "execution of the ransomware binary. Windows Event Logs may contain Event ID 1 (Sysmon "
            "process creation), Event ID 11 (file creation), and Event ID 4663 (object access). "
            "Registry keys may contain ransomware configuration or victim ID. Memory forensics may "
            "reveal encryption keys if the process is still running. Network artifacts include DNS "
            "lookups to C2 domains and potential exfiltration traffic before encryption began."
        ),
        "investigation_scenario": (
            "You are investigating a Windows 10 workstation where users reported being unable to "
            "open their documents. Files across multiple directories now have a .locked extension "
            "and ransom notes appear in every folder."
        ),
        "investigation_walkthrough": (
            "First, isolate the host from the network to prevent further lateral spread while "
            "preserving volatile evidence. Begin with memory acquisition using a tool like WinPMEM "
            "to capture potential encryption keys still in memory.\n\n"
            "Examine running and recently terminated processes using Task Manager or Process Explorer, "
            "and cross-reference with Sysmon Event ID 1 logs. Look for suspicious processes with high "
            "file I/O rates - the ransomware process may still be running or may have terminated after "
            "completing encryption.\n\n"
            "Check the MFT for mass file rename operations by parsing $MFT with tools like MFTECmd. "
            "The timestamps will reveal when encryption started and ended, giving you the execution "
            "window. Examine Prefetch files in C:\\Windows\\Prefetch for recently executed binaries "
            "you don't recognize.\n\n"
            "Review Windows Event Logs: Security log Event ID 4688 for process creation, PowerShell "
            "Event ID 4104 for script block logging, and Application log for any ransomware-related "
            "entries. Check for VSS deletion commands in command-line audit logs.\n\n"
            "Inspect the registry for persistence mechanisms - check Run/RunOnce keys, scheduled "
            "tasks, and services. Examine HKCU\\Software and HKLM\\Software for unfamiliar keys "
            "that may store ransomware configuration or victim IDs.\n\n"
            "Analyze network connections using netstat output or Sysmon Event ID 3 to identify C2 "
            "communication. Check DNS cache and browser history for suspicious domains contacted "
            "before and during encryption. Look for evidence of data exfiltration in firewall logs.\n\n"
            "Finally, collect samples of the ransomware binary, encrypted files, and ransom notes "
            "for threat intelligence correlation. The ransom note contents, file extension, and "
            "encryption behavior can help identify the specific ransomware family."
        ),
    },
    "T1490": {
        "ransomware_usage": (
            "Inhibiting system recovery is a near-universal precursor to ransomware encryption. "
            "Adversaries delete Volume Shadow Copies using vssadmin.exe delete shadows /all /quiet "
            "or wmic shadowcopy delete to prevent file restoration. They also disable Windows Recovery "
            "Environment with bcdedit /set {default} recoveryenabled No and bcdedit /set {default} "
            "bootstatuspolicy ignoreallfailures. Some variants disable System Restore via registry "
            "modification (HKLM\\SOFTWARE\\Policies\\Microsoft\\Windows NT\\SystemRestore "
            "DisableSR=1) and delete Windows backup catalogs with wbadmin delete catalog -quiet. "
            "Groups like Ryuk, Conti, and LockBit routinely chain these commands together in batch "
            "scripts executed immediately before the encryption payload begins."
        ),
        "detection": (
            "Monitor for execution of vssadmin.exe with 'delete shadows' arguments, wmic.exe with "
            "'shadowcopy delete', and bcdedit.exe with '/set' and 'recoveryenabled' arguments. "
            "Sysmon Event ID 1 and Windows Security Event ID 4688 with command-line auditing enabled "
            "will capture these executions. Watch for Windows Event ID 524 (System Catalog deleted) "
            "in the Application log. Monitor registry modifications to SystemRestore-related keys. "
            "The Volume Shadow Copy Service (VSS) also logs events when shadows are deleted. "
            "An EDR or SIEM rule correlating VSS deletion with subsequent high-volume file "
            "modifications provides high-confidence ransomware detection."
        ),
        "artifacts": (
            "Key artifacts include command-line history showing vssadmin, wmic, bcdedit, and wbadmin "
            "executions. Prefetch files for these system utilities will show recent execution times. "
            "The Windows Event Log Application channel records VSS-related events. The BCD store "
            "(C:\\Boot\\BCD) will reflect modified boot configuration. Registry forensics reveals "
            "changes to System Restore policies. If PowerShell was used, Script Block Logging "
            "(Event ID 4104) captures the commands. Parent-child process relationships in Sysmon "
            "logs often show the ransomware binary spawning these recovery-inhibiting processes."
        ),
        "investigation_scenario": (
            "You are investigating a Windows Server 2019 system where the IT team noticed all "
            "Volume Shadow Copies were deleted and System Restore was disabled. Shortly after, "
            "files began showing encryption artifacts."
        ),
        "investigation_walkthrough": (
            "Start by examining process execution logs. Query Windows Security Event ID 4688 and "
            "Sysmon Event ID 1 for executions of vssadmin.exe, wmic.exe, bcdedit.exe, and "
            "wbadmin.exe. Note the timestamps and parent process for each - the parent process "
            "is likely the ransomware binary or a dropper script.\n\n"
            "Check PowerShell logs (Event ID 4104) for script-based VSS deletion, as many ransomware "
            "families use PowerShell to chain recovery-inhibiting commands. Look for Get-WmiObject "
            "Win32_ShadowCopy | ForEach-Object { $_.Delete() } patterns.\n\n"
            "Examine the registry for modifications: check HKLM\\SOFTWARE\\Policies\\Microsoft\\"
            "Windows NT\\SystemRestore for DisableSR values, and HKLM\\SYSTEM\\CurrentControlSet\\"
            "Services\\VSS for service tampering. Use Registry Explorer to check last-written "
            "timestamps on these keys.\n\n"
            "Parse Prefetch files to determine exactly when vssadmin.exe and related tools were "
            "last run. Cross-reference these times with the file modification burst to establish "
            "the attack timeline - VSS deletion almost always occurs within minutes before "
            "encryption begins.\n\n"
            "Review the Application event log for VSS errors and warnings. Event ID 8193 and 13 "
            "indicate VSS failures. Check the System event log for service state changes that "
            "might indicate the VSS service was stopped.\n\n"
            "Trace the parent process chain to identify the initial execution vector and determine "
            "whether the recovery inhibition was triggered by a batch script, PowerShell, or "
            "directly by the ransomware executable."
        ),
    },
    "T1562": {
        "ransomware_usage": (
            "Ransomware operators routinely disable security tools before deploying the encryption "
            "payload. Common techniques include disabling Windows Defender via PowerShell "
            "(Set-MpPreference -DisableRealtimeMonitoring $true), stopping antivirus services "
            "(sc stop WinDefend), modifying Group Policy to prevent Defender operation, and killing "
            "security product processes. Groups like BlackCat use the driver vulnerability technique "
            "(BYOVD - Bring Your Own Vulnerable Driver) to load a signed but vulnerable kernel driver "
            "that can terminate EDR processes from kernel mode. Others modify Windows Firewall rules "
            "to allow C2 traffic, tamper with Event Log Service to impede forensic analysis, or use "
            "tools like GMER and PCHunter to unhook security product callbacks from the kernel."
        ),
        "detection": (
            "Monitor for changes to Windows Defender settings via PowerShell commands containing "
            "Set-MpPreference, especially -DisableRealtimeMonitoring, -DisableIOAVProtection, and "
            "-DisableBehaviorMonitoring. Watch for sc.exe commands targeting security services "
            "(WinDefend, MsMpSvc, SecurityHealthService). Track registry modifications under "
            "HKLM\\SOFTWARE\\Policies\\Microsoft\\Windows Defender. Monitor for suspicious driver "
            "loads (Sysmon Event ID 6) that could indicate BYOVD attacks. Watch for processes "
            "being terminated that belong to known security products. Event ID 1116 (Windows "
            "Defender malware detected) being absent after previously appearing regularly can "
            "indicate Defender was disabled."
        ),
        "artifacts": (
            "Forensic artifacts include modified registry keys under Windows Defender policy paths, "
            "Group Policy template changes in SYSVOL, Windows Defender operational logs showing "
            "service stops and configuration changes (Event ID 5001 for real-time protection "
            "disabled, Event ID 5010 for scanning disabled). Sysmon Event ID 13 captures registry "
            "value modifications. Loaded driver artifacts in the System event log indicate BYOVD "
            "usage. The Windows Firewall log at "
            "%SystemRoot%\\System32\\LogFiles\\Firewall\\pfirewall.log may show rule additions. "
            "Service Control Manager events (Event ID 7045 for new service, 7040 for service "
            "type change) record security service tampering."
        ),
        "investigation_scenario": (
            "You are investigating a Windows 10 endpoint where the SOC detected that Windows "
            "Defender was disabled and the EDR agent stopped reporting telemetry approximately "
            "30 minutes before a ransomware encryption event."
        ),
        "investigation_walkthrough": (
            "Begin by examining Windows Defender operational logs under Applications and Services "
            "Logs > Microsoft > Windows > Windows Defender > Operational. Look for Event ID 5001 "
            "(real-time protection disabled), Event ID 5010 (scan disabled), and Event ID 5004 "
            "(configuration changed). Note the exact timestamps.\n\n"
            "Check registry forensics for the Defender policy keys: HKLM\\SOFTWARE\\Policies\\"
            "Microsoft\\Windows Defender and its subkeys. Look for DisableAntiSpyware, "
            "DisableRealtimeMonitoring, and DisableBehaviorMonitoring values. Use Registry "
            "Explorer to determine when these keys were last written.\n\n"
            "Review process execution logs for PowerShell commands containing Set-MpPreference "
            "or Add-MpPreference, and for sc.exe or net.exe commands targeting security services. "
            "Check Sysmon Event ID 1 for the full command lines and parent process information.\n\n"
            "Investigate potential BYOVD attacks by examining Sysmon Event ID 6 (driver loaded) "
            "for recently loaded drivers, especially any that are signed but known-vulnerable. "
            "Cross-reference driver hashes with known BYOVD driver databases like LOLDrivers.\n\n"
            "Examine the EDR agent's local logs or status files to determine if the agent was "
            "stopped, uninstalled, or its communication was blocked via firewall rules. Check "
            "Windows Firewall configuration for recently added rules that might block EDR traffic.\n\n"
            "Correlate the defense impairment timeline with subsequent activity - the gap between "
            "Defender being disabled and encryption starting often contains lateral movement, "
            "credential theft, or data staging activity."
        ),
    },
    "T1059": {
        "ransomware_usage": (
            "Command and scripting interpreters are heavily used throughout the ransomware kill "
            "chain. PowerShell (T1059.001) is the most common, used for downloading payloads "
            "(Invoke-WebRequest, Net.WebClient), disabling defenses, and executing in-memory "
            "malware via Invoke-Expression or IEX. cmd.exe (T1059.003) executes batch scripts "
            "that chain VSS deletion, service stopping, and payload execution. Windows Script "
            "Host (T1059.005) runs VBScript/JScript droppers often delivered via phishing emails. "
            "Python (T1059.006) is used by some groups for cross-platform ransomware or tooling. "
            "Ransomware families like Ryuk use PowerShell for reconnaissance (Get-ADComputer), "
            "Emotet uses VBScript macros for initial delivery, and Conti uses batch scripts for "
            "mass deployment across SMB shares."
        ),
        "detection": (
            "Enable PowerShell Script Block Logging (Event ID 4104) and Module Logging (Event ID "
            "4103) to capture all PowerShell execution. Monitor for encoded commands via "
            "powershell.exe -enc or -EncodedCommand flags. Track cmd.exe spawning child processes "
            "like powershell.exe, wmic.exe, or net.exe. Watch for cscript.exe and wscript.exe "
            "executing scripts from unusual locations (Temp, Downloads, AppData). Monitor for "
            "process creation chains where Office applications spawn command interpreters. "
            "Sysmon Event ID 1 with full command-line capture is essential. Look for obfuscation "
            "indicators like base64-encoded strings, string concatenation, and character replacement "
            "patterns in command lines."
        ),
        "artifacts": (
            "PowerShell artifacts include Script Block Logs (Event ID 4104), transcription logs "
            "if enabled (stored in a configured directory), PSReadLine history at "
            "%AppData%\\Microsoft\\Windows\\PowerShell\\PSReadLine\\ConsoleHost_history.txt, and "
            "the PowerShell operational log. cmd.exe usage is captured in process creation events "
            "and may leave batch file artifacts on disk. Windows Script Host logs script errors "
            "in the Application event log. Prefetch files record interpreter execution times. "
            "AMSI (Antimalware Scan Interface) logs capture script content before execution. "
            "WMI persistence via scripting creates artifacts in the WMI repository at "
            "C:\\Windows\\System32\\wbem\\Repository."
        ),
        "investigation_scenario": (
            "You are investigating a Windows workstation where PowerShell Script Block Logging "
            "captured a heavily obfuscated script that appears to download and execute a "
            "second-stage payload. The user received a phishing email with a macro-enabled "
            "Office document earlier that day."
        ),
        "investigation_walkthrough": (
            "Start by examining PowerShell Script Block Logs (Event ID 4104) in the Microsoft-"
            "Windows-PowerShell/Operational log. Deobfuscate the captured script - common "
            "techniques include base64 encoding, string reversal, character code conversion "
            "(e.g., [char]72+[char]84), and variable substitution. The Script Block Log captures "
            "the deobfuscated version when AMSI processes it.\n\n"
            "Trace the execution chain backwards using Sysmon Event ID 1. Identify what launched "
            "PowerShell - look for WINWORD.EXE, EXCEL.EXE, or wscript.exe as parent processes, "
            "which indicates macro or script-based delivery. Check the Office application's "
            "Recent Files for the triggering document.\n\n"
            "Examine network connections (Sysmon Event ID 3) from the PowerShell process to "
            "identify download URLs for the second stage. Check the DNS cache (ipconfig /displaydns) "
            "and proxy logs for domains contacted. Look in the user's browser history and email "
            "client for the original phishing email.\n\n"
            "Search for the downloaded payload in common staging locations: %TEMP%, %APPDATA%, "
            "%LOCALAPPDATA%, and the user's Downloads folder. Calculate file hashes for any "
            "suspicious binaries found and query threat intelligence platforms.\n\n"
            "Check for persistence mechanisms that the script may have established: scheduled tasks "
            "(schtasks.exe), registry Run keys, WMI event subscriptions, or new services. Review "
            "PSReadLine history for additional commands executed interactively.\n\n"
            "Determine the scope of impact by checking if the PowerShell process spawned additional "
            "child processes, made lateral movement attempts (WMI, PsExec, SMB), or accessed "
            "credential stores (LSASS, SAM, NTDS.dit)."
        ),
    },
    "T1547": {
        "ransomware_usage": (
            "Ransomware operators establish persistence through autostart execution to survive "
            "reboots and ensure re-encryption of newly created files. Common registry persistence "
            "locations include HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run, "
            "HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Run, and RunOnce variants. "
            "Some families create scheduled tasks (schtasks.exe) that re-execute the ransomware "
            "binary at login or on a timer. Others install themselves as Windows services or "
            "modify the Winlogon\\Shell or Userinit values. Ryuk and Conti create Run key entries "
            "to maintain access during the encryption phase. Maze and Egregor have used both "
            "registry persistence and scheduled tasks as redundant persistence mechanisms."
        ),
        "detection": (
            "Monitor registry modifications to known autostart locations using Sysmon Event ID 13 "
            "(registry value set) and Event ID 12 (registry key creation). Key paths include "
            "HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run, RunOnce, RunServices, and "
            "HKCU equivalents, as well as Winlogon\\Shell, Winlogon\\Userinit, and "
            "Explorer\\Shell Folders. Track new scheduled task creation via Event ID 4698 in the "
            "Security log and Sysmon Event ID 1 showing schtasks.exe execution. Monitor for new "
            "services (Event ID 7045 in the System log). Watch for modifications to Startup folder "
            "shortcuts. Autoruns or similar tools can baseline and diff autostart entries."
        ),
        "artifacts": (
            "Forensic artifacts include modified registry keys with timestamps available through "
            "registry hive analysis. The NTUSER.DAT hive contains HKCU autostart entries, while "
            "the SYSTEM and SOFTWARE hives hold HKLM entries. Scheduled task XML definitions are "
            "stored in C:\\Windows\\System32\\Tasks. The Security event log contains Event ID 4698 "
            "for task creation. Service installation records appear in the SYSTEM hive under "
            "CurrentControlSet\\Services. Startup folder items leave LNK files that can be parsed "
            "for target paths. AmCache.hve records program execution metadata. ShimCache entries "
            "in the SYSTEM hive track binary execution order."
        ),
        "investigation_scenario": (
            "You are investigating a Windows 10 system that was partially encrypted by ransomware. "
            "After the initial incident, the system was rebooted and encryption resumed, suggesting "
            "the ransomware established persistence."
        ),
        "investigation_walkthrough": (
            "Begin by enumerating all autostart locations. Use Autoruns or manually check the key "
            "registry paths: HKCU and HKLM Run and RunOnce keys, Winlogon Shell and Userinit "
            "values, and Explorer\\Shell Folders. Compare against a known-good baseline if "
            "available.\n\n"
            "Check scheduled tasks by examining C:\\Windows\\System32\\Tasks directory for "
            "recently created XML files. Parse each task for the action (executable path and "
            "arguments) and trigger conditions. Cross-reference with Security Event ID 4698 "
            "to get the exact creation time and the account that created the task.\n\n"
            "Examine the Services registry key (HKLM\\SYSTEM\\CurrentControlSet\\Services) for "
            "newly created services. Focus on services with ImagePath pointing to unusual "
            "locations (Temp, AppData, ProgramData). Check the System event log for Event ID 7045 "
            "(new service installed) entries.\n\n"
            "Review the Startup folders for both the affected user "
            "(C:\\Users\\<username>\\AppData\\Roaming\\Microsoft\\Windows\\Start Menu\\Programs\\"
            "Startup) and all users (C:\\ProgramData\\Microsoft\\Windows\\Start Menu\\Programs\\"
            "Startup) for suspicious shortcuts or executables.\n\n"
            "Parse the AmCache.hve and ShimCache to determine when the ransomware binary was first "
            "executed and whether it was executed from the persistence location or the original "
            "drop location. This helps establish the timeline of persistence creation.\n\n"
            "Once the persistence mechanism is identified, trace back to determine how it was "
            "created - was it the ransomware binary itself creating Run keys, or did a dropper "
            "script establish persistence before launching the encryption payload? Remove all "
            "identified persistence mechanisms before allowing the system to reboot again."
        ),
    },
    "T1021": {
        "ransomware_usage": (
            "Lateral movement via remote services is how ransomware spreads across enterprise "
            "networks. RDP (T1021.001) is abused using stolen credentials or brute-forced accounts "
            "to access additional hosts. SMB/Windows Admin Shares (T1021.002) are used to copy "
            "ransomware binaries to C$ or ADMIN$ shares on target systems and execute them remotely. "
            "PsExec or similar tools execute payloads over named pipes on SMB. WMI (via wmic.exe "
            "or PowerShell) provides another remote execution channel. WannaCry exploited EternalBlue "
            "(MS17-010) to spread via SMB without credentials. Conti operators use a combination of "
            "PsExec, RDP, and WMI to deploy across domain-joined systems. LockBit uses Group Policy "
            "Objects pushed via SMB to deploy across the Active Directory domain."
        ),
        "detection": (
            "Monitor for unusual RDP connections by analyzing Windows Security Event ID 4624 "
            "(logon type 10 for RDP) and Event ID 4625 (failed logons indicating brute force). "
            "Track SMB share access via Event ID 5140 (network share accessed) and 5145 (share "
            "object checked). Watch for PsExec artifacts: new service creation (Event ID 7045 "
            "with PSEXESVC), named pipe creation, and SMB connections to ADMIN$ or IPC$ shares. "
            "Monitor for remote WMI execution via Event ID 4688 showing wmiprvse.exe spawning "
            "processes. Network-level detection should flag lateral SMB traffic (port 445) and "
            "RDP traffic (port 3389) between workstations, which is unusual in most environments."
        ),
        "artifacts": (
            "RDP artifacts include bitmap cache files in %LOCALAPPDATA%\\Microsoft\\Terminal Server "
            "Client\\Cache, RDP connection history in NTUSER.DAT under Terminal Server Client\\Servers, "
            "and Event ID 1149 in TerminalServices-RemoteConnectionManager. SMB lateral movement "
            "leaves Event ID 5140/5145 on target systems. PsExec creates the PSEXESVC service and "
            "a named pipe (\\\\pipe\\psexesvc). WMI execution creates artifacts in the WMI "
            "repository and generates Event ID 5857-5861 in WMI-Activity. On source systems, "
            "ShimCache and AmCache record remote tool execution. Network forensics reveals SMB "
            "file transfer patterns and RDP session metadata."
        ),
        "investigation_scenario": (
            "You are investigating a network where ransomware has spread to 15 systems within "
            "an hour. The initial compromised system was a domain controller, and you need to "
            "determine how the ransomware propagated to other hosts."
        ),
        "investigation_walkthrough": (
            "Start with the domain controller timeline. Examine Security Event ID 4624 logs for "
            "logon types 3 (network), 10 (RDP), and 2 (interactive) to identify the initial "
            "compromise. Check Event ID 4672 for special privilege logon events that indicate "
            "admin-level access.\n\n"
            "Map the lateral movement by correlating Event ID 4624 across all affected systems. "
            "For each host, identify the source IP, account used, and logon type. Build a timeline "
            "showing the order of compromise. Look for Event ID 5140 (network share accessed) "
            "to identify SMB-based deployment.\n\n"
            "Check for PsExec usage by searching for PSEXESVC service installation (Event ID 7045) "
            "on target systems. If PsExec was used, the source system will show Prefetch entries "
            "for psexec.exe and the target will show the service creation and the ransomware "
            "process spawned by the service.\n\n"
            "Examine WMI activity logs (Microsoft-Windows-WMI-Activity/Operational) for remote "
            "process creation. Look for wmiprvse.exe spawning suspicious child processes on target "
            "systems. Check for PowerShell remoting via Event ID 4104 containing "
            "Invoke-Command -ComputerName patterns.\n\n"
            "On the domain controller, check if Group Policy was abused to deploy ransomware. "
            "Examine the SYSVOL share for recently modified GPOs, especially those containing "
            "immediate scheduled tasks or startup scripts. Check Event ID 5136 (directory service "
            "changes) for GPO modifications.\n\n"
            "Correlate network flow data (NetFlow, firewall logs) to visualize the lateral "
            "movement pattern - was it sequential (one system at a time) or parallel (mass "
            "deployment via GPO or scripted PsExec)? This helps determine the adversary's "
            "automation level and deployment method."
        ),
    },
    "T1003": {
        "ransomware_usage": (
            "Credential dumping is essential for ransomware operators to escalate privileges and "
            "enable lateral movement. Mimikatz or its variants extract plaintext passwords, NTLM "
            "hashes, and Kerberos tickets from LSASS process memory. The SAM database "
            "(C:\\Windows\\System32\\config\\SAM) can be extracted for local account hashes. "
            "NTDS.dit from domain controllers contains all domain account hashes. DCSync attacks "
            "use the Directory Replication Service (DRS) protocol to request password data without "
            "touching NTDS.dit. Groups like Conti, REvil, and LockBit use Mimikatz, LaZagne, or "
            "custom credential harvesting tools to obtain domain admin credentials before deploying "
            "ransomware domain-wide. Cached credentials in the registry and credentials stored in "
            "browsers or vault are also targeted."
        ),
        "detection": (
            "Monitor for LSASS memory access using Sysmon Event ID 10 (ProcessAccess) with "
            "TargetImage of lsass.exe. Watch for suspicious access masks like 0x1010 (PROCESS_VM_READ "
            "| PROCESS_QUERY_INFORMATION). Detect SAM/SYSTEM registry hive extraction via reg.exe "
            "save commands targeting HKLM\\SAM or HKLM\\SYSTEM. Monitor for ntdsutil.exe or "
            "volume shadow copy methods used to extract NTDS.dit. DCSync detection requires "
            "monitoring Event ID 4662 for DS-Replication-Get-Changes and DS-Replication-Get-Changes-All "
            "access rights from non-domain-controller sources. Windows Credential Guard and "
            "Protected Process Light (PPL) for LSASS provide preventive controls."
        ),
        "artifacts": (
            "LSASS access artifacts appear in Sysmon Event ID 10 with the source process and "
            "access mask. Mimikatz often creates a log file (mimikatz.log) or outputs to console. "
            "SAM extraction leaves reg.exe Prefetch artifacts and command-line evidence in process "
            "creation logs. NTDS.dit extraction via ntdsutil creates artifacts in "
            "C:\\Windows\\Temp\\NTDSAudit. DCSync attacks generate Event ID 4662 with specific "
            "replication GUIDs (1131f6aa-9c07-11d1-f79f-00c04fc2dcd2 for DS-Replication-Get-Changes). "
            "Credential dumping tools often require debug privileges, generating Event ID 4672. "
            "Memory dumps of LSASS (e.g., via procdump.exe, comsvcs.dll MiniDump) leave crash "
            "dump files and process access artifacts."
        ),
        "investigation_scenario": (
            "You are investigating a Windows Server 2019 domain controller where Sysmon detected "
            "suspicious access to the LSASS process from an unknown binary, followed by anomalous "
            "directory replication requests from a workstation IP address."
        ),
        "investigation_walkthrough": (
            "Begin by examining Sysmon Event ID 10 entries targeting lsass.exe. Identify the source "
            "process (SourceImage), its access mask, and call trace. Common credential dumping tools "
            "use specific access patterns: Mimikatz typically requests 0x1010 or 0x1FFFFF access "
            "to LSASS. Check if the source binary is a known tool or a renamed copy.\n\n"
            "Check for DCSync indicators in the Security log. Filter Event ID 4662 for the "
            "DS-Replication-Get-Changes-All GUID (1131f6ad-9c07-11d1-f79f-00c04fc2dcd2). The "
            "SubjectUserName and SubjectLogonId reveal which account performed the replication, "
            "and the source IP in the associated Event ID 4624 shows where the request originated. "
            "Legitimate replication only comes from other domain controllers.\n\n"
            "Examine the workstation that initiated the DCSync. Check its process creation logs "
            "for mimikatz.exe (possibly renamed), PowerShell with Invoke-Mimikatz, or DCSync "
            "modules from frameworks like Cobalt Strike. Review network connections from that host.\n\n"
            "Check for SAM/SYSTEM hive extraction attempts. Look for reg.exe with 'save' "
            "arguments targeting HKLM\\SAM, HKLM\\SECURITY, or HKLM\\SYSTEM in process "
            "creation logs. Also check for volume shadow copy creation followed by file access "
            "to the shadow copies, which is an alternative NTDS.dit extraction method.\n\n"
            "Review privilege escalation events. Event ID 4672 shows special privilege assignments - "
            "the credential dumping account likely obtained SeDebugPrivilege. Trace how this account "
            "gained its elevated access.\n\n"
            "Assess the blast radius: determine which credentials were exposed. If NTDS.dit or "
            "DCSync was successful, assume all domain accounts are compromised. Initiate a "
            "domain-wide password reset starting with service accounts and domain admins."
        ),
    },
    "T1071": {
        "ransomware_usage": (
            "Ransomware operators use application layer protocols for command-and-control "
            "communication, data exfiltration, and payload delivery. HTTPS (T1071.001) is the "
            "most common C2 channel as it blends with normal web traffic and is encrypted. DNS "
            "tunneling (T1071.004) exfiltrates data or receives commands encoded in DNS queries "
            "and responses. Some groups use legitimate cloud services (OneDrive, Google Drive, "
            "Slack APIs) as C2 channels to further blend into normal traffic. Cobalt Strike, a "
            "common post-exploitation framework used by ransomware groups, supports HTTP, HTTPS, "
            "and DNS C2 profiles with configurable malleable profiles that mimic legitimate "
            "application traffic. LockBit and BlackCat exfiltrate stolen data to attacker-controlled "
            "infrastructure over HTTPS before deploying encryption."
        ),
        "detection": (
            "Monitor for unusual DNS query patterns such as high-frequency queries to a single "
            "domain, abnormally long subdomain labels (indicating data encoding), or queries to "
            "newly registered domains. For HTTPS C2, analyze TLS certificate metadata - self-signed "
            "certificates, unusual certificate authorities, or certificates with anomalous validity "
            "periods are suspicious. Use JA3/JA3S fingerprinting to identify known malware TLS "
            "client/server signatures. Monitor for beaconing behavior: regular-interval HTTP/HTTPS "
            "connections to the same destination with similar payload sizes. Track outbound data "
            "volume anomalies that could indicate exfiltration. DNS-over-HTTPS (DoH) usage to "
            "non-standard resolvers is increasingly used to evade DNS monitoring."
        ),
        "artifacts": (
            "Network forensic artifacts include DNS query logs (Windows DNS Client Event ID 3008, "
            "or Sysmon Event ID 22), HTTP proxy logs showing C2 URLs and User-Agent strings, "
            "and firewall connection logs with destination IPs, ports, and byte counts. Packet "
            "captures (PCAP) allow deep inspection of unencrypted protocols and TLS metadata. "
            "On the endpoint, browser history may reveal C2 URLs accessed via web-based C2. "
            "Sysmon Event ID 3 captures network connections per process, correlating C2 traffic "
            "to the responsible malware process. DNS cache (ipconfig /displaydns) contains "
            "recently resolved domains. Cobalt Strike leaves distinctive named pipes and "
            "service patterns on the endpoint."
        ),
        "investigation_scenario": (
            "You are investigating network traffic from a compromised endpoint that shows regular "
            "HTTPS beaconing to an external IP every 60 seconds with small payloads, and unusually "
            "long DNS queries to a single domain, suggesting both HTTP and DNS C2 channels."
        ),
        "investigation_walkthrough": (
            "Start with network traffic analysis. Examine firewall or proxy logs for the HTTPS "
            "beaconing destination. Check the IP's reputation against threat intelligence feeds "
            "and look up WHOIS/registration data. Analyze the TLS certificate via the proxy's "
            "SSL inspection logs or a PCAP - note the issuer, subject, validity period, and "
            "JA3/JA3S hashes.\n\n"
            "Investigate the DNS anomaly. Query DNS logs for the suspicious domain and analyze "
            "the subdomain patterns. If data appears encoded in subdomain labels (base32/base64-like "
            "strings), this confirms DNS tunneling. Calculate the query frequency and total data "
            "volume to estimate what was exfiltrated.\n\n"
            "On the endpoint, use Sysmon Event ID 3 to identify which process is generating "
            "the C2 traffic. Correlate the destination IP and port with the network-level "
            "observations. Check the process's binary hash against threat intelligence and "
            "examine its parent process chain to understand the infection vector.\n\n"
            "Look for Cobalt Strike indicators if the beaconing pattern suggests it: check for "
            "default named pipes (\\\\pipe\\msagent_*), malleable C2 profile artifacts in "
            "HTTP headers, and characteristic sleep/jitter patterns. Memory analysis of the "
            "beaconing process may reveal injected Cobalt Strike shellcode.\n\n"
            "Assess data exfiltration scope. Correlate the DNS tunneling start time with file "
            "access events (Sysmon Event ID 11, Security Event ID 4663) to determine what data "
            "was accessed before exfiltration. Check for staging directories where data may have "
            "been collected before transmission.\n\n"
            "Block the identified C2 channels at the network perimeter (firewall, DNS sinkhole) "
            "and search for the same indicators across all endpoints to identify additional "
            "compromised systems."
        ),
    },
}


def generate_qa_what(technique):
    """Generate 'What is [technique]?' QA pair."""
    desc = technique['description']
    # Truncate very long descriptions to 3-4 sentences
    sentences = desc.split('. ')
    if len(sentences) > 5:
        answer = '. '.join(sentences[:5]) + '.'
    else:
        answer = desc
    # Add tactic context
    tactic_str = ', '.join(technique['tactics'])
    answer = f"{technique['name']} ({technique['id']}) is a technique in the {tactic_str} tactic(s). {answer}"
    return make_example(
        f"What is the MITRE ATT&CK technique {technique['name']} ({technique['id']})?",
        answer,
        technique['id'],
        "qa",
        "technique_qa_v1",
    )


def generate_qa_ransomware_usage(technique):
    """Generate 'How do ransomware groups use [technique]?' QA pair."""
    tid = technique['id']
    base_id = tid.split('.')[0]

    # Use detailed knowledge for priority techniques
    if base_id in PRIORITY_KNOWLEDGE and 'ransomware_usage' in PRIORITY_KNOWLEDGE[base_id] and not technique['is_subtechnique']:
        answer = PRIORITY_KNOWLEDGE[base_id]['ransomware_usage']
    else:
        # Generate from description and usage data
        used_by_malware = [u for u in technique['used_by'] if u['type'] == 'malware']
        used_by_groups = [u for u in technique['used_by'] if u['type'] == 'intrusion-set']

        answer = f"Ransomware operators may leverage {technique['name']} as part of their attack chain. "
        answer += technique['description'][:500]
        if not answer.endswith('.'):
            answer = answer.rsplit('.', 1)[0] + '.'

        if used_by_malware:
            names = list(set(u['name'] for u in used_by_malware[:5]))
            answer += f" Known malware families using this technique include {', '.join(names)}."
        if used_by_groups:
            names = list(set(u['name'] for u in used_by_groups[:3]))
            answer += f" Threat groups such as {', '.join(names)} have been observed employing this technique."

    return make_example(
        f"How do ransomware groups use {technique['name']} ({technique['id']}) in their operations?",
        answer,
        technique['id'],
        "qa",
        "technique_qa_v1",
    )


def generate_qa_detection(technique):
    """Generate 'How do you detect [technique]?' QA pair."""
    tid = technique['id']
    base_id = tid.split('.')[0]

    if base_id in PRIORITY_KNOWLEDGE and 'detection' in PRIORITY_KNOWLEDGE[base_id] and not technique['is_subtechnique']:
        answer = PRIORITY_KNOWLEDGE[base_id]['detection']
    elif technique['detection']:
        answer = technique['detection']
    else:
        # Generate generic detection guidance
        answer = (
            f"Detection of {technique['name']} involves monitoring process creation events "
            f"(Sysmon Event ID 1, Windows Security Event ID 4688) for suspicious command lines "
            f"and parent-child process relationships. "
            f"Review Windows Event Logs for anomalous activity correlated with this technique. "
            f"Endpoint Detection and Response (EDR) tools can be configured with behavioral rules "
            f"to detect the patterns associated with {technique['name']}. "
            f"Network monitoring may reveal associated command-and-control or lateral movement traffic."
        )

    return make_example(
        f"How do you detect {technique['name']} ({technique['id']}) on a Windows host?",
        answer,
        technique['id'],
        "qa",
        "technique_qa_v1",
        topics=["ransomware_ttps", "detection"],
    )


def generate_qa_artifacts(technique):
    """Generate 'What artifacts does [technique] leave?' QA pair."""
    tid = technique['id']
    base_id = tid.split('.')[0]

    if base_id in PRIORITY_KNOWLEDGE and 'artifacts' in PRIORITY_KNOWLEDGE[base_id] and not technique['is_subtechnique']:
        answer = PRIORITY_KNOWLEDGE[base_id]['artifacts']
    else:
        answer = (
            f"{technique['name']} may leave several forensic artifacts on a Windows system. "
            f"Process execution evidence includes Prefetch files (C:\\Windows\\Prefetch), "
            f"ShimCache entries in the SYSTEM registry hive, and AmCache.hve records. "
        )
        if technique['detection']:
            answer += technique['detection'][:400]
            if not answer.endswith('.'):
                answer = answer.rsplit('.', 1)[0] + '.'
        else:
            answer += (
                f"Sysmon Event ID 1 records process creation with full command lines and parent process "
                f"information. Windows Security Event ID 4688 provides similar data when command-line "
                f"auditing is enabled. File system timestamps, MFT entries, and USN journal records "
                f"can reveal file creation and modification patterns associated with this technique."
            )

    return make_example(
        f"What forensic artifacts does {technique['name']} ({technique['id']}) leave on a Windows system?",
        answer,
        technique['id'],
        "qa",
        "technique_qa_v1",
        topics=["ransomware_ttps", "forensics"],
    )


def generate_qa_mitigations(technique):
    """Generate 'What are the mitigations for [technique]?' QA pair."""
    if technique['mitigations']:
        mitigation_parts = []
        for m in technique['mitigations'][:5]:
            desc = m['description'][:200]
            if not desc.endswith('.'):
                desc = desc.rsplit('.', 1)[0] + '.'
            mitigation_parts.append(f"{m['name']}: {desc}")
        answer = (
            f"MITRE recommends the following mitigations for {technique['name']} ({technique['id']}): "
            + " ".join(mitigation_parts)
        )
    else:
        answer = (
            f"{technique['name']} ({technique['id']}) has limited specific mitigations in the MITRE "
            f"ATT&CK framework. General defensive measures include network segmentation to limit "
            f"lateral movement, principle of least privilege for user and service accounts, application "
            f"whitelisting to prevent unauthorized execution, endpoint detection and response (EDR) "
            f"solutions with behavioral detection capabilities, and comprehensive logging and monitoring "
            f"to enable rapid detection and response."
        )

    return make_example(
        f"What are the recommended mitigations for {technique['name']} ({technique['id']})?",
        answer,
        technique['id'],
        "qa",
        "technique_qa_v1",
        topics=["ransomware_ttps", "mitigations"],
    )


def generate_instruction_example(technique):
    """Generate an investigation scenario instruction example."""
    tid = technique['id']
    base_id = tid.split('.')[0]

    if base_id in PRIORITY_KNOWLEDGE and not technique['is_subtechnique']:
        pk = PRIORITY_KNOWLEDGE[base_id]
        scenario = pk.get('investigation_scenario', '')
        walkthrough = pk.get('investigation_walkthrough', '')
        if scenario and walkthrough:
            return make_example(
                scenario + " Walk through your investigation process step by step.",
                walkthrough,
                technique['id'],
                "instruction",
                "technique_instruction_v1",
                topics=["ransomware_ttps", "incident_response"],
            )

    # Generate a generic investigation instruction
    tactic_str = ', '.join(technique['tactics'])
    name = technique['name']

    scenario = (
        f"You are investigating a Windows host where evidence suggests {name} "
        f"({technique['id']}) was used as part of a ransomware attack. "
        f"This technique falls under the {tactic_str} tactic(s). "
        f"Walk through your investigation process."
    )

    walkthrough = (
        f"Begin by collecting volatile evidence before it is lost. Capture a memory image and "
        f"record running processes, network connections, and logged-in users.\n\n"
    )

    walkthrough += (
        f"Examine process creation logs (Sysmon Event ID 1, Security Event ID 4688) for "
        f"evidence of {name}. Look for suspicious command-line arguments, unusual parent-child "
        f"process relationships, and processes running from temporary or user-writable directories.\n\n"
    )

    if technique['detection']:
        det = technique['detection'][:600]
        if not det.endswith('.'):
            det = det.rsplit('.', 1)[0] + '.'
        walkthrough += f"Specific detection guidance for this technique: {det}\n\n"

    walkthrough += (
        f"Check for related indicators by examining Windows Event Logs, registry modifications "
        f"(Sysmon Event ID 13), file system changes (Sysmon Event ID 11), and network connections "
        f"(Sysmon Event ID 3). Build a timeline correlating {name} activity with other "
        f"ransomware indicators such as defense evasion, credential access, and encryption events.\n\n"
    )

    walkthrough += (
        f"Assess the scope of compromise by searching for the same indicators across other hosts "
        f"in the environment. Determine whether the technique was used for initial access, "
        f"privilege escalation, lateral movement, or pre-encryption preparation, and trace the "
        f"full attack chain."
    )

    return make_example(
        scenario,
        walkthrough,
        technique['id'],
        "instruction",
        "technique_instruction_v1",
        topics=["ransomware_ttps", "incident_response"],
    )


# ---------------------------------------------------------------------------
# Additional QA generators for priority techniques (to reach ~400 examples)
# ---------------------------------------------------------------------------

def generate_priority_extra_qa(technique):
    """Generate extra QA pairs for priority techniques."""
    tid = technique['id']
    extras = []

    if tid == "T1486":
        extras.append(make_example(
            "What encryption methods do modern ransomware families typically use?",
            "Modern ransomware families typically use a hybrid encryption scheme combining symmetric "
            "and asymmetric cryptography. The ransomware generates a unique AES-256 (or ChaCha20) "
            "symmetric key per file or per session, which performs the actual file encryption due to "
            "its speed. This symmetric key is then encrypted with an RSA-2048 or RSA-4096 public key "
            "embedded in the ransomware binary. The encrypted symmetric key is stored alongside the "
            "encrypted file (often appended or prepended). Only the attacker's private RSA key can "
            "decrypt the symmetric key, which is needed to recover the files. Some families like "
            "LockBit use intermittent encryption, encrypting only every Nth byte or the first N "
            "megabytes of each file, dramatically speeding up encryption while still rendering "
            "files unusable. This approach also reduces the entropy signature that security tools "
            "use for detection.",
            "T1486", "qa", "technique_qa_v1",
            topics=["ransomware_ttps", "cryptography"],
        ))
        extras.append(make_example(
            "How can you differentiate ransomware encryption from legitimate encryption software "
            "like BitLocker or VeraCrypt on a Windows system?",
            "Ransomware encryption differs from legitimate encryption in several observable ways. "
            "BitLocker encrypts entire volumes and is managed through the TPM and BDE (BitLocker "
            "Drive Encryption) service, with Event IDs 24577-24620 in the BitLocker-API log. "
            "Ransomware encrypts individual files selectively, leaving the OS bootable. Key "
            "indicators of ransomware include: file extension changes to non-standard values "
            "across many files simultaneously, ransom notes dropped in affected directories, "
            "modification timestamps clustered in a narrow time window, high file I/O from a "
            "single non-system process, file entropy approaching 8.0 bits/byte on previously "
            "low-entropy documents, and the process responsible running from a user-writable "
            "directory rather than Program Files. Legitimate encryption tools are installed in "
            "standard paths, have valid digital signatures, and do not modify files across the "
            "entire file system in rapid succession.",
            "T1486", "qa", "technique_qa_v1",
            topics=["ransomware_ttps", "detection", "forensics"],
        ))
        extras.append(make_example(
            "What Windows Event IDs are most relevant for detecting ransomware file encryption activity?",
            "Several Windows Event IDs are critical for detecting ransomware encryption. Sysmon "
            "Event ID 11 (FileCreate) detects ransom note creation and encrypted file writes. "
            "Event ID 23 (FileDelete) captures file deletion that precedes replacement with encrypted "
            "copies. Security Event ID 4663 (object access audit) shows mass file access when "
            "configured on sensitive directories. Sysmon Event ID 1 (ProcessCreate) captures the "
            "ransomware process execution with full command line. Event ID 4688 (process creation) "
            "provides similar data with audit policy enabled. Sysmon Event ID 2 (FileCreationTime "
            "change) detects timestamp manipulation. For detection correlation, combine high-frequency "
            "Event ID 4663 events from a single process with Sysmon Event ID 15 (FileCreateStreamHash) "
            "showing files with high entropy alternate data streams. The VSS deletion events "
            "(Event ID 524 in Application log) typically precede encryption by minutes.",
            "T1486", "qa", "technique_qa_v1",
            topics=["ransomware_ttps", "detection", "event_logs"],
        ))
    elif tid == "T1490":
        extras.append(make_example(
            "What specific command-line patterns indicate Volume Shadow Copy deletion on Windows?",
            "Volume Shadow Copy deletion uses several recognizable command patterns. The most "
            "common is vssadmin.exe delete shadows /all /quiet, where the /quiet flag suppresses "
            "confirmation prompts. Alternatively, wmic shadowcopy delete achieves the same result "
            "through WMI. PowerShell variants include Get-WmiObject Win32_ShadowCopy | ForEach-Object "
            "{ $_.Delete() } and Get-CimInstance Win32_ShadowCopy | Remove-CimInstance. Some "
            "ransomware uses the COM-based Volume Shadow Copy API directly via VSSCoordinatorProc, "
            "bypassing command-line tools entirely. Additional recovery inhibition commands include: "
            "bcdedit /set {default} recoveryenabled No, bcdedit /set {default} bootstatuspolicy "
            "ignoreallfailures, wbadmin delete catalog -quiet (deletes Windows Backup catalog), "
            "and wbadmin delete systemstatebackup -keepVersions:0. Detection rules should cover "
            "all these variants including case variations and argument reordering.",
            "T1490", "qa", "technique_qa_v1",
            topics=["ransomware_ttps", "detection"],
        ))
        extras.append(make_example(
            "Why is Volume Shadow Copy deletion considered a high-confidence indicator of ransomware?",
            "Volume Shadow Copy deletion is a high-confidence ransomware indicator because the "
            "operation has very few legitimate use cases and is performed by nearly every ransomware "
            "family. Legitimate administrators rarely delete all shadow copies simultaneously - they "
            "might manage them individually or adjust storage allocation. In contrast, ransomware "
            "needs to delete shadows to prevent victims from using Previous Versions or System "
            "Restore to recover encrypted files without paying the ransom. The combination of "
            "VSS deletion with other indicators (defense evasion, file modification bursts) provides "
            "near-certain ransomware confirmation. When you see vssadmin delete shadows /all in "
            "the context of a non-administrative session or from a suspicious parent process, the "
            "false positive rate is extremely low. Security teams should treat VSS deletion as a "
            "critical alert requiring immediate investigation and potential automated containment.",
            "T1490", "qa", "technique_qa_v1",
            topics=["ransomware_ttps", "detection"],
        ))
    elif tid == "T1562":
        extras.append(make_example(
            "What is a Bring Your Own Vulnerable Driver (BYOVD) attack and how do ransomware "
            "groups use it to disable security tools?",
            "BYOVD (Bring Your Own Vulnerable Driver) is a technique where attackers load a "
            "legitimately signed but vulnerability-containing kernel driver onto a target system, "
            "then exploit that vulnerability to gain kernel-level access. Because the driver has a "
            "valid digital signature from a trusted publisher, Windows allows it to load even with "
            "driver signature enforcement enabled. From kernel mode, the attacker can terminate EDR "
            "processes that are protected at user level, remove callback registrations that security "
            "tools use to monitor system activity, and modify kernel memory to unhook detection "
            "mechanisms. Ransomware groups including BlackCat (ALPHV), Cuba, and AvosLocker have "
            "deployed BYOVD attacks using drivers from companies like Gigabyte, Micro-Star, and "
            "others. The LOLDrivers project maintains a database of known vulnerable drivers. "
            "Defenses include Windows Defender Application Control (WDAC) driver blocklist policies "
            "and Hypervisor-Protected Code Integrity (HVCI).",
            "T1562", "qa", "technique_qa_v1",
            topics=["ransomware_ttps", "defense_evasion"],
        ))
    elif tid == "T1059":
        extras.append(make_example(
            "What PowerShell obfuscation techniques do ransomware operators commonly use to "
            "evade detection?",
            "Ransomware operators employ several PowerShell obfuscation layers. String concatenation "
            "breaks recognizable commands into fragments: ('Inv'+'oke'+'-Web'+'Request'). Base64 "
            "encoding via -EncodedCommand (-enc) flag hides the entire script. Character code "
            "conversion uses [char] casts: ([char]73+[char]69+[char]88) spells 'IEX'. Variable "
            "substitution assigns command fragments to variables that are assembled at runtime. "
            "Tick-mark insertion adds backticks between characters: I`nv`oke-`Web`Req`uest. "
            "Environment variable slicing extracts substrings: $env:ComSpec[4,15,25]-join''. "
            "Script block logging (Event ID 4104) is the primary defense because AMSI evaluates "
            "the deobfuscated script before execution, and the deobfuscated version is what gets "
            "logged. However, AMSI bypass techniques (patching amsi.dll in memory) can circumvent "
            "this. Layered defenses should include Constrained Language Mode, AppLocker policies "
            "restricting PowerShell, and monitoring for powershell.exe with -enc flag.",
            "T1059", "qa", "technique_qa_v1",
            topics=["ransomware_ttps", "execution", "detection"],
        ))
    elif tid == "T1547":
        extras.append(make_example(
            "What are the most common Windows registry autostart locations used by ransomware "
            "for persistence?",
            "Ransomware uses several registry autostart locations for persistence. The most common "
            "are HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run and the HKLM equivalent, "
            "which execute programs at every user logon. RunOnce variants execute once then delete "
            "the entry. Less obvious locations include: HKLM\\SOFTWARE\\Microsoft\\Windows NT\\"
            "CurrentVersion\\Winlogon (Shell and Userinit values), HKLM\\SYSTEM\\"
            "CurrentControlSet\\Control\\Session Manager (BootExecute for pre-logon execution), "
            "HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\User Shell Folders "
            "(redirecting Startup folder), and HKLM\\SOFTWARE\\Microsoft\\Windows\\"
            "CurrentVersion\\Explorer\\SharedTaskScheduler. Registry-based persistence is preferred "
            "by ransomware because it requires no additional files (the binary path is simply "
            "referenced) and survives reboots. Detection requires monitoring Sysmon Event ID 13 "
            "(registry value set) on these specific keys. Autoruns can enumerate all autostart "
            "locations for a comprehensive audit.",
            "T1547", "qa", "technique_qa_v1",
            topics=["ransomware_ttps", "persistence"],
        ))
    elif tid == "T1021":
        extras.append(make_example(
            "How do ransomware groups use PsExec and SMB admin shares for mass deployment across "
            "a Windows domain?",
            "Ransomware mass deployment via PsExec and SMB admin shares follows a pattern. After "
            "obtaining domain admin credentials, operators enumerate domain-joined systems using "
            "Active Directory queries (net group 'Domain Computers' /domain or PowerShell "
            "Get-ADComputer). They then copy the ransomware binary to target systems via the C$ "
            "or ADMIN$ administrative shares (e.g., copy ransom.exe \\\\target\\C$\\Windows\\Temp). "
            "PsExec connects to each target's ADMIN$ share, deploys the PSEXESVC service, and "
            "executes the binary remotely. Some groups script this with batch files or PowerShell "
            "loops to hit hundreds of systems simultaneously. Conti operators famously used a "
            "batch script iterating over a list of IPs to deploy via PsExec. Detection focuses on "
            "Event ID 7045 (service installation) for PSEXESVC, Event ID 5140 for admin share "
            "access, and workstation-to-workstation SMB traffic which is abnormal in most networks. "
            "Network segmentation and restricting admin share access are key preventive controls.",
            "T1021", "qa", "technique_qa_v1",
            topics=["ransomware_ttps", "lateral_movement"],
        ))
    elif tid == "T1003":
        extras.append(make_example(
            "How do you detect Mimikatz and LSASS credential dumping on a Windows system?",
            "Detecting LSASS credential dumping requires multiple telemetry sources. The primary "
            "indicator is Sysmon Event ID 10 (ProcessAccess) with TargetImage lsass.exe and "
            "suspicious GrantedAccess masks: 0x1010 (PROCESS_VM_READ | PROCESS_QUERY_INFORMATION), "
            "0x1FFFFF (PROCESS_ALL_ACCESS), or 0x1410. The SourceImage reveals the dumping tool. "
            "For procdump-style attacks, watch for lsass.exe appearing as a command-line argument "
            "to procdump.exe, rundll32.exe (with comsvcs.dll MiniDump), or taskmgr.exe creating "
            "dump files. Monitor for the creation of .dmp files in temp directories. Windows "
            "Credential Guard isolates LSASS credentials in a virtualization-based security "
            "enclave, preventing in-memory extraction. Running LSASS as a Protected Process Light "
            "(PPL) via the RunAsPPL registry setting blocks most user-mode access. For Mimikatz "
            "specifically, look for its default output patterns in file system scans, and check "
            "for debug privilege assignment (Event ID 4672 with SeDebugPrivilege) to non-admin "
            "processes.",
            "T1003", "qa", "technique_qa_v1",
            topics=["ransomware_ttps", "credential_access", "detection"],
        ))

    return extras


# ---------------------------------------------------------------------------
# Additional cross-technique QA pairs for diversity
# ---------------------------------------------------------------------------

def generate_cross_technique_examples():
    """Generate QA examples that span multiple techniques for attack-chain understanding."""
    examples = []

    examples.append(make_example(
        "Describe the typical attack chain used by a modern ransomware group from initial "
        "access to encryption, referencing specific MITRE ATT&CK techniques.",
        "A modern ransomware attack chain typically proceeds through several phases mapped to "
        "ATT&CK tactics. Initial Access (TA0001) often begins with phishing emails containing "
        "malicious attachments or links (T1566), or exploitation of public-facing applications "
        "(T1190) like VPN gateways or Exchange servers. After gaining a foothold, the attacker "
        "uses Command and Scripting Interpreters (T1059) - typically PowerShell - to download "
        "additional tools. Persistence (T1547) is established via registry Run keys or scheduled "
        "tasks. Credential Access follows with OS Credential Dumping (T1003) targeting LSASS to "
        "obtain domain admin credentials. Lateral Movement (T1021) via RDP, SMB admin shares, "
        "or PsExec spreads access across the network. Before encryption, the attacker performs "
        "Collection (T1560) and Exfiltration (T1041) of sensitive data for double extortion. "
        "Defense Evasion (T1562) disables security tools. Finally, Inhibit System Recovery (T1490) "
        "deletes backups, and Data Encrypted for Impact (T1486) deploys the ransomware payload "
        "across all compromised systems simultaneously.",
        "attack_chain", "qa", "technique_qa_v1",
        topics=["ransomware_ttps", "attack_chain"],
    ))

    examples.append(make_example(
        "How does the dwell time between initial access and ransomware deployment affect "
        "detection opportunities?",
        "The dwell time (time between initial compromise and ransomware deployment) in modern "
        "attacks typically ranges from hours to weeks, with the median around 5-7 days for "
        "human-operated ransomware. This window creates critical detection opportunities at each "
        "attack phase. During the first 24-48 hours, look for initial access indicators: unusual "
        "authentication patterns (T1078), phishing payload execution (T1566), or exploit artifacts "
        "(T1190). Days 2-5 typically involve reconnaissance using T1059 (PowerShell/cmd discovery "
        "commands), credential dumping (T1003), and lateral movement testing (T1021). The final "
        "24-48 hours before encryption see data staging and exfiltration (T1560, T1041), defense "
        "evasion (T1562), and recovery inhibition (T1490). Each phase produces detectable "
        "telemetry. The most reliable detection windows are the credential dumping phase (LSASS "
        "access is highly anomalous) and the pre-encryption defense evasion phase (VSS deletion "
        "and Defender disabling are high-confidence indicators). Organizations with 24/7 SOC "
        "coverage and automated correlation can often intervene during the lateral movement phase.",
        "attack_chain", "qa", "technique_qa_v1",
        topics=["ransomware_ttps", "detection", "incident_response"],
    ))

    examples.append(make_example(
        "What is double extortion ransomware and what MITRE ATT&CK techniques are specifically "
        "associated with the data theft component?",
        "Double extortion ransomware combines traditional file encryption (T1486) with data theft "
        "and threatened publication. The data theft component involves several ATT&CK techniques. "
        "Collection (TA0009) encompasses T1560 (Archive Collected Data) where attackers compress "
        "stolen data using tools like 7zip or WinRAR, T1213 (Data from Information Repositories) "
        "targeting SharePoint, file servers, and databases, and T1039 (Data from Network Shared "
        "Drive) harvesting sensitive files from mapped drives. Exfiltration (TA0010) uses T1041 "
        "(Exfiltration Over C2 Channel) or T1567 (Exfiltration Over Web Service) to transfer "
        "data to attacker infrastructure, often using legitimate cloud storage services like "
        "Mega.nz or rclone to cloud providers to blend with normal traffic. The threat of "
        "publication on leak sites creates pressure to pay even if backups enable recovery from "
        "encryption. Detection should focus on unusual outbound data volumes, execution of "
        "archiving tools by non-standard users, and large file transfers to cloud services "
        "outside normal business patterns.",
        "double_extortion", "qa", "technique_qa_v1",
        topics=["ransomware_ttps", "exfiltration"],
    ))

    examples.append(make_example(
        "How should an incident responder prioritize which MITRE ATT&CK techniques to "
        "investigate first when ransomware is confirmed on a network?",
        "When ransomware is confirmed, prioritize investigation of techniques that determine "
        "scope and contain the threat. First, investigate Lateral Movement techniques (T1021) "
        "to identify all compromised hosts - check Event ID 4624 logon events across the domain "
        "for the compromised accounts to map the blast radius. Second, examine Credential Access "
        "(T1003) to determine what credentials were stolen - if domain admin was compromised, "
        "assume all systems are at risk. Third, assess Defense Evasion (T1562) to understand "
        "which security controls were bypassed or disabled, as this affects your ability to "
        "trust telemetry from affected hosts. Fourth, review Exfiltration techniques (T1041, "
        "T1567) to determine if data was stolen, which changes the legal and regulatory response. "
        "Fifth, analyze the Execution (T1059) and Persistence (T1547) techniques to identify "
        "the malware and ensure all instances are contained. The encryption itself (T1486) is "
        "typically the last concern since it is already visible. Document findings in a timeline "
        "format to support both remediation and potential law enforcement engagement.",
        "incident_response", "qa", "technique_qa_v1",
        topics=["ransomware_ttps", "incident_response"],
    ))

    examples.append(make_example(
        "You are performing triage on a Windows domain where 50 workstations were encrypted "
        "overnight. The ransomware note references LockBit. Walk through your initial triage "
        "to determine the scope and initial access vector.",
        "Begin with containment: isolate affected network segments at the switch/VLAN level while "
        "preserving evidence. Do not shut down affected systems as memory may contain encryption "
        "keys and attacker tools.\n\n"
        "Identify the scope by querying Active Directory for recent authentication events. Pull "
        "Security Event ID 4624 from the domain controllers for the past 72 hours, filtering for "
        "Type 3 (network) and Type 10 (RDP) logons from the compromised accounts. Cross-reference "
        "with the list of encrypted systems to confirm the lateral movement path.\n\n"
        "Determine the initial access vector by identifying the first compromised system. Sort "
        "encrypted hosts by the earliest file modification timestamp - the first system encrypted "
        "is likely closest to the entry point, but the actual initial access system may not be "
        "encrypted at all. Check VPN logs, email gateway logs, and web proxy logs for the 72 "
        "hours preceding the first encryption event.\n\n"
        "Examine the domain controllers for evidence of credential harvesting. Look for Sysmon "
        "Event ID 10 targeting lsass.exe, DCSync indicators (Event ID 4662 with replication "
        "GUIDs from non-DC sources), and ntdsutil execution. LockBit operators typically use "
        "Mimikatz or Cobalt Strike for credential access.\n\n"
        "Check for data exfiltration indicators, as LockBit operates a double-extortion model "
        "with a leak site. Review firewall logs for large outbound transfers, especially to "
        "cloud storage services. Look for rclone.exe, megasync.exe, or unusual file archiving "
        "activity in process logs.\n\n"
        "Preserve forensic evidence from at least 3-5 representative systems: memory dumps, "
        "disk images, and event log exports. Include the suspected initial access system, a "
        "domain controller, and the first and last encrypted workstations to support a full "
        "timeline reconstruction.",
        "triage", "instruction", "technique_instruction_v1",
        topics=["ransomware_ttps", "incident_response", "triage"],
    ))

    examples.append(make_example(
        "What is the relationship between T1486 (Data Encrypted for Impact) and T1490 "
        "(Inhibit System Recovery) in ransomware operations?",
        "T1486 and T1490 are almost always paired in ransomware operations, with T1490 serving "
        "as a prerequisite for maximizing the impact of T1486. Inhibiting system recovery (T1490) "
        "is typically executed 1-5 minutes before encryption begins, ensuring that Volume Shadow "
        "Copies, Windows Recovery Environment, and backup catalogs are destroyed before the "
        "victim can use them to recover encrypted files. The operational sequence is usually: "
        "(1) T1562 disables security tools, (2) T1490 deletes recovery options, (3) T1486 "
        "encrypts files. This ordering is so consistent that detecting T1490 (VSS deletion) "
        "provides a critical early-warning window before encryption starts. A SIEM rule that "
        "triggers automated network isolation upon VSS deletion detection can prevent or limit "
        "the encryption phase. From a forensic perspective, the parent process that executes "
        "vssadmin or wmic for shadow deletion is typically the same process or script that "
        "subsequently launches the encryption routine, making it a key pivot point for "
        "identifying the ransomware binary.",
        "T1486_T1490", "qa", "technique_qa_v1",
        topics=["ransomware_ttps", "attack_chain"],
    ))

    examples.append(make_example(
        "How do ransomware groups abuse Group Policy Objects (GPO) for mass deployment, and "
        "how do you detect this?",
        "Ransomware groups with domain admin access abuse GPOs to deploy ransomware across an "
        "entire Active Directory domain simultaneously. The attacker creates or modifies a GPO "
        "that contains either a startup script pointing to the ransomware binary (placed on the "
        "SYSVOL share), an immediate scheduled task that executes on next Group Policy refresh, "
        "or software installation policies. LockBit and Ryuk have used this technique. The "
        "ransomware binary is typically placed in \\\\domain\\SYSVOL\\domain\\scripts or a "
        "similar accessible location. Detection focuses on several indicators: Event ID 5136 "
        "(Active Directory object modification) showing GPO changes, new files appearing in "
        "SYSVOL directories, Event ID 4698 (scheduled task creation) from Group Policy, and "
        "gpupdate.exe execution followed by suspicious child processes. Monitoring SYSVOL for "
        "new executable content and alerting on GPO modifications outside change management "
        "windows provides early detection. Preventive controls include requiring separate "
        "credentials for GPO management and implementing just-in-time access for domain admin "
        "privileges.",
        "gpo_deployment", "qa", "technique_qa_v1",
        topics=["ransomware_ttps", "lateral_movement", "detection"],
    ))

    examples.append(make_example(
        "Explain how entropy analysis can be used to detect ransomware encryption and what its "
        "limitations are.",
        "Entropy analysis measures the randomness (Shannon entropy) of file contents on a scale "
        "of 0 to 8 bits per byte. Unencrypted text files typically have entropy of 3.5-5.0, "
        "structured documents (Office, PDF) range from 4.0-7.0, while encrypted content approaches "
        "7.9-8.0 because encryption produces pseudorandom output indistinguishable from random "
        "data. By monitoring file write operations and computing entropy of written content, "
        "security tools can flag files that transition from low to high entropy as potential "
        "ransomware encryption. Limitations include: (1) compressed files (ZIP, JPEG, PNG) already "
        "have high entropy and produce false positives, (2) modern ransomware using intermittent "
        "encryption only encrypts portions of files, producing mixed entropy that may not trigger "
        "thresholds, (3) real-time entropy computation on all file writes has significant "
        "performance overhead, (4) the detection is reactive - by the time entropy change is "
        "detected, that file is already encrypted. Effective implementations focus on specific "
        "file types known to have low baseline entropy (text, documents) and use entropy as one "
        "signal in a multi-factor detection model rather than a standalone indicator.",
        "entropy_detection", "qa", "technique_qa_v1",
        topics=["ransomware_ttps", "detection", "forensics"],
    ))

    examples.append(make_example(
        "What are the key differences between automated (commodity) ransomware and "
        "human-operated ransomware from an ATT&CK technique perspective?",
        "Commodity ransomware and human-operated ransomware differ significantly in technique "
        "breadth and sophistication. Commodity ransomware (e.g., early WannaCry, Dharma) uses "
        "a narrow set of techniques: automated Initial Access via exploits (T1190) or email "
        "attachments (T1566), immediate Execution (T1059), and rapid Encryption (T1486) with "
        "basic Recovery Inhibition (T1490). The entire chain is automated and executes within "
        "minutes. Human-operated ransomware (LockBit, Conti, BlackCat) employs the full ATT&CK "
        "matrix over days to weeks. Operators perform hands-on-keyboard Discovery (T1018, T1087) "
        "to map the environment, Credential Access (T1003) for privilege escalation, extensive "
        "Lateral Movement (T1021) to maximize reach, Collection and Exfiltration (T1560, T1041) "
        "for double extortion, and methodical Defense Evasion (T1562). Human operators adapt "
        "tactics based on what they find - they may pivot from one credential dumping method to "
        "another if the first is blocked. This adaptability makes them harder to detect with "
        "static signatures and requires behavioral detection across multiple technique indicators.",
        "commodity_vs_human", "qa", "technique_qa_v1",
        topics=["ransomware_ttps", "attack_chain"],
    ))

    examples.append(make_example(
        "You find evidence of both Cobalt Strike and ransomware on a compromised Windows host. "
        "How do you differentiate the two toolsets and trace the attack timeline?",
        "Cobalt Strike is a post-exploitation framework typically used for access maintenance and "
        "lateral movement before ransomware deployment, so distinguishing the two tools helps "
        "reconstruct the attack timeline.\n\n"
        "Identify Cobalt Strike artifacts first: look for its beacon in memory (reflective DLL "
        "injection), default named pipes (\\\\pipe\\msagent_*, \\\\pipe\\postex_*), and service "
        "creation patterns. Beacon configuration can be extracted from memory using tools like "
        "BeaconEye. Check for HTTP/HTTPS beaconing with regular intervals (Sysmon Event ID 3) "
        "and malleable C2 profile artifacts in proxy logs.\n\n"
        "Separate the ransomware artifacts: encrypted files with modified extensions, ransom "
        "notes, and the encryption binary. Check Prefetch and process creation logs for the "
        "ransomware executable, which is distinct from the Cobalt Strike beacon.\n\n"
        "Build the timeline by correlating timestamps. Typically: (1) Cobalt Strike beacon "
        "establishes C2 channel (earliest network connection to C2 IP), (2) discovery and "
        "credential dumping via Cobalt Strike modules (Sysmon Event ID 10 for LSASS access), "
        "(3) lateral movement using stolen credentials (Event ID 4624 type 3/10), (4) defense "
        "evasion and recovery inhibition (VSS deletion, Defender disable), (5) ransomware "
        "binary deployed and executed. The gap between steps 3 and 5 is where data exfiltration "
        "typically occurs.\n\n"
        "Check if Cobalt Strike was the delivery mechanism for the ransomware by examining "
        "parent-child process relationships - the beacon process may have spawned the ransomware "
        "executable or a staging script.",
        "cobalt_strike_ransomware", "instruction", "technique_instruction_v1",
        topics=["ransomware_ttps", "incident_response", "forensics"],
    ))

    return examples


# ---------------------------------------------------------------------------
# Main generation pipeline
# ---------------------------------------------------------------------------

def select_techniques_for_generation(techniques):
    """Select and prioritize techniques for example generation.

    Target: ~400 examples total.  Budget breakdown (approximate):
      - 8 priority parent techniques: 5 QA + 1 instruction + extras ≈ 70
      - Priority subtechniques (capped at 3 per parent): 2 QA ≈ 45
      - Non-priority parent techniques in key tactics (~30): 3 QA + 1 instruction ≈ 120
      - Remaining Windows-relevant parent techniques (capped): 2 QA ≈ 100
      - Cross-technique examples: 10
      Total ≈ 345-415

    Returns a list of (technique, num_qa, generate_instruction) tuples.
    """
    selected = []

    # Index by ID
    by_id = {t['id']: t for t in techniques}

    # Priority parent techniques: 5 QA + 1 instruction each (plus extras via generate_priority_extra_qa)
    for pid in PRIORITY_TECHNIQUE_IDS:
        if pid in by_id:
            selected.append((by_id[pid], 5, True))

    # Priority subtechniques: 2 QA, no instruction. Cap at 3 per parent to control budget.
    from collections import defaultdict as _dd
    sub_counts = _dd(int)
    for t in sorted(techniques, key=lambda x: x['id']):
        if t['is_subtechnique']:
            base = t['id'].split('.')[0]
            if base in PRIORITY_TECHNIQUE_IDS and sub_counts[base] < 3:
                # Prefer Windows-relevant subtechniques
                if 'Windows' in t['platforms'] or not t['platforms']:
                    selected.append((t, 2, False))
                    sub_counts[base] += 1

    # Get the IDs we've already selected
    selected_ids = {s[0]['id'] for s in selected}

    # Non-priority parent techniques in key tactics: 3 QA + 1 instruction
    # Only include if Windows is in platforms
    key_tactics = {'impact', 'initial-access', 'execution', 'persistence', 'credential-access'}
    for t in techniques:
        if t['id'] not in selected_ids and not t['is_subtechnique']:
            if set(t['tactics']) & key_tactics:
                if 'Windows' in t['platforms'] or not t['platforms']:
                    selected.append((t, 3, True))
                    selected_ids.add(t['id'])

    # Remaining Windows-relevant parent techniques: 2 QA, no instruction
    # Cap at 30 to control total budget (~400 target)
    remaining_count = 0
    remaining_cap = 30
    for t in techniques:
        if remaining_count >= remaining_cap:
            break
        if t['id'] not in selected_ids and not t['is_subtechnique']:
            if 'Windows' in t['platforms'] or not t['platforms']:
                selected.append((t, 2, False))
                selected_ids.add(t['id'])
                remaining_count += 1

    return selected


def main():
    print("Loading STIX data...")
    objects_by_type, objects_by_id = load_stix_data()

    print("Extracting techniques...")
    techniques = extract_techniques(objects_by_type, objects_by_id)
    print(f"Found {len(techniques)} relevant techniques")

    print("Selecting techniques for generation...")
    selected = select_techniques_for_generation(techniques)

    print(f"Selected {len(selected)} techniques for example generation")

    # Estimate total
    total_estimate = sum(nqa + (1 if gen_inst else 0) for _, nqa, gen_inst in selected)
    print(f"Estimated examples from techniques: {total_estimate}")

    examples = []
    technique_counts = Counter()

    for technique, num_qa, gen_instruction in selected:
        tid = technique['id']
        batch = []

        # Always generate "what is" QA
        batch.append(generate_qa_what(technique))

        # Generate ransomware usage if enough QA slots
        if num_qa >= 2:
            batch.append(generate_qa_ransomware_usage(technique))

        # Generate detection QA
        if num_qa >= 3:
            batch.append(generate_qa_detection(technique))

        # Generate artifacts QA
        if num_qa >= 4:
            batch.append(generate_qa_artifacts(technique))

        # Generate mitigations QA
        if num_qa >= 5:
            batch.append(generate_qa_mitigations(technique))

        # Generate instruction example
        if gen_instruction:
            batch.append(generate_instruction_example(technique))

        # Extra QA for priority techniques
        base_id = tid.split('.')[0]
        if tid in PRIORITY_TECHNIQUE_IDS:
            batch.extend(generate_priority_extra_qa(technique))

        examples.extend(batch)
        technique_counts[tid] = len(batch)

    # Add cross-technique examples
    cross_examples = generate_cross_technique_examples()
    examples.extend(cross_examples)
    technique_counts['cross_technique'] = len(cross_examples)

    # Write output
    print(f"\nWriting {len(examples)} examples to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(examples)} total examples generated")
    print(f"{'='*60}")

    # Group by parent technique
    parent_counts = Counter()
    for tid, count in technique_counts.items():
        if tid == 'cross_technique':
            parent_counts[tid] = count
        else:
            parent = tid.split('.')[0]
            parent_counts[parent] += count

    print(f"\nExamples per parent technique (top 20):")
    for tid, count in parent_counts.most_common(20):
        print(f"  {tid}: {count}")

    print(f"\nRemaining techniques: {len(parent_counts) - 20} "
          f"(each with {sum(c for _, c in parent_counts.most_common()[20:])} examples total)")

    # By example type
    type_counts = Counter()
    for ex in examples:
        type_counts[ex['metadata']['example_type']] += 1
    print(f"\nBy example type:")
    for t, c in type_counts.most_common():
        print(f"  {t}: {c}")

    # Validate
    print(f"\nValidation:")
    for ex in examples:
        msgs = ex['messages']
        assert len(msgs) == 3, f"Expected 3 messages, got {len(msgs)}"
        assert msgs[0]['role'] == 'system'
        assert msgs[1]['role'] == 'user'
        assert msgs[2]['role'] == 'assistant'
        assert len(msgs[1]['content']) > 10, f"User message too short: {msgs[1]['content']}"
        assert len(msgs[2]['content']) > 50, f"Assistant message too short for {ex['metadata']['source_section']}"
        assert ex['metadata']['source'] == 'mitre_attack_stix'
    print("  All examples passed validation")


if __name__ == '__main__':
    main()
