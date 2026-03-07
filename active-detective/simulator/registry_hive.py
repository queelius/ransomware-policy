"""RegistryHive: mutable Windows registry state.

Tracks registry keys/values the agent can inspect via query_registry.
Attack generators modify run keys, disable Defender, etc.
"""

from __future__ import annotations

from datetime import datetime

from simulator.models import RegistryKey, RegistryValue


class RegistryHive:
    """Mutable registry hive tracking Windows registry state.

    Dict-based, keyed by full registry path.
    """

    def __init__(self) -> None:
        self._keys: dict[str, RegistryKey] = {}

    def __len__(self) -> int:
        return len(self._keys)

    def set_value(
        self,
        key_path: str,
        value_name: str,
        data: str,
        value_type: str = "REG_SZ",
        now: datetime | None = None,
    ) -> RegistryKey:
        """Set a value in a registry key (creates key if needed)."""
        now = now or datetime(2025, 1, 1)
        if key_path not in self._keys:
            self._keys[key_path] = RegistryKey(path=key_path, modified_at=now)

        key = self._keys[key_path]
        key.values[value_name] = RegistryValue(
            name=value_name, data=data, value_type=value_type,
        )
        key.modified_at = now
        return key

    def get_value(
        self, key_path: str, value_name: str,
    ) -> RegistryValue | None:
        """Get a specific value from a key."""
        key = self._keys.get(key_path)
        if key is None:
            return None
        return key.values.get(value_name)

    def delete_key(self, key_path: str) -> RegistryKey | None:
        """Delete an entire key and its values."""
        return self._keys.pop(key_path, None)

    def enumerate_keys(self, prefix: str) -> list[str]:
        """List all key paths under a prefix."""
        if not prefix.endswith("\\"):
            prefix = prefix + "\\"
        return sorted(k for k in self._keys if k.startswith(prefix))

    def get_key(self, key_path: str) -> RegistryKey | None:
        """Get a key with all its values."""
        return self._keys.get(key_path)

    # ── Convenience methods for common ATT&CK patterns ────────────

    def add_run_key(
        self, name: str, command: str, now: datetime,
    ) -> RegistryKey:
        """Add a persistence entry to Run key (T1547.001)."""
        return self.set_value(
            r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Run",
            name, command, "REG_SZ", now,
        )

    def disable_defender(self, now: datetime) -> RegistryKey:
        """Disable Windows Defender real-time monitoring (T1562.001)."""
        return self.set_value(
            r"HKLM\SOFTWARE\Policies\Microsoft\Windows Defender\Real-Time Protection",
            "DisableRealtimeMonitoring", "1", "REG_DWORD", now,
        )

    def disable_vss(self, now: datetime) -> RegistryKey:
        """Disable Volume Shadow Copy service (T1490)."""
        return self.set_value(
            r"HKLM\SYSTEM\CurrentControlSet\Services\VSS",
            "Start", "4", "REG_DWORD", now,
        )

    def seed_registry(self, now: datetime) -> None:
        """Populate with typical Windows registry state."""
        # Standard autostart entries
        self.set_value(
            r"HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Run",
            "SecurityHealth", r"C:\Windows\System32\SecurityHealthSystray.exe",
            "REG_SZ", now,
        )
        # Windows Defender enabled (baseline)
        self.set_value(
            r"HKLM\SOFTWARE\Policies\Microsoft\Windows Defender\Real-Time Protection",
            "DisableRealtimeMonitoring", "0", "REG_DWORD", now,
        )
        # VSS enabled
        self.set_value(
            r"HKLM\SYSTEM\CurrentControlSet\Services\VSS",
            "Start", "3", "REG_DWORD", now,
        )
        # PowerShell execution policy
        self.set_value(
            r"HKLM\SOFTWARE\Microsoft\PowerShell\1\ShellIds\Microsoft.PowerShell",
            "ExecutionPolicy", "Restricted", "REG_SZ", now,
        )
        # Windows Firewall
        self.set_value(
            r"HKLM\SYSTEM\CurrentControlSet\Services\SharedAccess\Parameters\FirewallPolicy\StandardProfile",
            "EnableFirewall", "1", "REG_DWORD", now,
        )
