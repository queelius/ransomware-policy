"""ConnectionTable: mutable network connection state.

Tracks active connections the agent can inspect via list_connections
and inspect_connection tools. Attack generators create C2 connections;
benign generators create background traffic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from simulator.models import ConnectionRecord


class ConnectionTable:
    """Mutable connection table tracking active network connections.

    Follows the same dict-based pattern as FileRegistry.
    """

    def __init__(self) -> None:
        self._connections: dict[int, ConnectionRecord] = {}
        self._next_id: int = 1

    def __len__(self) -> int:
        return len(self._connections)

    def open_connection(
        self,
        pid: int,
        local_port: int,
        remote_address: str,
        remote_port: int,
        protocol: str = "tcp",
        state: Literal["established", "listening", "closed", "syn_sent"] = "established",
        now: datetime | None = None,
    ) -> ConnectionRecord:
        """Create a new connection."""
        conn_id = self._next_id
        self._next_id += 1

        record = ConnectionRecord(
            conn_id=conn_id,
            pid=pid,
            local_port=local_port,
            remote_address=remote_address,
            remote_port=remote_port,
            protocol=protocol,
            state=state,
            opened_at=now or datetime(2025, 1, 1),
        )
        self._connections[conn_id] = record
        return record

    def close_connection(self, conn_id: int) -> ConnectionRecord | None:
        """Close a connection (marks as closed, keeps in table)."""
        record = self._connections.get(conn_id)
        if record is None:
            return None
        record.state = "closed"
        return record

    def transfer_data(
        self,
        conn_id: int,
        bytes_sent: int = 0,
        bytes_received: int = 0,
    ) -> ConnectionRecord | None:
        """Record data transfer on a connection."""
        record = self._connections.get(conn_id)
        if record is None:
            return None
        record.bytes_sent += bytes_sent
        record.bytes_received += bytes_received
        return record

    def get_connection(self, conn_id: int) -> ConnectionRecord | None:
        return self._connections.get(conn_id)

    def all_connections(self) -> list[ConnectionRecord]:
        return list(self._connections.values())

    def list_by_pid(self, pid: int) -> list[ConnectionRecord]:
        return [c for c in self._connections.values() if c.pid == pid]

    def list_by_state(
        self, state: str,
    ) -> list[ConnectionRecord]:
        return [c for c in self._connections.values() if c.state == state]

    def list_by_remote(self, remote_address: str) -> list[ConnectionRecord]:
        return [c for c in self._connections.values()
                if c.remote_address == remote_address]

    def seed_connections(
        self,
        process_pids: list[int],
        now: datetime,
    ) -> None:
        """Populate with typical background connections."""
        backgrounds = [
            (process_pids[0] if process_pids else 4, 443, "13.107.42.14", 443, "tcp"),
            (process_pids[0] if process_pids else 4, 80, "204.79.197.200", 80, "tcp"),
            (process_pids[0] if process_pids else 4, 53, "8.8.8.8", 53, "udp"),
        ]
        for pid, lport, raddr, rport, proto in backgrounds:
            self.open_connection(pid, lport, raddr, rport, proto, "established", now)
