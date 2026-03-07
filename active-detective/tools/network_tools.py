"""Network investigation tools: list_connections, inspect_connection."""

from __future__ import annotations

from simulator.network import ConnectionTable


def list_connections(
    connections: ConnectionTable,
    filter_state: str | None = None,
) -> dict:
    """List active network connections with optional state filter. Cost: -0.03.

    Returns connection summaries: conn_id, pid, remote, state, bytes.
    """
    if filter_state:
        conns = connections.list_by_state(filter_state)
    else:
        conns = connections.all_connections()

    return {
        "connections": [
            {
                "conn_id": c.conn_id,
                "pid": c.pid,
                "remote": f"{c.remote_address}:{c.remote_port}",
                "protocol": c.protocol,
                "state": c.state,
                "bytes_sent": c.bytes_sent,
                "bytes_received": c.bytes_received,
            }
            for c in conns
        ]
    }


def inspect_connection(
    connections: ConnectionTable,
    conn_id: int,
) -> dict:
    """Inspect a specific connection in detail. Cost: -0.03.

    Returns full connection details including local port and timestamps.
    """
    record = connections.get_connection(conn_id)
    if record is None:
        return {"error": f"Connection not found: conn_id={conn_id}"}

    return {
        "conn_id": record.conn_id,
        "pid": record.pid,
        "local_port": record.local_port,
        "remote_address": record.remote_address,
        "remote_port": record.remote_port,
        "protocol": record.protocol,
        "state": record.state,
        "bytes_sent": record.bytes_sent,
        "bytes_received": record.bytes_received,
        "opened_at": record.opened_at.isoformat(),
    }
