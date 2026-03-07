"""HostState: composition of all mutable sub-registries.

Single object passed to generators and tools, replacing the previous
(registry, ptable, rng, now) positional argument pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from simulator.clock import SimClock
from simulator.event_log import EventLog
from simulator.network import ConnectionTable
from simulator.registry import FileRegistry, ProcessTable
from simulator.registry_hive import RegistryHive


@dataclass
class HostState:
    """Composed mutable host state for the simulated environment.

    Generators mutate this; tools query it; the environment owns it.
    """

    files: FileRegistry
    processes: ProcessTable
    connections: ConnectionTable
    registry: RegistryHive
    event_log: EventLog
    clock: SimClock

    @classmethod
    def create(
        cls,
        rng: np.random.RandomState,
        now: datetime | None = None,
    ) -> HostState:
        """Create a fully seeded host state.

        Seeds all 6 sub-registries with consistent initial state.
        """
        now = now or datetime(2025, 6, 15, 10, 0, 0)

        files = FileRegistry()
        files.seed_filesystem(rng, now)

        processes = ProcessTable()
        processes.seed_processes(now)

        connections = ConnectionTable()
        connections.seed_connections(processes.all_pids(), now)

        registry = RegistryHive()
        registry.seed_registry(now)

        event_log = EventLog()
        event_log.seed_events(now)

        clock = SimClock(now)

        return cls(
            files=files,
            processes=processes,
            connections=connections,
            registry=registry,
            event_log=event_log,
            clock=clock,
        )
