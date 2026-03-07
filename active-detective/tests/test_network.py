"""Tests for ConnectionTable."""

from datetime import datetime

import pytest

from simulator.network import ConnectionTable


class TestConnectionTable:
    @pytest.fixture
    def table(self):
        return ConnectionTable()

    def test_open_connection(self, table):
        now = datetime(2025, 1, 1)
        conn = table.open_connection(100, 443, "10.0.0.1", 443, "tcp",
                                     "established", now)
        assert conn.conn_id == 1
        assert conn.pid == 100
        assert conn.state == "established"
        assert len(table) == 1

    def test_close_connection(self, table):
        now = datetime(2025, 1, 1)
        conn = table.open_connection(100, 443, "10.0.0.1", 443, now=now)
        result = table.close_connection(conn.conn_id)
        assert result is not None
        assert result.state == "closed"

    def test_close_missing(self, table):
        assert table.close_connection(999) is None

    def test_transfer_data(self, table):
        now = datetime(2025, 1, 1)
        conn = table.open_connection(100, 443, "10.0.0.1", 443, now=now)
        table.transfer_data(conn.conn_id, bytes_sent=1024, bytes_received=512)
        updated = table.get_connection(conn.conn_id)
        assert updated.bytes_sent == 1024
        assert updated.bytes_received == 512

    def test_transfer_accumulates(self, table):
        now = datetime(2025, 1, 1)
        conn = table.open_connection(100, 443, "10.0.0.1", 443, now=now)
        table.transfer_data(conn.conn_id, bytes_sent=100)
        table.transfer_data(conn.conn_id, bytes_sent=200)
        assert table.get_connection(conn.conn_id).bytes_sent == 300

    def test_transfer_missing(self, table):
        assert table.transfer_data(999) is None

    def test_get_connection(self, table):
        now = datetime(2025, 1, 1)
        conn = table.open_connection(100, 443, "10.0.0.1", 443, now=now)
        assert table.get_connection(conn.conn_id) is conn
        assert table.get_connection(999) is None

    def test_all_connections(self, table):
        now = datetime(2025, 1, 1)
        table.open_connection(100, 443, "10.0.0.1", 443, now=now)
        table.open_connection(200, 80, "10.0.0.2", 80, now=now)
        assert len(table.all_connections()) == 2

    def test_list_by_pid(self, table):
        now = datetime(2025, 1, 1)
        table.open_connection(100, 443, "10.0.0.1", 443, now=now)
        table.open_connection(100, 80, "10.0.0.2", 80, now=now)
        table.open_connection(200, 22, "10.0.0.3", 22, now=now)
        assert len(table.list_by_pid(100)) == 2
        assert len(table.list_by_pid(200)) == 1
        assert len(table.list_by_pid(999)) == 0

    def test_list_by_state(self, table):
        now = datetime(2025, 1, 1)
        c1 = table.open_connection(100, 443, "10.0.0.1", 443,
                                   state="established", now=now)
        c2 = table.open_connection(200, 80, "10.0.0.2", 80,
                                   state="listening", now=now)
        assert len(table.list_by_state("established")) == 1
        assert len(table.list_by_state("listening")) == 1

    def test_list_by_remote(self, table):
        now = datetime(2025, 1, 1)
        table.open_connection(100, 443, "10.0.0.1", 443, now=now)
        table.open_connection(200, 443, "10.0.0.1", 443, now=now)
        table.open_connection(300, 80, "10.0.0.2", 80, now=now)
        assert len(table.list_by_remote("10.0.0.1")) == 2

    def test_seed_connections(self, table):
        now = datetime(2025, 1, 1)
        table.seed_connections([100, 200], now)
        assert len(table) == 3
        # All should be established
        for conn in table.all_connections():
            assert conn.state == "established"

    def test_auto_incrementing_ids(self, table):
        now = datetime(2025, 1, 1)
        c1 = table.open_connection(100, 443, "10.0.0.1", 443, now=now)
        c2 = table.open_connection(200, 80, "10.0.0.2", 80, now=now)
        assert c2.conn_id == c1.conn_id + 1
