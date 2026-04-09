"""Tests for inter-agent communication."""

from cosmos_agi.agents.communication import Blackboard, Message, MessageBus


class TestBlackboard:
    def test_write_and_read(self):
        bb = Blackboard()
        bb.write("facts", "sky_color", "blue", author="researcher")
        assert bb.read("facts", "sky_color") == "blue"

    def test_read_missing(self):
        bb = Blackboard()
        assert bb.read("facts", "nonexistent") is None
        assert bb.read("facts", "nonexistent", "default") == "default"

    def test_read_namespace(self):
        bb = Blackboard()
        bb.write("facts", "a", 1)
        bb.write("facts", "b", 2)
        bb.write("plans", "x", 10)
        assert bb.read_namespace("facts") == {"a": 1, "b": 2}

    def test_list_namespaces(self):
        bb = Blackboard()
        bb.write("facts", "a", 1)
        bb.write("plans", "b", 2)
        ns = bb.list_namespaces()
        assert "facts" in ns
        assert "plans" in ns

    def test_search(self):
        bb = Blackboard()
        bb.write("facts", "weather", "sunny and warm")
        bb.write("facts", "temperature", "25 degrees")
        results = bb.search("warm")
        assert len(results) == 1
        assert results[0][1] == "weather"

    def test_to_text(self):
        bb = Blackboard()
        bb.write("facts", "sky", "blue")
        text = bb.to_text()
        assert "facts" in text
        assert "sky" in text
        assert "blue" in text

    def test_clear(self):
        bb = Blackboard()
        bb.write("facts", "a", 1)
        bb.clear()
        assert bb.read("facts", "a") is None
        assert bb.list_namespaces() == []

    def test_overwrite(self):
        bb = Blackboard()
        bb.write("facts", "count", 1)
        bb.write("facts", "count", 2)
        assert bb.read("facts", "count") == 2


class TestMessageBus:
    def test_send_and_receive(self):
        bus = MessageBus()
        msg = Message(sender="planner", recipient="executor", content="Do task 1")
        bus.send(msg)

        received = bus.receive("executor")
        assert len(received) == 1
        assert received[0].content == "Do task 1"

    def test_receive_drains_queue(self):
        bus = MessageBus()
        bus.send(Message(sender="a", recipient="b", content="msg1"))
        bus.receive("b")
        assert bus.receive("b") == []

    def test_broadcast(self):
        bus = MessageBus()
        # Subscribe agents first so the bus knows about them
        bus.subscribe("agent_a", lambda m: None)
        bus.subscribe("agent_b", lambda m: None)
        bus.send(Message(sender="leader", recipient="*", content="attention all"))

        # All subscribed agents except sender should get it
        assert len(bus.receive("agent_a")) == 1
        assert len(bus.receive("agent_b")) == 1
        assert len(bus.receive("leader")) == 0  # sender excluded

    def test_priority_ordering(self):
        bus = MessageBus()
        bus.send(Message(sender="a", recipient="b", content="low", priority=0))
        bus.send(Message(sender="a", recipient="b", content="high", priority=10))
        bus.send(Message(sender="a", recipient="b", content="medium", priority=5))

        received = bus.receive("b")
        assert received[0].content == "high"
        assert received[1].content == "medium"
        assert received[2].content == "low"

    def test_filter_by_type(self):
        bus = MessageBus()
        bus.send(Message(sender="a", recipient="b", content="info1", msg_type="info"))
        bus.send(Message(sender="a", recipient="b", content="req1", msg_type="request"))

        requests = bus.receive("b", msg_type="request")
        assert len(requests) == 1
        assert requests[0].content == "req1"

    def test_peek(self):
        bus = MessageBus()
        assert bus.peek("agent_x") == 0
        bus.send(Message(sender="a", recipient="agent_x", content="hi"))
        assert bus.peek("agent_x") == 1

    def test_subscribe_callback(self):
        bus = MessageBus()
        received = []
        bus.subscribe("listener", lambda msg: received.append(msg))
        bus.send(Message(sender="sender", recipient="listener", content="hello"))
        assert len(received) == 1
        assert received[0].content == "hello"

    def test_broadcast_log(self):
        bus = MessageBus()
        bus.send(Message(sender="a", recipient="*", content="broadcast1"))
        bus.send(Message(sender="b", recipient="*", content="broadcast2"))
        log = bus.get_broadcast_log()
        assert len(log) == 2


class TestMessage:
    def test_defaults(self):
        msg = Message(sender="a", recipient="b", content="hello")
        assert msg.msg_type == "info"
        assert msg.priority == 0
        assert msg.in_reply_to is None
        assert msg.timestamp  # should have auto-set

    def test_reply(self):
        msg = Message(
            sender="b",
            recipient="a",
            content="reply",
            in_reply_to="msg_123",
            msg_type="response",
        )
        assert msg.in_reply_to == "msg_123"
