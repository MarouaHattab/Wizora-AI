import inspect


def test_chroma_can_call_the_locked_telemetry_client():
    import posthog

    inspect.signature(posthog.capture).bind("local-test-user", "ClientStartEvent", {})
