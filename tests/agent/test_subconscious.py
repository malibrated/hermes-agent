from agent.subconscious import SubconsciousSelector


def test_subconscious_selector_prefers_blockers_and_hypotheses_without_llm(monkeypatch):
    monkeypatch.setattr("agent.subconscious.get_text_auxiliary_client", lambda task="": (None, None))

    packet = [
        {"node_id": "a1", "type": "artifact", "score": 0.91},
        {"node_id": "b1", "type": "blocker", "score": 0.62},
        {"node_id": "h1", "type": "hypothesis", "score": 0.58},
        {"node_id": "g1", "type": "goal", "score": 0.95},
    ]

    out = SubconsciousSelector().choose_expansions(
        query="why did the vault lookup fail?",
        packet=packet,
        max_expand=2,
    )

    assert out == ["b1", "h1"]
