import json
from types import SimpleNamespace

from agent.local_mlx import (
    LocalMLXClient,
    _recover_terminal_tool_calls,
    _recover_xml_tool_calls,
)


def test_local_mlx_tool_call_parsing(monkeypatch):
    monkeypatch.setenv("LOCAL_MLX_SHARED_ENDPOINT", "0")

    def fake_chat_completion(messages, **kwargs):
        message = SimpleNamespace(
            role="assistant",
            content=None,
            tool_calls=[
                SimpleNamespace(
                    id="mlx_call_1",
                    type="function",
                    function=SimpleNamespace(
                        name="search_files",
                        arguments=json.dumps({"pattern": "foo"}),
                    ),
                )
            ],
        )
        choice = SimpleNamespace(index=0, message=message, finish_reason="tool_calls")
        return SimpleNamespace(choices=[choice], model="mlx-test", usage=SimpleNamespace())

    fake_service = SimpleNamespace(chat_completion=fake_chat_completion)
    monkeypatch.setattr("agent.local_mlx.LocalMLXService.instance", lambda: fake_service)

    client = LocalMLXClient(model_name="mlx-test")
    response = client.chat.completions.create(
        model="mlx-test",
        messages=[{"role": "user", "content": "find foo"}],
        tools=[{"function": {"name": "search_files", "parameters": {"type": "object"}}}],
    )

    assert response.choices[0].finish_reason == "tool_calls"
    assert response.choices[0].message.tool_calls[0].function.name == "search_files"


def test_local_mlx_uses_shared_endpoint(monkeypatch):
    monkeypatch.setenv("LOCAL_MLX_SHARED_ENDPOINT", "1")
    monkeypatch.setattr("agent.local_mlx._ensure_shared_local_mlx_server", lambda: True)
    monkeypatch.setattr(
        "agent.local_mlx._mlx_http_request",
        lambda method, path, **kwargs: {
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "mlx_call_http",
                                "type": "function",
                                "function": {
                                    "name": "search_files",
                                    "arguments": json.dumps({"pattern": "bar"}),
                                },
                            }
                        ],
                    },
                }
            ],
            "model": "mlx-http",
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        },
    )

    client = LocalMLXClient(model_name="mlx-test")
    response = client.chat.completions.create(
        model="mlx-test",
        messages=[{"role": "user", "content": "find bar"}],
        tools=[{"function": {"name": "search_files", "parameters": {"type": "object"}}}],
    )

    assert response.model == "mlx-http"
    assert response.choices[0].finish_reason == "tool_calls"
    assert response.choices[0].message.tool_calls[0].function.name == "search_files"


def test_local_mlx_recovers_terminal_call_from_bash_block(monkeypatch):
    recovered = _recover_terminal_tool_calls(
        "Let me check it.\n```bash\nls -la ~/Documents/Work/Matters/\n```",
        [{"function": {"name": "terminal", "parameters": {"type": "object"}}}],
    )

    assert recovered is not None
    assert recovered[0].function.name == "terminal"
    assert "ls -la ~/Documents/Work/Matters/" in recovered[0].function.arguments


def test_local_mlx_recovers_tool_call_xml_wrapper():
    recovered = _recover_xml_tool_calls(
        '<tool_call>{"name":"read_file","arguments":{"path":"/tmp/subagent_results.json","offset":1,"limit":50}}</tool_call>',
        [{"function": {"name": "read_file", "parameters": {"type": "object"}}}],
    )

    assert recovered is not None
    assert recovered[0].function.name == "read_file"
    assert json.loads(recovered[0].function.arguments) == {
        "path": "/tmp/subagent_results.json",
        "offset": 1,
        "limit": 50,
    }


def test_local_mlx_recovers_direct_xml_tool_tag():
    recovered = _recover_xml_tool_calls(
        "<read_file><path>/tmp/subagent_results.json</path><offset>1</offset><limit>50</limit></read_file>",
        [{"function": {"name": "read_file", "parameters": {"type": "object"}}}],
    )

    assert recovered is not None
    assert recovered[0].function.name == "read_file"
    assert json.loads(recovered[0].function.arguments) == {
        "path": "/tmp/subagent_results.json",
        "offset": 1,
        "limit": 50,
    }


def test_local_mlx_recovers_malformed_direct_xml_tool_tag():
    recovered = _recover_xml_tool_calls(
        "<web_search><query>Apple iMessage custom subaddress registration</query></query>",
        [{"function": {"name": "web_search", "parameters": {"type": "object"}}}],
    )

    assert recovered is not None
    assert recovered[0].function.name == "web_search"
    assert json.loads(recovered[0].function.arguments) == {
        "query": "Apple iMessage custom subaddress registration",
    }
