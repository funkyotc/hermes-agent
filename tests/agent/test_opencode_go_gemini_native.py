"""Gemini on OpenCode Go speaks Google's wire — on the Go relay, without touching its siblings.

The Zen-scoped sibling change (#138631) deliberately leaves OpenCode Go on the OpenAI wire, so Go
Gemini requests still reach ``/chat/completions`` and come back HTTP 400 ``ModelProtocolUnsupported``
(issue #83753). Go serves the same model only from the documented ``/inference/go`` root: the legacy
``/zen/go`` root answers 404 for ``:generateContent`` / ``:streamGenerateContent``.

Offline: every test drives the real routing helpers and clients against an httpx MockTransport.
"""

import json

import httpx
import pytest

from agent.agent_runtime_helpers import create_openai_client
from agent.gemini_native_adapter import GeminiNativeClient, normalize_gemini_base_url
from agent.gemini_routing import (
    OPENCODE_GO_GEMINI_BASE_URL,
    is_native_gemini_request,
    is_native_gemini_route,
    is_opencode_go_gemini_route,
)
from agent.opencode_affinity import merge_session_affinity_headers
from agent.transports.chat_completions import ChatCompletionsTransport
from providers import get_provider_profile

GO_LEGACY = "https://opencode.ai/zen/go/v1"
GO_DOCUMENTED = "https://opencode.ai/inference/go/v1"
MODEL = "gemini-4-argon-contributor"
SESSION = "go-gemini-session"


@pytest.mark.parametrize(
    "provider,model,base_url,native",
    [
        ("opencode-go", MODEL, GO_LEGACY, True),
        ("go", MODEL, GO_LEGACY, True),
        ("opencode-go-sub", MODEL, GO_LEGACY, True),
        ("opencode-go", MODEL, GO_DOCUMENTED, True),
        ("opencode-go", MODEL, "https://opencode.ai/zen/go", True),
        # Sibling families on the same relay keep the wire they already had.
        ("opencode-go", "deepseek-v4.1-flash", GO_LEGACY, False),
        ("opencode-go", "glm-5.3", GO_LEGACY, False),
        ("opencode-go", "claude-haiku-5-5", GO_LEGACY, False),
        # Zen roots and unknown hosts are not Go's business, in either direction.
        ("opencode-go", MODEL, "https://opencode.ai/zen/v1", False),
        ("opencode-zen", MODEL, "https://opencode.ai/zen/v1", False),
        ("opencode-go", MODEL, "https://example.invalid/v1", False),
    ],
)
def test_go_gemini_route_matrix(provider, model, base_url, native):
    assert is_opencode_go_gemini_route(provider, model, base_url) is native


@pytest.mark.parametrize("provider,model,base_url", [
    ("opencode-go", MODEL, GO_LEGACY),
    ("go", MODEL, GO_LEGACY),
    ("opencode-go", MODEL, GO_DOCUMENTED),
    # The Zen half of the same fix: one shared answer, whichever relay carries the model.
    ("opencode-zen", MODEL, "https://opencode.ai/zen/v1"),
])
def test_both_relays_answer_the_native_question(provider, model, base_url):
    assert is_native_gemini_route(provider, model, base_url) is True
    assert is_native_gemini_request(provider, model, base_url) is True


@pytest.mark.parametrize(
    "raw,expected",
    [
        (GO_LEGACY, GO_DOCUMENTED),
        ("https://opencode.ai/zen/go", GO_DOCUMENTED),
        ("https://opencode.ai/inference/go", GO_DOCUMENTED),
        ("https://opencode.ai/inference/go/v1", GO_DOCUMENTED),
        # Everything else keeps its existing normalization.
        ("https://opencode.ai/zen/v1", "https://opencode.ai/zen/v1"),
        ("https://generativelanguage.googleapis.com/v1beta", "https://generativelanguage.googleapis.com/v1beta"),
        ("http://localhost:4000/gemini", "http://localhost:4000/gemini/v1beta"),
    ],
)
def test_go_gemini_base_url_moves_to_the_google_wire_root(raw, expected):
    assert normalize_gemini_base_url(raw) == expected
    assert OPENCODE_GO_GEMINI_BASE_URL == GO_DOCUMENTED


def _serve(requests, *, stream):
    def serve(request):
        requests.append(request)
        if "/models/" in request.url.path:
            payload = {"candidates": [{"content": {"parts": [{"text": "OK"}]}, "finishReason": "STOP"}]}
        else:
            payload = {"id": "chat-test", "object": "chat.completion", "model": MODEL,
                       "choices": [{"index": 0, "message": {"role": "assistant", "content": "OK"},
                                    "finish_reason": "stop"}]}
        if stream:
            if "/models/" not in request.url.path:
                payload["object"] = "chat.completion.chunk"
                payload["choices"][0]["delta"] = payload["choices"][0].pop("message")
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  text=f"data: {json.dumps(payload)}\n\n")
        return httpx.Response(200, json=payload)
    return serve


@pytest.mark.parametrize("stream", [False, True])
def test_go_gemini_main_turn_uses_generate_content(stream):
    from types import SimpleNamespace

    requests = []
    profile = get_provider_profile("opencode-go")
    agent = SimpleNamespace(provider="opencode-go", requested_provider="opencode-go", model=MODEL,
                            _client_log_context=lambda: "test",
                            _build_keepalive_http_client=lambda *a, **k: None)
    client_kwargs = {"api_key": "test-key", "base_url": GO_LEGACY,
                     "default_headers": {**profile.default_headers, "x-test-default": "preserved"},
                     "http_client": httpx.Client(transport=httpx.MockTransport(_serve(requests, stream=stream)))}
    client = create_openai_client(agent, client_kwargs, reason="main", shared=False)
    try:
        assert isinstance(client, GeminiNativeClient)
        assert client.base_url == GO_DOCUMENTED
        kwargs = ChatCompletionsTransport().build_kwargs(
            MODEL, [{"role": "user", "content": "Reply OK"}], provider_profile=profile,
            provider_name="opencode-go", base_url=GO_LEGACY,
            reasoning_config={"enabled": False, "effort": "none"},
        )
        kwargs.update(stream=stream, max_tokens=64)
        merge_session_affinity_headers(kwargs, "opencode-go", GO_LEGACY, SESSION)
        result = client.chat.completions.create(**kwargs)
        if stream:
            assert "".join(chunk.choices[0].delta.content or "" for chunk in result) == "OK"
        else:
            assert result.choices[0].message.content == "OK"
    finally:
        client.close()

    request = requests[-1]
    method = "streamGenerateContent" if stream else "generateContent"
    assert request.url.path == f"/inference/go/v1/models/{MODEL}:{method}", request.url
    assert request.url.params.get("alt") == ("sse" if stream else None)
    assert request.headers["x-goog-api-key"] == "test-key"
    # Session affinity and the profile's own headers survive the native path.
    assert request.headers["x-opencode-session"] == SESSION
    assert request.headers["x-test-default"] == "preserved"


def test_go_sibling_model_keeps_the_chat_completions_wire():
    from types import SimpleNamespace

    requests = []
    profile = get_provider_profile("opencode-go")
    agent = SimpleNamespace(provider="opencode-go", requested_provider="opencode-go",
                            model="deepseek-v4.1-flash", _client_log_context=lambda: "test",
                            _build_keepalive_http_client=lambda *a, **k: None)
    client = create_openai_client(
        agent,
        {"api_key": "test-key", "base_url": GO_LEGACY,
         "http_client": httpx.Client(transport=httpx.MockTransport(_serve(requests, stream=False)))},
        reason="main", shared=False,
    )
    try:
        assert not isinstance(client, GeminiNativeClient)
        kwargs = ChatCompletionsTransport().build_kwargs(
            "deepseek-v4.1-flash", [{"role": "user", "content": "Reply OK"}], provider_profile=profile,
            provider_name="opencode-go", base_url=GO_LEGACY,
            reasoning_config={"enabled": False, "effort": "none"},
        )
        kwargs.update(max_tokens=16)
        client.chat.completions.create(**kwargs)
    finally:
        client.close()
    assert requests[-1].url.path.endswith("/chat/completions"), requests[-1].url


def test_go_gemini_reasoning_rides_thinking_config():
    profile = get_provider_profile("opencode-go")
    extra_body, top_level = profile.build_api_kwargs_extras(
        reasoning_config={"enabled": True, "effort": "high"}, model=MODEL, base_url=GO_LEGACY,
    )
    assert extra_body.get("thinking_config")
    assert not top_level

    # A sibling model keeps its own projection (GLM-5.2's top-level reasoning_effort).
    sibling_extra, sibling_top = profile.build_api_kwargs_extras(
        reasoning_config={"enabled": True, "effort": "high"}, model="glm-5.2", base_url=GO_LEGACY,
    )
    assert not sibling_extra and sibling_top.get("reasoning_effort") in {"high", "max"}
