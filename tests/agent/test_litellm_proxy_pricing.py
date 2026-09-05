"""Pricing for model aliases served through a LiteLLM proxy.

A LiteLLM proxy's OpenAI-compatible ``/v1/models`` lists bare alias ids with
no pricing, so every session routed through one priced as ``unknown`` and
``estimated_cost_usd`` stayed NULL in state.db. The proxy publishes the rate
card under ``/model/info``; these tests pin the enrichment path and the
upstream-model fallback for aliases LiteLLM itself cannot price.
"""

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

import agent.model_metadata as model_metadata
from agent.model_metadata import fetch_endpoint_model_metadata
from agent.usage_pricing import CanonicalUsage, estimate_usage_cost, get_pricing_entry

BASE = "http://127.0.0.1:4000/v1"

MODELS_PAYLOAD = {
    "object": "list",
    "data": [
        {"id": "bedrock-fable-5-1", "object": "model", "owned_by": "openai"},
        {"id": "bedrock-sol", "object": "model", "owned_by": "openai"},
        {"id": "local-cua", "object": "model", "owned_by": "openai"},
        {"id": "bedrock-mystery", "object": "model", "owned_by": "openai"},
    ],
}

MODEL_INFO_PAYLOAD = {
    "data": [
        {
            "model_name": "bedrock-fable-5-1",
            "litellm_params": {"model": "bedrock/us.anthropic.claude-fable-5-1", "aws_region_name": "us-east-1"},
            "model_info": {
                "input_cost_per_token": 1.1e-05,
                "output_cost_per_token": 5.5e-05,
                "cache_read_input_token_cost": 2.75e-07,
                "cache_creation_input_token_cost": 1.375e-05,
                "max_input_tokens": 1_000_000,
            },
        },
        {
            # LiteLLM has no rate card for this upstream: reports 0 / None.
            "model_name": "bedrock-sol",
            "litellm_params": {
                "model": "openai/responses/openai.gpt-5.6-sol",
                "api_base": "https://bedrock-mantle.us-east-1.api.aws/openai/v1",
            },
            "model_info": {
                "input_cost_per_token": 0,
                "output_cost_per_token": 0,
                "cache_read_input_token_cost": None,
                "cache_creation_input_token_cost": None,
            },
        },
        {
            # Operator explicitly declared this local route free.
            "model_name": "local-cua",
            "litellm_params": {
                "model": "openai/local-cua",
                "api_base": "http://127.0.0.1:8101/v1",
                "input_cost_per_token": 0,
                "output_cost_per_token": 0,
            },
            "model_info": {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0},
        },
        {
            # Unknown upstream, no declared price: must stay unknown, not $0.
            "model_name": "bedrock-mystery",
            "litellm_params": {"model": "bedrock/converse/vendor.mystery-model"},
            "model_info": {"input_cost_per_token": 0, "output_cost_per_token": 0},
        },
    ]
}


def _resp(payload, ok=True):
    resp = MagicMock()
    resp.ok = ok
    resp.status_code = 200 if ok else 404
    resp.raise_for_status.return_value = None
    resp.json.return_value = payload
    return resp


def _fake_get(url, **kwargs):
    if url.endswith("/model/info"):
        return _resp(MODEL_INFO_PAYLOAD)
    if url.endswith("/models"):
        return _resp(MODELS_PAYLOAD)
    return _resp({}, ok=False)


@pytest.fixture(autouse=True)
def _clear_endpoint_cache():
    model_metadata._endpoint_model_metadata_cache.clear()
    model_metadata._endpoint_model_metadata_cache_time.clear()
    yield
    model_metadata._endpoint_model_metadata_cache.clear()
    model_metadata._endpoint_model_metadata_cache_time.clear()


@pytest.fixture
def litellm_proxy():
    with patch("agent.model_metadata.detect_local_server_type", return_value="unknown"), \
         patch("agent.model_metadata.requests.get", side_effect=_fake_get) as mock_get:
        yield mock_get


def test_model_info_enriches_alias_pricing(litellm_proxy):
    meta = fetch_endpoint_model_metadata(BASE, force_refresh=True)
    fable = meta["bedrock-fable-5-1"]
    assert fable["pricing"]["prompt"] == 1.1e-05
    assert fable["pricing"]["completion"] == 5.5e-05
    assert fable["pricing"]["cache_read"] == 2.75e-07
    assert fable["pricing"]["cache_write"] == 1.375e-05
    assert fable["upstream_model"] == "bedrock/us.anthropic.claude-fable-5-1"
    # Enrichment must not smuggle in context_length changes (compression
    # thresholds are out of scope for a pricing fix).
    assert "context_length" not in fable
    called = [c.args[0] for c in litellm_proxy.call_args_list]
    assert any(u.endswith("/model/info") for u in called)


def test_model_info_zero_rate_card_is_not_free(litellm_proxy):
    meta = fetch_endpoint_model_metadata(BASE, force_refresh=True)
    assert "pricing" not in meta["bedrock-sol"]
    assert "pricing" not in meta["bedrock-mystery"]
    # Operator-declared zero IS honored.
    assert meta["local-cua"]["pricing"] == {"prompt": 0.0, "completion": 0.0}


def test_alias_prices_through_custom_provider(litellm_proxy):
    entry = get_pricing_entry("bedrock-fable-5-1", provider="custom", base_url=BASE)
    assert entry is not None
    assert entry.source == "provider_models_api"
    assert entry.input_cost_per_million == Decimal("11.0")
    assert entry.output_cost_per_million == Decimal("55.0")
    assert entry.cache_read_cost_per_million == Decimal("0.275")
    assert entry.cache_write_cost_per_million == Decimal("13.75")

    usage = CanonicalUsage(input_tokens=34, output_tokens=5802, cache_read_tokens=800_366, cache_write_tokens=95_547)
    cost = estimate_usage_cost("bedrock-fable-5-1", usage, provider="custom", base_url=BASE)
    assert cost.status == "estimated"
    assert cost.amount_usd is not None
    assert cost.amount_usd == pytest.approx(Decimal("1.8533559"), abs=Decimal("1e-6"))


def test_unpriced_alias_falls_back_to_upstream_official_docs(litellm_proxy):
    entry = get_pricing_entry("bedrock-sol", provider="custom", base_url=BASE)
    assert entry is not None
    assert entry.source == "official_docs_snapshot"
    assert entry.pricing_version == "bedrock-openai-model-card-2026-09"
    assert entry.input_cost_per_million == Decimal("4.40")
    assert entry.output_cost_per_million == Decimal("22.00")
    cost = estimate_usage_cost(
        "bedrock-sol",
        CanonicalUsage(input_tokens=1_000_000, output_tokens=1_000_000),
        provider="custom",
        base_url=BASE,
    )
    assert cost.status == "estimated"
    assert cost.amount_usd == Decimal("26.40")


def test_operator_declared_free_route_is_included(litellm_proxy):
    cost = estimate_usage_cost(
        "local-cua",
        CanonicalUsage(input_tokens=12, output_tokens=2),
        provider="custom",
        base_url=BASE,
    )
    assert cost.amount_usd == Decimal("0")
    assert cost.status == "estimated"


def test_unknown_upstream_stays_unknown(litellm_proxy):
    cost = estimate_usage_cost(
        "bedrock-mystery",
        CanonicalUsage(input_tokens=10, output_tokens=10),
        provider="custom",
        base_url=BASE,
    )
    assert cost.amount_usd is None
    assert cost.status == "unknown"


def test_non_litellm_endpoint_is_unaffected():
    def plain_get(url, **kwargs):
        if url.endswith("/models"):
            return _resp({"data": [{"id": "some-model", "owned_by": "vllm"}]})
        return _resp({"detail": "Not Found"}, ok=False)

    with patch("agent.model_metadata.detect_local_server_type", return_value="unknown"), \
         patch("agent.model_metadata.requests.get", side_effect=plain_get):
        meta = fetch_endpoint_model_metadata("http://127.0.0.1:9999/v1", force_refresh=True)
    assert meta == {"some-model": {"name": "some-model"}}
