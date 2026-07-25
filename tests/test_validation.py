"""Tests for validation: crash on bad input, types, Literal enforcement."""

import json
import pathlib
from typing import Literal

import httpx
import pytest

from vastai_mcp import tools
from vastai_mcp.client import APIError, VastClient
from vastai_mcp.server import (
    _all_grouped,
    _build_help,
    _build_params_model,
    _dispatch,
    _format_type,
    _group_ops,
    _to_pascal,
)
from vastai_mcp.tools import (
    _build_offer_query,
    _parse_date_ts,
    _parse_order,
    _parse_order_by,
    _parse_ram_mb,
    _ram_mb_ceil,
    _ram_mb_floor,
    _resolve_gpu_name,
    _validate_env,
    _validate_search_params,
    search_invoices,
    search_offers,
    search_templates,
    show_invoices_v1,
    vastai_version,
)

# -- _parse_ram_mb --------------------

class TestParseRamMb:
    def test_gb(self):
        assert _parse_ram_mb("24GB") == 24 * 1024

    def test_gb_lowercase(self):
        assert _parse_ram_mb("24gb") == 24 * 1024

    def test_mb(self):
        assert _parse_ram_mb("24564MB") == 24564

    def test_float_gb(self):
        assert _parse_ram_mb("1.5GB") == 1.5 * 1024

    def test_bare_int_crashes(self):
        with pytest.raises(TypeError, match="must include units"):
            _parse_ram_mb(24)

    def test_bare_float_crashes(self):
        with pytest.raises(TypeError, match="must include units"):
            _parse_ram_mb(24.0)

    def test_bare_string_crashes(self):
        with pytest.raises(ValueError, match="must include units"):
            _parse_ram_mb("24")

    def test_no_unit_crashes(self):
        with pytest.raises(ValueError, match="must include units"):
            _parse_ram_mb("24 potatoes")

    def test_empty_crashes(self):
        with pytest.raises(ValueError, match="must include units"):
            _parse_ram_mb("")


# -- tolerance --------------------

class TestRamTolerance:
    def test_floor_catches_24gb_gpu(self):
        # RTX 4090 reports 24564 MB, nominal 24GB = 24576 MB
        floor = _ram_mb_floor(24 * 1024)
        assert floor < 24564, f"floor {floor} should be below 24564"

    def test_ceil_catches_24gb_gpu(self):
        ceil = _ram_mb_ceil(24 * 1024)
        assert ceil > 24564, f"ceil {ceil} should be above 24564"

    def test_floor_catches_48gb_gpu(self):
        # 48GB variant reports 49140 MB, nominal 48GB = 49152 MB
        floor = _ram_mb_floor(48 * 1024)
        assert floor < 49140, f"floor {floor} should be below 49140"

    def test_ceil_catches_48gb_gpu(self):
        ceil = _ram_mb_ceil(48 * 1024)
        assert ceil > 49140, f"ceil {ceil} should be above 49140"

    def test_floor_does_not_leak_to_lower_tier(self):
        # 24GB floor should not catch 16GB GPUs (~16384 MB)
        floor = _ram_mb_floor(24 * 1024)
        assert floor > 16384, f"floor {floor} should be above 16384"

    def test_ceil_does_not_leak_to_upper_tier(self):
        # 24GB ceil should not catch 48GB GPUs (~49140 MB)
        ceil = _ram_mb_ceil(24 * 1024)
        assert ceil < 49140, f"ceil {ceil} should be below 49140"


# -- _build_offer_query --------------------

class TestBuildOfferQuery:
    def test_defaults(self):
        q = _build_offer_query()
        assert q["verified"] == {"eq": True}
        assert q["rentable"] == {"eq": True}
        assert q["rented"] == {"eq": False}
        assert "gpu_ram" not in q

    def test_gpu_name(self):
        q = _build_offer_query(gpu_name="RTX 4090")
        assert q["gpu_name"] == {"eq": "RTX 4090"}

    def test_gpu_ram_min(self):
        q = _build_offer_query(gpu_ram_min_mb=24576.0)
        assert "gte" in q["gpu_ram"]
        assert q["gpu_ram"]["gte"] < 24576  # tolerance applied

    def test_gpu_ram_max(self):
        q = _build_offer_query(gpu_ram_max_mb=24576.0)
        assert "lte" in q["gpu_ram"]
        assert q["gpu_ram"]["lte"] > 24576  # tolerance applied

    def test_gpu_ram_both(self):
        q = _build_offer_query(gpu_ram_min_mb=24000.0, gpu_ram_max_mb=50000.0)
        assert "gte" in q["gpu_ram"]
        assert "lte" in q["gpu_ram"]

    def test_verified_override(self):
        q = _build_offer_query(verified=False)
        assert q["verified"] == {"eq": False}

    def test_type_passthrough(self):
        q = _build_offer_query(type="on-demand")
        assert q["type"] == "on-demand"

    def test_dph_total(self):
        q = _build_offer_query(dph_total=0.5)
        assert q["dph_total"] == {"lte": 0.5}

    def test_reliability(self):
        q = _build_offer_query(reliability=0.99)
        assert q["reliability2"] == {"gte": 0.99}


# -- _parse_order --------------------

class TestParseOrder:
    def test_none(self):
        assert _parse_order(None) is None

    def test_asc_default(self):
        assert _parse_order("dph_total") == [["dph_total", "asc"]]

    def test_desc_prefix(self):
        assert _parse_order("-dph_total") == [["dph_total", "desc"]]

    def test_explicit_asc(self):
        assert _parse_order("dph_total-asc") == [["dph_total", "asc"]]

    def test_explicit_desc(self):
        assert _parse_order("dph_total-desc") == [["dph_total", "desc"]]


# -- params model --------------------

class TestParamsModel:
    """Validation now comes from the Pydantic model built off the signature."""

    def test_unknown_param_crashes(self):
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            _dispatch("SearchOffers", "vastai_read", {"bogus": 1})

    def test_unknown_param_names_the_key(self):
        with pytest.raises(ValueError, match="bogus"):
            _dispatch("SearchOffers", "vastai_read", {"bogus": 1})

    def test_error_points_at_schema(self):
        with pytest.raises(ValueError, match="operation='schema'"):
            _dispatch("SearchOffers", "vastai_read", {"bogus": 1})

    def test_wrong_type_crashes(self):
        with pytest.raises(ValueError, match="Input should be a valid integer"):
            _dispatch("SearchOffers", "vastai_read", {"limit": "abc"})

    def test_wrong_type_names_the_field(self):
        with pytest.raises(ValueError) as exc:
            _dispatch("SearchOffers", "vastai_read", {"limit": "abc"})
        assert "- limit:" in str(exc.value)

    def test_wrong_type_echoes_the_input(self):
        with pytest.raises(ValueError, match="got 'abc'"):
            _dispatch("SearchOffers", "vastai_read", {"limit": "abc"})

    def test_missing_required_crashes(self):
        with pytest.raises(ValueError, match="Field required"):
            _dispatch("ShowInstance", "vastai_read", {})

    def test_invalid_literal_crashes(self):
        with pytest.raises(ValueError, match="Input should be"):
            _dispatch("SearchOffers", "vastai_read", {"type": "spot"})

    def test_invalid_literal_lists_accepted(self):
        with pytest.raises(ValueError, match="'on-demand', 'bid' or 'interruptible'"):
            _dispatch("SearchOffers", "vastai_read", {"type": "spot"})

    def test_valid_literal_reaches_query(self, fake_client):
        client = fake_client({"offers": []})
        _dispatch("SearchOffers", "vastai_read", {"type": "on-demand"})
        assert client.calls[0][2]["json"]["type"] == "on-demand"

    def test_numeric_string_still_coerced(self, fake_client):
        client = fake_client({"offers": []})
        _dispatch("SearchOffers", "vastai_read", {"limit": "5"})
        assert client.calls[0][2]["json"]["limit"] == 5

    def test_model_forbids_extra(self):
        model = _build_params_model(tools.search_offers)
        assert model.model_config["extra"] == "forbid"

    def test_model_keeps_signature_order(self):
        model = _build_params_model(tools.search_offers)
        assert list(model.model_fields)[:3] == ["limit", "gpu_name", "num_gpus"]

    def test_annotated_description_reaches_field(self):
        model = _build_params_model(tools.search_offers)
        assert "24GB" in model.model_fields["gpu_ram_min"].description


# -- _format_type --------------------

class TestFormatType:
    def test_str(self):
        assert _format_type(str) == "str"

    def test_int(self):
        assert _format_type(int) == "int"

    def test_bool(self):
        assert _format_type(bool) == "bool"

    def test_literal(self):
        assert _format_type(Literal["a", "b"]) == "a|b"

    def test_optional_literal(self):
        result = _format_type(Literal["x", "y"] | None)
        assert result == "x|y"

    def test_optional_str(self):
        result = _format_type(str | None)
        assert result == "str"


# -- _build_help --------------------

class TestBuildHelp:
    def test_shows_types(self):
        h = _build_help("vastai_read")
        assert "gpu_ram_min?: str" in h
        assert "dph_total?: float" in h

    def test_marks_required_params(self):
        h = _build_help("vastai_read")
        assert "ShowInstance(id: int)" in h

    def test_shows_defaults(self):
        h = _build_help("vastai_read")
        assert "limit?: int=20" in h

    def test_shows_literal_values(self):
        h = _build_help("vastai_read")
        assert "on-demand|bid|interruptible" in h

    def test_shows_operation_count(self):
        h = _build_help("vastai_read")
        assert h.startswith("24 operations available.")

    def test_points_at_schema(self):
        h = _build_help("vastai_read")
        assert "operation='schema'" in h

    def test_bullets_come_from_field_descriptions(self):
        h = _build_help("vastai_read")
        assert "    gpu_ram_min: Units required" in h

    def test_only_described_params_get_a_bullet(self):
        h = _build_help("vastai_read")
        assert "    num_gpus:" not in h

    def test_constraints_left_the_docstring_body(self):
        """A constraint must reach help through Field(description=...), not both ways."""
        doc = tools.search_offers.__doc__
        assert "24GB" not in doc


# -- _to_pascal --------------------

class TestToPascal:
    def test_basic(self):
        assert _to_pascal("search_offers") == "SearchOffers"

    def test_single(self):
        assert _to_pascal("show") == "Show"


# -- edge cases: _parse_ram_mb --------------------

class TestParseRamMbEdge:
    def test_whitespace_around(self):
        assert _parse_ram_mb("  24GB  ") == 24 * 1024

    def test_zero_gb(self):
        assert _parse_ram_mb("0GB") == 0

    def test_mixed_case(self):
        assert _parse_ram_mb("24Gb") == 24 * 1024

    def test_negative_crashes(self):
        with pytest.raises(ValueError, match="must include units"):
            _parse_ram_mb("-24GB")

    def test_gb_without_number_crashes(self):
        with pytest.raises(ValueError, match="must include units"):
            _parse_ram_mb("GB")

    def test_two_units_crashes(self):
        with pytest.raises(ValueError, match="must include units"):
            _parse_ram_mb("24GBMB")

    def test_bool_true_crashes(self):
        """Agent might coerce True from JSON."""
        with pytest.raises(TypeError, match="must include units"):
            _parse_ram_mb(True)

    def test_list_crashes(self):
        """Agent might send [24, 'GB']."""
        with pytest.raises((ValueError, TypeError)):
            _parse_ram_mb([24, "GB"])


# -- edge cases: tolerance boundary math --------------------

class TestRamToleranceBoundary:
    """Real GPU sizes from Vast.ai API vs nominal."""

    # Known real values: RTX 4090=24564, RTX 4090 48GB=49140,
    # A100 40GB~40960, A100 80GB~81920, H100 80GB~81559

    @pytest.mark.parametrize("nominal_gb,real_mb", [
        (24, 24564),   # RTX 4090
        (48, 49140),   # RTX 4090 48GB
        (40, 40960),   # A100 40GB
        (80, 81559),   # H100 80GB
        (16, 16384),   # RTX 4080
        (12, 12288),   # RTX 4070
    ])
    def test_floor_catches_real_gpu(self, nominal_gb, real_mb):
        floor = _ram_mb_floor(nominal_gb * 1024)
        assert floor <= real_mb, (
            f"{nominal_gb}GB floor={floor} should be <= real {real_mb}"
        )

    @pytest.mark.parametrize("nominal_gb,real_mb", [
        (24, 24564),
        (48, 49140),
        (40, 40960),
        (80, 81559),
    ])
    def test_ceil_catches_real_gpu(self, nominal_gb, real_mb):
        ceil = _ram_mb_ceil(nominal_gb * 1024)
        assert ceil >= real_mb, (
            f"{nominal_gb}GB ceil={ceil} should be >= real {real_mb}"
        )

    @pytest.mark.parametrize("lower_gb,upper_real_mb", [
        (24, 49140),   # 24GB ceil must not reach 48GB
        (16, 24564),   # 16GB ceil must not reach 24GB
        (48, 81559),   # 48GB ceil must not reach 80GB
    ])
    def test_ceil_does_not_bleed_up(self, lower_gb, upper_real_mb):
        ceil = _ram_mb_ceil(lower_gb * 1024)
        assert ceil < upper_real_mb, (
            f"{lower_gb}GB ceil={ceil} must not reach {upper_real_mb}"
        )

    @pytest.mark.parametrize("upper_gb,lower_real_mb", [
        (48, 24564),   # 48GB floor must not drop to 24GB
        (24, 16384),   # 24GB floor must not drop to 16GB
        (80, 49140),   # 80GB floor must not drop to 48GB
    ])
    def test_floor_does_not_bleed_down(self, upper_gb, lower_real_mb):
        floor = _ram_mb_floor(upper_gb * 1024)
        assert floor > lower_real_mb, (
            f"{upper_gb}GB floor={floor} must not drop to {lower_real_mb}"
        )


# -- params model: tricky agent inputs --------------------

class TestParamsModelEdge:
    def test_multiple_unknown_params(self):
        with pytest.raises(ValueError) as exc:
            _dispatch("SearchOffers", "vastai_read", {"foo": 1, "bar": 2, "baz": 3})
        msg = str(exc.value)
        assert "bar" in msg and "baz" in msg and "foo" in msg

    def test_old_gpu_ram_param_rejected(self):
        """Agent using old API should get a clear error, not silent ignore."""
        with pytest.raises(ValueError, match="gpu_ram: Extra inputs are not permitted"):
            _dispatch("SearchOffers", "vastai_read", {"gpu_ram": "24GB"})

    def test_old_raw_query_param_rejected(self):
        with pytest.raises(ValueError, match="raw_query: Extra inputs are not permitted"):
            _dispatch(
                "SearchOffers", "vastai_read", {"raw_query": {"gpu_name": {"eq": "RTX 4090"}}}
            )

    def test_empty_params_ok(self, fake_client):
        """Empty params should work - all SearchOffers params have defaults."""
        client = fake_client({"offers": []})
        _dispatch("SearchOffers", "vastai_read", {})
        assert client.calls[0][1] == "/api/v0/bundles/"

    def test_none_value_for_optional_is_omitted(self, fake_client):
        client = fake_client({"offers": []})
        _dispatch("SearchOffers", "vastai_read", {"gpu_name": None})
        assert "gpu_name" not in client.calls[0][2]["json"]

    def test_literal_case_sensitive(self):
        """'On-Demand' is not 'on-demand'."""
        with pytest.raises(ValueError, match="Input should be"):
            _dispatch("SearchOffers", "vastai_read", {"type": "On-Demand"})

    def test_literal_none_for_optional_is_omitted(self, fake_client):
        """None should bypass Literal validation for optional params."""
        client = fake_client({"offers": []})
        _dispatch("SearchOffers", "vastai_read", {"type": None})
        assert "type" not in client.calls[0][2]["json"]

    def test_manage_instance_literal(self):
        with pytest.raises(ValueError, match="'running' or 'stopped'"):
            _dispatch("ManageInstance", "vastai_write", {"id": 1, "state": "paused"})

    def test_manage_instance_valid_state(self, fake_client):
        client = fake_client({"success": True})
        _dispatch("ManageInstance", "vastai_write", {"id": 1, "state": "running"})
        assert client.calls[0][2]["json"] == {"state": "running"}

    def test_bool_coercion_from_string(self, fake_client):
        client = fake_client({"offers": []})
        _dispatch("SearchOffers", "vastai_read", {"verified": "true"})
        assert client.calls[0][2]["json"]["verified"] == {"eq": True}

    def test_bool_coercion_from_int(self, fake_client):
        client = fake_client({"offers": []})
        _dispatch("SearchOffers", "vastai_read", {"verified": 1})
        assert client.calls[0][2]["json"]["verified"] == {"eq": True}


# -- _dispatch --------------------

class TestDispatch:
    def test_wrong_group_crashes_with_hint(self):
        with pytest.raises(ValueError, match="vastai_delete"):
            _dispatch("DestroyInstance", "vastai_read", {})

    def test_unknown_operation_crashes(self):
        with pytest.raises(ValueError, match="Unknown operation"):
            _dispatch("BogusOp", "vastai_read", {})

    def test_unknown_operation_points_at_help(self):
        with pytest.raises(ValueError, match="help"):
            _dispatch("BogusOp", "vastai_read", {})

    def test_help_operation(self):
        # _dispatch is not called for help (handled in tool_fn), but
        # _build_help should not crash for any registered group
        for group_name in _group_ops:
            h = _build_help(group_name)
            assert "operations available" in h


# -- operation='schema' --------------------

class TestSchemaOperation:
    def test_no_op_lists_operations(self):
        result = _dispatch("schema", "vastai_read", {})
        assert "SearchOffers" in result["operations"]

    def test_no_op_hints_at_op_param(self):
        result = _dispatch("schema", "vastai_read", {})
        assert "op" in result["hint"]

    def test_op_returns_json_schema(self):
        schema = _dispatch("schema", "vastai_read", {"op": "SearchOffers"})
        assert schema["additionalProperties"] is False
        assert schema["properties"]["limit"]["type"] == "integer"

    def test_schema_carries_literal_enum(self):
        schema = _dispatch("schema", "vastai_read", {"op": "SearchOffers"})
        variants = schema["properties"]["type"]["anyOf"]
        assert {"enum": ["on-demand", "bid", "interruptible"], "type": "string"} in variants

    def test_schema_carries_param_description(self):
        schema = _dispatch("schema", "vastai_read", {"op": "SearchOffers"})
        assert "24GB" in schema["properties"]["gpu_ram_min"]["description"]

    def test_schema_lists_required_fields(self):
        schema = _dispatch("schema", "vastai_read", {"op": "ShowInstance"})
        assert schema["required"] == ["id"]

    def test_schema_carries_docstring(self):
        schema = _dispatch("schema", "vastai_read", {"op": "SearchOffers"})
        assert schema["description"].startswith("Search GPU offers")

    def test_unknown_op_crashes(self):
        with pytest.raises(ValueError, match="Unknown operation"):
            _dispatch("schema", "vastai_read", {"op": "BogusOp"})

    def test_every_op_has_a_schema(self):
        for group_name, ops in _group_ops.items():
            for op_name in ops:
                schema = _dispatch("schema", group_name, {"op": op_name})
                assert schema["type"] == "object"


# -- _validate_env --------------------

class TestValidateEnv:
    def test_port_mapping_crashes(self):
        with pytest.raises(ValueError, match="port mappings"):
            _validate_env("-p 8080:8080 -e FOO=bar")

    def test_port_at_end_crashes(self):
        with pytest.raises(ValueError, match="port mappings"):
            _validate_env("-e FOO=bar -p")

    def test_valid_env_passes(self):
        _validate_env("-e FOO=bar -e BAZ=qux")

    def test_space_in_value_crashes(self):
        with pytest.raises(ValueError, match="will be lost"):
            _validate_env("-e KEY=val1 val2")


# -- _validate_search_params --------------------

class TestValidateSearchParams:
    def test_missing_rentable_crashes(self):
        with pytest.raises(ValueError, match="rentable"):
            _validate_search_params("rented=false gpu_name=RTX_4090")

    def test_missing_rented_crashes(self):
        with pytest.raises(ValueError, match="rented"):
            _validate_search_params("rentable=true gpu_name=RTX_4090")

    def test_bare_gpu_ram_crashes(self):
        with pytest.raises(ValueError, match="must include units"):
            _validate_search_params(
                "rentable=true rented=false gpu_ram>=10000"
            )

    def test_gpu_ram_with_units_passes(self):
        result = _validate_search_params(
            "rentable=true rented=false gpu_ram>=12GB"
        )
        assert "gpu_ram" in result
        # Should have converted to bare GB number
        assert "GB" not in result

    def test_valid_params_pass(self):
        result = _validate_search_params(
            "rentable=true rented=false gpu_name=RTX_4090"
        )
        assert result == "rentable=true rented=false gpu_name=RTX_4090"


# -- _parse_order_by --------------------

class TestParseOrderBy:
    def test_none(self):
        assert _parse_order_by(None) is None

    def test_asc_default(self):
        assert _parse_order_by("count_created") == [{"col": "count_created", "dir": "asc"}]

    def test_desc_prefix(self):
        assert _parse_order_by("-count_created") == [{"col": "count_created", "dir": "desc"}]


# -- _resolve_gpu_name --------------------

@pytest.fixture
def gpu_catalog(monkeypatch):
    monkeypatch.setattr(tools, "_GPU_NAMES", ["RTX 4090", "A100 PCIE", "GTX 1650 S"])


class TestResolveGpuName:
    def test_exact(self, gpu_catalog):
        assert _resolve_gpu_name("RTX 4090") == "RTX 4090"

    def test_underscores(self, gpu_catalog):
        assert _resolve_gpu_name("RTX_4090") == "RTX 4090"

    def test_dashes_and_case(self, gpu_catalog):
        assert _resolve_gpu_name("a100-pcie") == "A100 PCIE"

    def test_unknown_crashes(self, gpu_catalog):
        with pytest.raises(ValueError, match="not a known GPU"):
            _resolve_gpu_name("RTX 9090")

    def test_unknown_lists_valid(self, gpu_catalog):
        with pytest.raises(ValueError, match="GTX 1650 S"):
            _resolve_gpu_name("RTX 9090")


# -- _parse_date_ts --------------------

class TestParseDateTs:
    def test_iso_date(self):
        assert _parse_date_ts("2026-01-01", "start_date") == 1767225600

    def test_epoch_int(self):
        assert _parse_date_ts(1767225600, "start_date") == 1767225600

    def test_epoch_string(self):
        assert _parse_date_ts("1767225600", "start_date") == 1767225600

    def test_garbage_crashes(self):
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            _parse_date_ts("last tuesday", "start_date")

    def test_error_names_field(self):
        with pytest.raises(ValueError, match="end_date="):
            _parse_date_ts("nope", "end_date")


# -- request shapes --------------------

class _FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, path, **kwargs):
        self.calls.append(("GET", path, kwargs))
        return self.response

    def post(self, path, **kwargs):
        self.calls.append(("POST", path, kwargs))
        return self.response

    def put(self, path, **kwargs):
        self.calls.append(("PUT", path, kwargs))
        return self.response


@pytest.fixture
def fake_client(monkeypatch):
    def install(response):
        client = _FakeClient(response)
        monkeypatch.setattr(tools, "_client", client)
        return client
    return install


class TestSearchOffersRequest:
    def test_limit_is_plain_int(self, fake_client):
        """API rejects the {"eq": n} form with 400."""
        client = fake_client({"offers": []})
        search_offers(limit=5)
        assert client.calls[0][2]["json"]["limit"] == 5

    def test_limit_coerced_to_int(self, fake_client):
        """A string limit is silently ignored by the API."""
        client = fake_client({"offers": []})
        search_offers(limit="5")
        assert client.calls[0][2]["json"]["limit"] == 5

    def test_gpu_name_resolved(self, fake_client, gpu_catalog):
        client = fake_client({"offers": []})
        search_offers(gpu_name="RTX_4090")
        assert client.calls[0][2]["json"]["gpu_name"] == {"eq": "RTX 4090"}


class TestSearchTemplatesRequest:
    def test_select_filters_always_sent(self, fake_client):
        """API returns 400 when select_filters is absent."""
        client = fake_client({"templates": []})
        search_templates()
        assert json.loads(client.calls[0][2]["params"]["select_filters"]) == {}

    def test_filters_and_order(self, fake_client):
        client = fake_client({"templates": []})
        search_templates(name="pytorch", recommended=True, order="-count_created", limit=5)
        params = client.calls[0][2]["params"]
        assert json.loads(params["select_filters"]) == {
            "name": "pytorch", "recommended": {"eq": True},
        }
        assert json.loads(params["order_by"]) == [{"col": "count_created", "dir": "desc"}]
        assert params["limit"] == 5

    def test_templates_slimmed(self, fake_client):
        fake_client({"templates": [{"id": 1, "name": "x", "docker_login_pass": "secret"}]})
        result = search_templates()
        assert result["templates"] == [{"id": 1, "name": "x"}]


class TestInvoicesRequest:
    def test_v0_path_has_trailing_slash(self, fake_client):
        """Without it the API 301s and the HTML body breaks json()."""
        client = fake_client([])
        search_invoices()
        assert client.calls[0][1] == "/api/v0/invoices/"

    def test_v1_defaults_to_date_range(self, fake_client):
        """API answers 'Invalid date range' when no range is sent."""
        client = fake_client({"results": []})
        show_invoices_v1()
        window = json.loads(client.calls[0][2]["params"]["select_filters"])["when"]
        assert window["lte"] - window["gte"] == 30 * 24 * 60 * 60

    def test_v1_explicit_dates(self, fake_client):
        client = fake_client({"results": []})
        show_invoices_v1(start_date="2026-01-01", end_date="2026-01-08")
        window = json.loads(client.calls[0][2]["params"]["select_filters"])["when"]
        assert window == {"gte": 1767225600, "lte": 1767830400}

    def test_v1_bad_date_crashes(self, fake_client):
        fake_client({"results": []})
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            show_invoices_v1(start_date="yesterday")


# -- client error handling --------------------

class TestRepoAscii:
    def test_sources_are_ascii(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        targets = (
            sorted(root.glob("src/**/*.py"))
            + sorted(root.glob("tests/**/*.py"))
            + sorted(root.glob(".github/workflows/*.yml"))
            + [
                root / "README.md", root / "docs" / "index.html",
                root / "dev.py", root / ".githooks" / "pre-commit",
            ]
        )
        offenders = []
        for path in targets:
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                if any(ord(ch) > 127 for ch in line):
                    offenders.append(f"{path.relative_to(root)}:{lineno}: {line}")
        assert not offenders, "non-ASCII found:\n" + "\n".join(offenders)


class TestVastaiVersion:
    def test_service_error_propagates(self, monkeypatch):
        """A dead API must surface as APIError, not as {"status": "error"}."""
        class _DeadClient:
            def get(self, path, **kwargs):
                raise APIError(500, "GET", path, {"msg": "down"})

        monkeypatch.setattr(tools, "_client", _DeadClient())
        with pytest.raises(APIError):
            vastai_version()

    def test_service_ok(self, fake_client):
        fake_client({"id": 1})
        assert vastai_version()["service"] == {"status": "ok"}


class TestClientHandle:
    def test_redirect_body_is_text(self):
        """301 bodies are HTML: keep the text, never let the parser error escape."""
        r = httpx.Response(
            301, request=httpx.Request("GET", "https://console.vast.ai/api/v0/invoices"),
            html="<html>moved</html>",
        )
        with pytest.raises(APIError) as excinfo:
            VastClient()._handle(r)
        assert "moved" in excinfo.value.body

    def test_json_error_body_is_parsed(self):
        r = httpx.Response(
            400, request=httpx.Request("POST", "https://console.vast.ai/api/v0/bundles/"),
            json={"msg": "limit: Input should be a valid integer"},
        )
        with pytest.raises(APIError) as excinfo:
            VastClient()._handle(r)
        assert excinfo.value.body["msg"].startswith("limit:")

    def test_redirect_raises(self):
        r = httpx.Response(
            301, request=httpx.Request("GET", "https://console.vast.ai/api/v0/invoices"),
            html="<html>moved</html>",
        )
        with pytest.raises(APIError):
            VastClient()._handle(r)

    def test_error_raises(self):
        r = httpx.Response(
            400, request=httpx.Request("POST", "https://console.vast.ai/api/v0/bundles/"),
            json={"msg": "bad"},
        )
        with pytest.raises(APIError):
            VastClient()._handle(r)


# -- registration integrity --------------------

class TestRegistration:
    def test_all_ops_have_docstrings(self):
        for group_name, ops in _group_ops.items():
            for pascal_name, fn in ops.items():
                assert fn.__doc__, (
                    f"{group_name}.{pascal_name} has no docstring"
                )

    def test_no_duplicate_operations_across_groups(self):
        seen = {}
        for group_name, ops in _group_ops.items():
            for pascal_name in ops:
                assert pascal_name not in seen, (
                    f"{pascal_name} registered in both "
                    f"{seen[pascal_name]} and {group_name}"
                )
                seen[pascal_name] = group_name

    def test_all_grouped_index_complete(self):
        """_all_grouped must have every operation from every group."""
        for group_name, ops in _group_ops.items():
            for pascal_name in ops:
                assert pascal_name in _all_grouped
                assert _all_grouped[pascal_name] == group_name

    def test_group_names_match_tool_names(self):
        expected = {"vastai_read", "vastai_write", "vastai_execute", "vastai_delete"}
        assert set(_group_ops.keys()) == expected
