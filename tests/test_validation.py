"""Tests for validation: crash on bad input, types, Literal enforcement."""

import pytest

from vastai_mcp.tools import (
    _build_offer_query,
    _parse_order,
    _parse_ram_mb,
    _ram_mb_ceil,
    _ram_mb_floor,
    _validate_env,
    _validate_search_params,
)
from vastai_mcp.server import (
    _all_grouped,
    _build_help,
    _coerce_call,
    _dispatch,
    _format_type,
    _get_literal_values,
    _group_ops,
    _to_pascal,
)


# ── _parse_ram_mb ────────────────────────────────────────────────────

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
        with pytest.raises(ValueError, match="must include units"):
            _parse_ram_mb(24)

    def test_bare_float_crashes(self):
        with pytest.raises(ValueError, match="must include units"):
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


# ── tolerance ────────────────────────────────────────────────────────

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


# ── _build_offer_query ───────────────────────────────────────────────

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


# ── _parse_order ─────────────────────────────────────────────────────

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


# ── _coerce_call ─────────────────────────────────────────────────────

class TestCoerceCall:
    def _get_search_offers(self):
        return _group_ops["vastai_read"]["SearchOffers"]

    def test_unknown_param_crashes(self):
        fn = self._get_search_offers()
        with pytest.raises(ValueError, match="Unknown parameters.*bogus"):
            _coerce_call(fn, {"bogus": 1})

    def test_unknown_param_lists_valid(self):
        fn = self._get_search_offers()
        with pytest.raises(ValueError, match="Valid:.*gpu_name"):
            _coerce_call(fn, {"bogus": 1})

    def test_invalid_literal_crashes(self):
        fn = self._get_search_offers()
        with pytest.raises(ValueError, match="Invalid value.*spot"):
            _coerce_call(fn, {"type": "spot"})

    def test_invalid_literal_lists_accepted(self):
        fn = self._get_search_offers()
        with pytest.raises(ValueError, match="on-demand.*bid.*interruptible"):
            _coerce_call(fn, {"type": "spot"})

    def test_valid_literal_passes(self):
        # Should not raise — just validates, doesn't call API
        fn = self._get_search_offers()
        # This will fail at API call, but should pass validation
        try:
            _coerce_call(fn, {"type": "on-demand"})
        except ValueError:
            pytest.fail("Valid Literal value should not raise ValueError")
        except Exception:
            pass  # API call failure is expected without credentials


# ── _get_literal_values ──────────────────────────────────────────────

class TestGetLiteralValues:
    def test_plain_literal(self):
        from typing import Literal
        assert _get_literal_values(Literal["a", "b"]) == ("a", "b")

    def test_optional_literal(self):
        from typing import Literal, Optional
        vals = _get_literal_values(Literal["a", "b"] | None)
        assert vals == ("a", "b")

    def test_non_literal(self):
        assert _get_literal_values(str) is None

    def test_plain_int(self):
        assert _get_literal_values(int) is None


# ── _format_type ─────────────────────────────────────────────────────

class TestFormatType:
    def test_str(self):
        assert _format_type(str) == "str"

    def test_int(self):
        assert _format_type(int) == "int"

    def test_bool(self):
        assert _format_type(bool) == "bool"

    def test_literal(self):
        from typing import Literal
        assert _format_type(Literal["a", "b"]) == "a|b"

    def test_optional_literal(self):
        from typing import Literal
        result = _format_type(Literal["x", "y"] | None)
        assert result == "x|y"

    def test_optional_str(self):
        result = _format_type(str | None)
        assert result == "str"


# ── _build_help ──────────────────────────────────────────────────────

class TestBuildHelp:
    def test_shows_types(self):
        h = _build_help("vastai_read")
        assert "gpu_ram_min: str" in h
        assert "dph_total: float" in h

    def test_shows_literal_values(self):
        h = _build_help("vastai_read")
        assert "on-demand|bid|interruptible" in h

    def test_shows_operation_count(self):
        h = _build_help("vastai_read")
        assert h.startswith("23 operations available:")


# ── _to_pascal ───────────────────────────────────────────────────────

class TestToPascal:
    def test_basic(self):
        assert _to_pascal("search_offers") == "SearchOffers"

    def test_single(self):
        assert _to_pascal("show") == "Show"


# ── edge cases: _parse_ram_mb ────────────────────────────────────────

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
        with pytest.raises(ValueError, match="must include units"):
            _parse_ram_mb(True)

    def test_list_crashes(self):
        """Agent might send [24, 'GB']."""
        with pytest.raises((ValueError, TypeError)):
            _parse_ram_mb([24, "GB"])


# ── edge cases: tolerance boundary math ──────────────────────────────

class TestRamToleranceBoundary:
    """Real GPU sizes from Vast.ai API vs nominal."""

    # Known real values: RTX 4090=24564, RTX 4090 48GB=49140,
    # A100 40GB≈40960, A100 80GB≈81920, H100 80GB≈81559

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


# ── _coerce_call: tricky agent inputs ───────────────────────────────

class TestCoerceCallEdge:
    def _get_fn(self, group, op):
        return _group_ops[group][op]

    def test_multiple_unknown_params(self):
        fn = self._get_fn("vastai_read", "SearchOffers")
        with pytest.raises(ValueError, match="Unknown parameters") as exc:
            _coerce_call(fn, {"foo": 1, "bar": 2, "baz": 3})
        msg = str(exc.value)
        assert "bar" in msg and "baz" in msg and "foo" in msg

    def test_old_gpu_ram_param_rejected(self):
        """Agent using old API should get a clear error, not silent ignore."""
        fn = self._get_fn("vastai_read", "SearchOffers")
        with pytest.raises(ValueError, match="Unknown parameters.*gpu_ram"):
            _coerce_call(fn, {"gpu_ram": "24GB"})

    def test_old_raw_query_param_rejected(self):
        fn = self._get_fn("vastai_read", "SearchOffers")
        with pytest.raises(ValueError, match="Unknown parameters.*raw_query"):
            _coerce_call(fn, {"raw_query": {"gpu_name": {"eq": "RTX 4090"}}})

    def test_empty_params_ok(self):
        """Empty params should work — all SearchOffers params have defaults."""
        fn = self._get_fn("vastai_read", "SearchOffers")
        try:
            _coerce_call(fn, {})
        except ValueError:
            pytest.fail("Empty params should not raise ValueError")
        except Exception:
            pass  # API failure expected

    def test_none_value_for_optional_passes(self):
        fn = self._get_fn("vastai_read", "SearchOffers")
        try:
            _coerce_call(fn, {"gpu_name": None})
        except ValueError:
            pytest.fail("None for optional param should not raise ValueError")
        except Exception:
            pass

    def test_literal_case_sensitive(self):
        """'On-Demand' is not 'on-demand'."""
        fn = self._get_fn("vastai_read", "SearchOffers")
        with pytest.raises(ValueError, match="Invalid value"):
            _coerce_call(fn, {"type": "On-Demand"})

    def test_literal_none_for_optional_passes(self):
        """None should bypass Literal validation for optional params."""
        fn = self._get_fn("vastai_read", "SearchOffers")
        try:
            _coerce_call(fn, {"type": None})
        except ValueError:
            pytest.fail("None for optional Literal should not raise")
        except Exception:
            pass

    def test_manage_instance_literal(self):
        fn = self._get_fn("vastai_write", "ManageInstance")
        with pytest.raises(ValueError, match="Invalid value.*paused"):
            _coerce_call(fn, {"id": 1, "state": "paused"})

    def test_manage_instance_valid_state(self):
        fn = self._get_fn("vastai_write", "ManageInstance")
        try:
            _coerce_call(fn, {"id": 1, "state": "running"})
        except ValueError:
            pytest.fail("'running' is a valid state")
        except Exception:
            pass

    def test_bool_coercion_from_string(self):
        fn = self._get_fn("vastai_read", "SearchOffers")
        try:
            _coerce_call(fn, {"verified": "true"})
        except ValueError:
            pytest.fail("String 'true' should coerce to bool")
        except Exception:
            pass

    def test_bool_coercion_from_int(self):
        fn = self._get_fn("vastai_read", "SearchOffers")
        try:
            _coerce_call(fn, {"verified": 1})
        except ValueError:
            pytest.fail("Int 1 should coerce to bool")
        except Exception:
            pass


# ── _dispatch ────────────────────────────────────────────────────────

class TestDispatch:
    def test_wrong_group_gives_hint(self):
        result = _dispatch("DestroyInstance", "vastai_read", {})
        assert "error" in result
        assert "vastai_delete" in result["error"]

    def test_unknown_operation(self):
        result = _dispatch("BogusOp", "vastai_read", {})
        assert "error" in result
        assert "help" in result["error"]

    def test_help_operation(self):
        from vastai_mcp.server import _build_help
        # _dispatch is not called for help (handled in tool_fn), but
        # _build_help should not crash for any registered group
        for group_name in _group_ops:
            h = _build_help(group_name)
            assert "operations available" in h


# ── _validate_env ────────────────────────────────────────────────────

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


# ── _validate_search_params ──────────────────────────────────────────

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


# ── registration integrity ──────────────────────────────────────────

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
