import json
import pytest
import numpy as np
import xarray as xr
from unittest.mock import MagicMock

from oedisi.types.data_types import (
    PowersReal,
    PowersImaginary,
    MeasurementArray,
    EquipmentNodeArray,
)

from hub_federate import (
    eqarray_to_xarray,
    measurement_to_xarray,
    xarray_to_dict,
    xarray_to_eqarray,
    xarray_to_powers_cart,
)


# ---------------------------------------------------------------------------
# eqarray_to_xarray
# ---------------------------------------------------------------------------

def test_eqarray_to_xarray_normal():
    """T1 — normal case."""
    eq = EquipmentNodeArray(ids=["a", "b"], equipment_ids=["eq1", "eq2"], values=[1.0, 2.0], units="kW")
    da = eqarray_to_xarray(eq)
    assert list(da.data) == [1.0, 2.0]
    assert list(da.coords["ids"].data) == ["a", "b"]
    assert list(da.coords["equipment_ids"].data) == ["eq1", "eq2"]


def test_eqarray_to_xarray_empty():
    """T2 — empty arrays."""
    eq = EquipmentNodeArray(ids=[], equipment_ids=[], values=[], units="kW")
    da = eqarray_to_xarray(eq)
    assert da.shape == (0,)
    assert list(da.coords["ids"].data) == []
    assert list(da.coords["equipment_ids"].data) == []


def test_eqarray_to_xarray_mismatched():
    """T3 — mismatched ids/values raises an exception (xarray raises at conversion time)."""
    eq = EquipmentNodeArray(ids=["a", "b"], equipment_ids=["eq1", "eq2"], values=[1.0], units="kW")
    with pytest.raises(Exception):
        eqarray_to_xarray(eq)


# ---------------------------------------------------------------------------
# measurement_to_xarray
# ---------------------------------------------------------------------------

def test_measurement_to_xarray_normal():
    """T4 — normal case."""
    ma = MeasurementArray(ids=["x", "y"], values=[3.0, 4.0], units="kW")
    da = measurement_to_xarray(ma)
    assert list(da.data) == [3.0, 4.0]
    assert list(da.coords["ids"].data) == ["x", "y"]


def test_measurement_to_xarray_empty():
    """T5 — empty arrays."""
    ma = MeasurementArray(ids=[], values=[], units="kW")
    da = measurement_to_xarray(ma)
    assert da.shape == (0,)


def test_measurement_to_xarray_mismatched():
    """T6 — mismatched ids/values raises an exception (xarray raises at conversion time)."""
    ma = MeasurementArray(ids=["x"], values=[1.0, 2.0], units="kW")
    with pytest.raises(Exception):
        measurement_to_xarray(ma)


# ---------------------------------------------------------------------------
# xarray_to_dict
# ---------------------------------------------------------------------------

def test_xarray_to_dict_normal():
    """T7 — normal case."""
    da = xr.DataArray([10.0, 20.0], coords={"ids": ["a", "b"]}, dims=("ids",))
    result = xarray_to_dict(da)
    assert result["values"] == [10.0, 20.0]
    assert result["ids"] == ["a", "b"]


def test_xarray_to_dict_multiple_coords():
    """T8 — multiple coordinates all appear as top-level keys."""
    da = xr.DataArray(
        [1.0, 2.0],
        dims=("ids",),
        coords={
            "ids": ["a", "b"],
            "equipment_ids": ("ids", ["eq1", "eq2"]),
        },
    )
    result = xarray_to_dict(da)
    assert "ids" in result
    assert "equipment_ids" in result
    assert result["ids"] == ["a", "b"]
    assert result["equipment_ids"] == ["eq1", "eq2"]


def test_xarray_to_dict_empty():
    """T9 — empty DataArray."""
    da = xr.DataArray([], coords={"ids": []}, dims=("ids",))
    result = xarray_to_dict(da)
    assert result["values"] == []
    assert result["ids"] == []


# ---------------------------------------------------------------------------
# xarray_to_eqarray
# ---------------------------------------------------------------------------

def test_xarray_to_eqarray_identical_to_dict():
    """T10 — xarray_to_eqarray produces identical output to xarray_to_dict."""
    da = xr.DataArray([10.0, 20.0], coords={"ids": ["a", "b"]}, dims=("ids",))
    assert xarray_to_eqarray(da) == xarray_to_dict(da)


# ---------------------------------------------------------------------------
# xarray_to_powers_cart
# ---------------------------------------------------------------------------

def _eq_da(values, ids, equipment_ids=None):
    """Helper to build a DataArray with optional equipment_ids coord."""
    if equipment_ids is None:
        equipment_ids = [f"eq{i}" for i in range(len(ids))]
    return xr.DataArray(
        np.array(values),
        dims=("ids",),
        coords={
            "ids": ids,
            "equipment_ids": ("ids", equipment_ids),
        },
    )


def test_xarray_to_powers_cart_complex():
    """T11 — complex input."""
    da = _eq_da([1 + 2j, 3 + 4j], ["a", "b"], ["eq1", "eq2"])
    real, imag = xarray_to_powers_cart(da)
    assert isinstance(real, PowersReal)
    assert isinstance(imag, PowersImaginary)
    assert real.values == [1.0, 3.0]
    assert real.ids == ["a", "b"]
    assert imag.values == [2.0, 4.0]
    assert imag.ids == ["a", "b"]


def test_xarray_to_powers_cart_real_produces_zero_imag():
    """T12 — purely real input produces zero imaginary values."""
    da = _eq_da([5.0, 6.0], ["a", "b"])
    _, imag = xarray_to_powers_cart(da)
    assert imag.values == [0.0, 0.0]


def test_xarray_to_powers_cart_kwargs_forwarded():
    """T13 — kwargs are forwarded to both PowersReal and PowersImaginary."""
    da = _eq_da([1.0 + 2.0j], ["a"], ["eq1"])
    real, imag = xarray_to_powers_cart(da, time=42)
    # time=42 is coerced by pydantic to a datetime; verify it was forwarded to both
    assert real.time is not None
    assert real.time == imag.time


def test_xarray_to_powers_cart_empty():
    """T14 — empty DataArray."""
    da = xr.DataArray(
        np.array([], dtype=complex),
        dims=("ids",),
        coords={
            "ids": [],
            "equipment_ids": ("ids", []),
        },
    )
    real, imag = xarray_to_powers_cart(da)
    assert real.values == []
    assert imag.values == []


# ---------------------------------------------------------------------------
# HubFederate — publish_real / publish_imag (mocked HELICS)
# ---------------------------------------------------------------------------

def _make_mock_sub(is_updated=False, json_val=None):
    """Build a mock HELICS subscription."""
    sub = MagicMock()
    sub.is_updated.return_value = is_updated
    if json_val is not None:
        sub.json = json_val  # HELICS .json returns a dict; parse_obj expects a dict
    return sub


def _powers_real_dict(ids, equipment_ids, values, time=0):
    """Build a PowersReal and return it as a plain dict (mimicking HELICS .json)."""
    p = PowersReal(ids=ids, equipment_ids=equipment_ids, values=values, time=time)
    return json.loads(p.model_dump_json())


def _powers_imag_dict(ids, equipment_ids, values, time=0):
    """Build a PowersImaginary and return it as a plain dict."""
    q = PowersImaginary(ids=ids, equipment_ids=equipment_ids, values=values, time=time)
    return json.loads(q.model_dump_json())


def _make_hub():
    """Construct a HubFederate with __init__ bypassed — no HELICS calls needed."""
    from hub_federate import HubFederate, StaticConfig, Subscriptions

    hub = HubFederate.__new__(HubFederate)

    hub.static = StaticConfig()
    hub.static.name = "test_hub"
    hub.static.max_itr = 5
    hub.static.t_steps = 10

    hub.pub_area_p = [MagicMock() for _ in range(6)]
    hub.pub_area_q = [MagicMock() for _ in range(6)]

    hub.sub = Subscriptions()
    for attr in ("p0", "p1", "p2", "p3", "p4", "q0", "q1", "q2", "q3", "q4"):
        setattr(hub.sub, attr, _make_mock_sub())

    return hub


def test_publish_real_no_updates():
    """T15 — no subscriptions updated → publishes empty to all 6 areas."""
    hub = _make_hub()
    hub.publish_real()

    for pub in hub.pub_area_p:
        pub.publish.assert_called_once()
        payload = json.loads(pub.publish.call_args[0][0])
        assert payload["values"] == []
        assert payload["ids"] == []
        assert payload["equipment_ids"] == []


def test_publish_real_one_subscription_updated():
    """T16 — one subscription updated → values published to all 6 areas."""
    hub = _make_hub()
    hub.sub.p0 = _make_mock_sub(
        is_updated=True,
        json_val=_powers_real_dict(
            ids=["bus1", "bus2", "bus3"],
            equipment_ids=["eq1", "eq2", "eq3"],
            values=[10.0, 20.0, 30.0],
        ),
    )

    hub.publish_real()

    for pub in hub.pub_area_p:
        pub.publish.assert_called_once()
        payload = json.loads(pub.publish.call_args[0][0])
        assert payload["values"] == [10.0, 20.0, 30.0]
        assert payload["ids"] == ["bus1", "bus2", "bus3"]


def test_publish_real_multiple_subscriptions_updated():
    """T17 — multiple subscriptions updated → values concatenated (p0 first, then p2)."""
    hub = _make_hub()
    hub.sub.p0 = _make_mock_sub(
        is_updated=True,
        json_val=_powers_real_dict(ids=["bus1"], equipment_ids=["eq1"], values=[1.0]),
    )
    hub.sub.p2 = _make_mock_sub(
        is_updated=True,
        json_val=_powers_real_dict(ids=["bus2"], equipment_ids=["eq2"], values=[2.0]),
    )

    hub.publish_real()

    for pub in hub.pub_area_p:
        pub.publish.assert_called_once()
        payload = json.loads(pub.publish.call_args[0][0])
        assert payload["ids"] == ["bus1", "bus2"]
        assert payload["values"] == [1.0, 2.0]


def test_publish_imag_no_updates():
    """T18 — no subscriptions updated → publishes empty to all 6 areas."""
    hub = _make_hub()
    hub.publish_imag()

    for pub in hub.pub_area_q:
        pub.publish.assert_called_once()
        payload = json.loads(pub.publish.call_args[0][0])
        assert payload["values"] == []
        assert payload["ids"] == []
        assert payload["equipment_ids"] == []


def test_publish_imag_one_subscription_updated():
    """T19 — one subscription updated → values published to all 6 areas."""
    hub = _make_hub()
    hub.sub.q0 = _make_mock_sub(
        is_updated=True,
        json_val=_powers_imag_dict(
            ids=["bus1", "bus2", "bus3"],
            equipment_ids=["eq1", "eq2", "eq3"],
            values=[5.0, 6.0, 7.0],
        ),
    )

    hub.publish_imag()

    for pub in hub.pub_area_q:
        pub.publish.assert_called_once()
        payload = json.loads(pub.publish.call_args[0][0])
        assert payload["values"] == [5.0, 6.0, 7.0]
        assert payload["ids"] == ["bus1", "bus2", "bus3"]


def test_publish_imag_multiple_subscriptions_updated():
    """T20 — multiple subscriptions updated → values concatenated (q0 first, then q2)."""
    hub = _make_hub()
    hub.sub.q0 = _make_mock_sub(
        is_updated=True,
        json_val=_powers_imag_dict(ids=["bus1"], equipment_ids=["eq1"], values=[3.0]),
    )
    hub.sub.q2 = _make_mock_sub(
        is_updated=True,
        json_val=_powers_imag_dict(ids=["bus2"], equipment_ids=["eq2"], values=[4.0]),
    )

    hub.publish_imag()

    for pub in hub.pub_area_q:
        pub.publish.assert_called_once()
        payload = json.loads(pub.publish.call_args[0][0])
        assert payload["ids"] == ["bus1", "bus2"]
        assert payload["values"] == [3.0, 4.0]


def test_register_subscription_partial():
    """T21 — partial input mapping registers only mapped subscriptions."""
    from hub_federate import HubFederate, Subscriptions, StaticConfig
    from oedisi.types.common import BrokerConfig

    # Mock helics call to value federate
    fed_mock = MagicMock()
    broker = BrokerConfig(broker_ip="127.0.0.1", broker_port=23404)

    original_init = HubFederate.__init__
    def patch_init(self, broker_config):
        self.inputs = {
            "sub_p0": "feeder0/pub_p0",
            # sub_p1, sub_p2... and q0, q1... are missing/empty
            "sub_q2": "feeder2/pub_q2",
            "sub_p3": "",
        }
        self.sub = Subscriptions()
        self.static = StaticConfig()
        self.static.name = "test_hub"
        self.static.max_itr = 5
        self.static.t_steps = 10
        self.fed = fed_mock
        self.register_subscription()

    HubFederate.__init__ = patch_init
    try:
        fed = HubFederate(broker)
        assert fed_mock.register_subscription.call_count == 2
        fed_mock.register_subscription.assert_any_call("feeder0/pub_p0", "")
        fed_mock.register_subscription.assert_any_call("feeder2/pub_q2", "")
        assert fed.sub.p0 is not None
        assert fed.sub.p1 is None
        assert fed.sub.q2 is not None
        assert fed.sub.q3 is None
    finally:
        HubFederate.__init__ = original_init
