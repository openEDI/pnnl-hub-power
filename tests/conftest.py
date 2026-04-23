import sys
import os
from unittest.mock import MagicMock

# Add the source directory to sys.path so hub_federate and server are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "pnnl-hub-power"))

# Mock helics and uvicorn before any test module imports them
_helics_mock = MagicMock()
_helics_mock.HELICS_CORE_TYPE_ZMQ = 0
_helics_mock.HELICS_DATA_TYPE_STRING = 0
_helics_mock.HELICS_PROPERTY_TIME_PERIOD = 0

sys.modules.setdefault("helics", _helics_mock)
sys.modules.setdefault("uvicorn", MagicMock())
