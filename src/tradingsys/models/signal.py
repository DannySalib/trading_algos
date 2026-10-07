
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, model_validator
from inspect import signature

from .. import signal_functions

class Signal(BaseModel):
    name: str
    function_name: str
    data_need: str
    function_args: dict[str, Any]
    lookback_days: int
    data_need: str 

    @model_validator(mode="after")
    def validate_function(self) -> "Signal":
        if not hasattr(signal_functions, self.function_name):
            raise ValueError(
                f"Signal function '{self.function_name}' not found in signal_functions"
            )

        func = getattr(signal_functions, self.function_name)

        if not callable(func):
            raise ValueError(
                f"'{self.function_name}' exists in signal_functions but is not callable"
            )

        expected = signature(func).parameters
        provided = self.function_args

        unexpected = provided.keys() - expected.keys()

        if unexpected:
            raise ValueError(
                f"Unexpected arguments for '{self.function_name}': "
                f"{sorted(unexpected)}"
            )

        return self