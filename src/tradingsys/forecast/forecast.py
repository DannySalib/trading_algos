from __future__ import annotations 

from typing import Callable
from ..models.signal import Signal
from .. import signal_functions

import pandas as pd 

SignalFunction = Callable[
    [pd.DataFrame | pd.Series],
    pd.DataFrame | pd.Series,
]

SignalFunctionFactory = Callable[..., SignalFunction]

class Forecast:
    def __init__(self, signal: Signal):
        self._name = signal.name
        
        # safe b/c of pydantic validations ahead of time
        function_getter: SignalFunctionFactory = getattr(signal_functions, signal.function_name)

        self._signal_function: SignalFunction = function_getter(
            lookback_days=signal.lookback_days,
            **signal.function_args # this is also safe with model validation 
        )

        self._data_need = signal.data_need

        self._data: pd.DataFrame | None = None

    @property 
    def name(self): return self._name 
        
    @property
    def signal_function(self): return self._signal_function

    @property
    def data_need(self): return self._data_need

    @property
    def data(self):
        if self._data is None:
            raise RuntimeError(f"Asked for forecast {self._name} data but is None")
        return self._data
    

    def apply(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        # assumes correct data is provided
        # based on `data_need` property 
        self._data = self._signal_function(data)


    def __str__(self):
        return f'Forecast name: {self._name} needs {self._data_need}'
