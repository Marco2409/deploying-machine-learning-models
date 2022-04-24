from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from classification_model.config.core import config


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    validated_data = input_data[config.model_config.features].copy()
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleTitanicDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class TitanicDataInputSchema(BaseModel):
    pclass: Optional[int]
    survived: Optional[int]
    name: Optional[str]
    sex: Optional[str]
    age: Optional[str]
    sibsp: Optional[int]
    parch: Optional[int]
    ticket: Optional[str]
    fare: Optional[str]
    cabin: Optional[str]
    embarked: Optional[str]
    boat: Optional[str]
    body: Optional[str]
    # home.dest: Optional[str]


class MultipleTitanicDataInputs(BaseModel):
    inputs: List[TitanicDataInputSchema]
