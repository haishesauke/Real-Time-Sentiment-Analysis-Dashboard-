import os
import pytest
from src.inference import SentimentModel

def test_model_missing_raises():
    # When no trained model exists yet, inference should raise.
    with pytest.raises(RuntimeError):
        SentimentModel().predict(["hello world"])  # no model until trained
