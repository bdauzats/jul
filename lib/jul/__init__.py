"""jul — Just a LLM: local typed decisions, with the interface of the TypeSafe (Jev) Python SDK.

Swap the import and existing code keeps working:

    # from typesafe_sdk import TypeSafeClient, Choice, Noul, Score
    from jul import TypeSafeClient, Choice, Noul, Score

Two extras Jev does not have: a `Context` describing the data to sort, and `client.autotune(...)`, a small
head trained in seconds on labeled examples while the model itself stays untouched.
"""

from .client import AsyncTypeSafeClient, TypeSafeClient
from .context import Context
from .presets import PRESETS, Preset
from .tuning import TuningReport
from .types import (Choice, ChoiceAnswer, Noul, NoulAnswer, NoulCriteria, Score, ScoreAnswer,
                    SystemOneResponse, Usage)

__version__ = "0.1.0"

__all__ = [
    "TypeSafeClient", "AsyncTypeSafeClient",
    "Choice", "Noul", "NoulCriteria", "Score",
    "ChoiceAnswer", "NoulAnswer", "ScoreAnswer", "SystemOneResponse", "Usage",
    "Context", "TuningReport", "Preset", "PRESETS", "__version__",
]
