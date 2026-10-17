"""Building blocks shared by spec sections: fixed-or-range parameters and the section base class."""

from __future__ import annotations

from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, field_validator

Range = tuple[float, float]
Param = Optional[Union[float, Range]]


class Section(BaseModel):
    """Base for spec sections: rejects unknown keys and checks ranges."""

    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="after")
    @classmethod
    def _check_range(cls, v):
        """Reject ranges whose low value exceeds the high value."""
        if isinstance(v, tuple) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
            if v[0] > v[1]:
                raise ValueError(f"range low value {v[0]} exceeds high value {v[1]}")
        return v
