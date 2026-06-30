"""Core inference engine for TBJU deployment app."""

from .tbju_rknn_core import (
    ModelConfig,
    DetectionRow,
    FrameResult,
    TBJURKNNEngine,
    ResultWriter,
    validate_debris_region,
    TemporalConsistencyFilter,
    DEBRIS_REGION_NAMES,
    resolve_class_ids,
    resolve_debris_ids,
)
