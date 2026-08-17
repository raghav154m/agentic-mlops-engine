from typing import TypedDict, Optional, Dict, Any


class MLOpsState(TypedDict):
    """
    Centralized memory state tracking the lifecycle of the MLOps pipeline.
    """
    dataset_path: str
    target_column: str
    profile: Optional[Dict[str, Any]]
    strategy: Optional[Dict[str, Any]]
    generated_code: Optional[str]
    script_path: Optional[str]
    execution_status: Optional[bool]
    stdout: Optional[str]
    stderr: Optional[str]
    retry_count: int
    error_logs: Optional[str]