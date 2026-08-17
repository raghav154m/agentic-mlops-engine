import sys
import subprocess
from pathlib import Path
from typing import Tuple


def execute_script(script_path: str, timeout: int = 60) -> Tuple[bool, str, str]:
    """
    Executes a Python script in an isolated subprocess with a timeout.
    
    Args:
        script_path (str): Relative or absolute path to the target Python script.
        timeout (int): Maximum time in seconds before terminating execution.

    Returns:
        Tuple[bool, str, str]: (success_boolean, stdout_logs, stderr_logs)
    """
    resolved_path = Path(script_path).resolve()
    
    if not resolved_path.exists():
        return False, "", f"FileNotFoundError: Script '{resolved_path}' does not exist."

    # Use current active python interpreter
    python_executable = sys.executable

    try:
        process = subprocess.run(
            [python_executable, str(resolved_path)],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        stdout = process.stdout.strip()
        stderr = process.stderr.strip()
        success = (process.returncode == 0)

        return success, stdout, stderr

    except subprocess.TimeoutExpired as exc:
        return False, "", f"ExecutionTimedOut: Script exceeded {timeout} seconds limit."
    except Exception as exc:
        return False, "", f"SandboxRunnerError: {str(exc)}"


class SandboxRunner:
    """Wrapper class for object-oriented interfaces."""
    @staticmethod
    def run(script_path: str, timeout: int = 60) -> Tuple[bool, str, str]:
        return execute_script(script_path, timeout)