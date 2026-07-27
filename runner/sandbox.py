import subprocess
import sys
from pathlib import Path
from typing import Tuple


class SandboxRunner:
    """Handles controlled, isolated execution of dynamically generated Python scripts."""

    def __init__(
        self,
        workspace_dir: str = "sandbox_workspace",
        timeout_seconds: int = 60,
    ):
        """
        Args:
            workspace_dir (str): Directory where generated scripts execute.
            timeout_seconds (int): Maximum allowed execution time before process termination.
        """
        self.workspace_dir = Path(workspace_dir).resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds

    def execute_script(self, script_path: str) -> Tuple[bool, str, str]:
        """Executes a Python script in an isolated subprocess.

        Args:
            script_path (str): File path to the Python script to execute.

        Returns:
            Tuple[bool, str, str]: (Success status, stdout logs, stderr logs)
        """
        # Resolve to absolute path to prevent duplicate directory issues
        path = Path(script_path).resolve()

        if not path.is_file():
            return False, "", f"Error: Script file '{script_path}' not found."

        try:
            # Execute script in isolated subprocess using absolute path
            result = subprocess.run(
                [sys.executable, str(path)],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=str(self.workspace_dir),
            )

            # Exit code 0 indicates success
            is_success = result.returncode == 0
            return is_success, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            return (
                False,
                "",
                f"ExecutionTimedOut: Script exceeded timeout limit of {self.timeout_seconds} seconds.",
            )
        except Exception as e:
            return False, "", f"ExecutionSystemError: {str(e)}"