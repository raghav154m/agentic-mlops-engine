import os
import sys
import subprocess
from typing import Tuple


def execute_script(script_path: str, timeout: int = 45) -> Tuple[bool, str, str]:
    """
    Executes a generated Python script in an isolated subprocess sandbox.
    Includes guardrails against placeholders like '...' or empty scripts.
    """
    if not os.path.exists(script_path):
        return False, "", f"Script not found at path: {script_path}"

    with open(script_path, "r", encoding="utf-8") as f:
        code_content = f.read().strip()

    # Guardrail: Catch lazy ellipses or incomplete generation
    if code_content in ("...", "…") or len(code_content.splitlines()) < 8:
        return False, "", "ValidationError: Generated code is incomplete or contains only placeholders ('...')."

    python_executable = sys.executable

    try:
        process = subprocess.run(
            [python_executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        stdout = process.stdout
        stderr = process.stderr
        success = (process.returncode == 0)

        return success, stdout, stderr

    except subprocess.TimeoutExpired:
        return False, "", f"Execution timed out after {timeout} seconds."
    except Exception as e:
        return False, "", str(e)