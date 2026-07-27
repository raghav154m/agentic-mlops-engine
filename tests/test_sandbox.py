from pathlib import Path
from runner.sandbox import SandboxRunner


def test_sandbox_execution():
    runner = SandboxRunner(workspace_dir="sandbox_workspace")
    workspace = Path("sandbox_workspace")
    workspace.mkdir(exist_ok=True)

    print("==========================================")
    print(" Running Sandbox Execution Verification   ")
    print("==========================================")

    # Test Case 1: Valid Execution
    valid_script = workspace / "test_valid.py"
    valid_script.write_text("import pandas as pd\nprint('Pandas loaded successfully!')")

    success, stdout, stderr = runner.execute_script(str(valid_script))
    print("\n--- Test 1: Valid Script Execution ---")
    print(f"Status Success : {success}")
    print(f"Stdout Output  : {stdout.strip()}")
    print(f"Stderr Output  : {stderr.strip()}")

    # Test Case 2: Intentional Runtime Crash (KeyError)
    failing_script = workspace / "test_error.py"
    failing_script.write_text(
        "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3]})\nprint(df['missing_column'])"
    )

    success, stdout, stderr = runner.execute_script(str(failing_script))
    print("\n--- Test 2: Error Interception Execution ---")
    print(f"Status Success : {success}")
    print(f"Stdout Output  : {stdout.strip()}")
    print(f"Captured Stderr Stack Trace:\n{stderr.strip()}")
    print("==========================================\n")


if __name__ == "__main__":
    test_sandbox_execution()