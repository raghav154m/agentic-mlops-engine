from langgraph.graph import StateGraph, START, END
from workflow.state import MLOpsState
from profiler.dataset_profiler import DatasetProfiler
from agents.strategy_agent import StrategyAgent
from agents.code_generator import CodeGeneratorAgent
from agents.debugger_agent import DebuggerAgent
from runner.sandbox import execute_script


# --- Node Definitions ---

def profiler_node(state: MLOpsState) -> dict:
    print("\n🔍 [Node 1: Profiler] Analyzing dataset statistics...")
    profiler = DatasetProfiler(state["dataset_path"])
    profile = profiler.generate_profile()
    return {"profile": profile}


def strategy_node(state: MLOpsState) -> dict:
    print("\n🧠 [Node 2: Strategy Agent] Formulating data transformation plan...")
    agent = StrategyAgent()
    strategy = agent.generate_strategy(
        dataset_profile=state["profile"],
        target_column=state["target_column"]
    )
    return {"strategy": strategy}


def coder_node(state: MLOpsState) -> dict:
    print("\n💻 [Node 3: Coder Agent] Synthesizing initial ML pipeline script...")
    coder = CodeGeneratorAgent()
    code = coder.generate_code(
        dataset_path=state["dataset_path"],
        strategy=state["strategy"]
    )
    script_path = coder.save_script(code)
    return {"generated_code": code, "script_path": script_path}


def sandbox_node(state: MLOpsState) -> dict:
    print(f"\n⚙️ [Node 4: Sandbox Runner] Executing script (Attempt {state.get('retry_count', 0) + 1})...")
    script_path = state.get("script_path", "sandbox_workspace/generated_pipeline.py")
    success, stdout, stderr = execute_script(script_path)

    if success:
        print("✅ Pipeline executed successfully in sandbox!")
        print("\n--- Sandbox STDOUT Output ---")
        print(stdout)
    else:
        print("❌ Runtime error intercepted by sandbox runner:")
        print(stderr)

    return {
        "execution_status": success,
        "stdout": stdout,
        "stderr": stderr,
        "error_logs": stderr if not success else None
    }


def debugger_node(state: MLOpsState) -> dict:
    current_retry = state.get("retry_count", 0) + 1
    print(f"\n🛠️ [Node 5: Self-Healing Debugger] Diagnosing error & patching code (Retry #{current_retry})...")
    
    debugger = DebuggerAgent()
    fixed_code = debugger.fix_code(
        broken_code=state["generated_code"],
        error_logs=state["error_logs"]
    )
    script_path = debugger.save_script(fixed_code)
    
    return {
        "generated_code": fixed_code,
        "script_path": script_path,
        "retry_count": current_retry
    }


# --- Conditional Routing Logic ---

def route_after_sandbox(state: MLOpsState) -> str:
    """
    Decides whether to conclude workflow or trigger self-healing loop.
    """
    if state.get("execution_status"):
        return END
    
    if state.get("retry_count", 0) >= 3:
        print("\n⚠️ Maximum retry limit (3) reached. Terminating self-healing loop.")
        return END
        
    return "debugger_node"


# --- Build StateGraph ---

def build_mlops_graph():
    builder = StateGraph(MLOpsState)

    # Register All Nodes
    builder.add_node("profiler_node", profiler_node)
    builder.add_node("strategy_node", strategy_node)
    builder.add_node("coder_node", coder_node)
    builder.add_node("sandbox_node", sandbox_node)
    builder.add_node("debugger_node", debugger_node)

    # Base Flow
    builder.add_edge(START, "profiler_node")
    builder.add_edge("profiler_node", "strategy_node")
    builder.add_edge("strategy_node", "coder_node")
    builder.add_edge("coder_node", "sandbox_node")

    # Dynamic Self-Healing Loop
    builder.add_conditional_edges(
        "sandbox_node",
        route_after_sandbox,
        {
            "debugger_node": "debugger_node",
            END: END
        }
    )
    # Loop back to sandbox runner after fixing
    builder.add_edge("debugger_node", "sandbox_node")

    return builder.compile()


if __name__ == "__main__":
    app = build_mlops_graph()

    initial_input = {
        "dataset_path": "data/sample_dataset.csv",
        "target_column": "churn",
        "retry_count": 0
    }

    print("🚀 Starting Self-Healing MLOps StateGraph Workflow...")
    final_output = app.invoke(initial_input)

    print("\n" + "=" * 45)
    print("🏁 LangGraph Workflow Execution Summary")
    print("=" * 45)
    print(f"Final Status: {'SUCCESS ✅' if final_output.get('execution_status') else 'FAILED ❌'}")
    print(f"Total Retries Used: {final_output.get('retry_count', 0)}")
    print(f"Script Location: {final_output.get('script_path')}")