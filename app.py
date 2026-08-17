import os
import streamlit as st
import pandas as pd
from workflow.graph import build_mlops_graph

st.set_page_config(
    page_title="Autonomous Agentic MLOps Engine",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Autonomous Agentic MLOps Engine")
st.markdown(
    "An autonomous MLOps platform that profiles data, formulates transformation strategies, "
    "writes machine learning code, and self-heals runtime errors inside an isolated execution sandbox."
)

st.divider()

# Left Sidebar: Inputs & Configuration
st.sidebar.header("📁 Dataset Configuration")
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    os.makedirs("data", exist_ok=True)
    temp_path = os.path.join("data", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    df = pd.read_csv(temp_path)
    
    st.sidebar.subheader("🎯 Target Selection")
    target_column = st.sidebar.selectbox("Select Target Column", options=list(df.columns))

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2 = st.columns([1, 4])
    with col1:
        run_btn = st.button("🚀 Run Autonomous Pipeline", type="primary", use_container_width=True)

    if run_btn:
        st.divider()
        st.subheader("⚙️ Autonomous State Machine Execution")
        
        status_container = st.status("Initializing workflow graph...", expanded=True)
        
        with status_container:
            st.write("🔍 **Node 1: Profiler** — Extracting dataset statistics and missingness ratios...")
            st.write("🧠 **Node 2: Strategy Agent** — Formulating preprocessing and modeling plan...")
            st.write("💻 **Node 3: Coder Agent** — Synthesizing modular Python pipeline code...")
            st.write("⚙️ **Node 4: Sandbox Runner** — Executing pipeline in isolated subprocess...")
            st.write("🛠️ **Node 5: Self-Healing Debugger** — Active for runtime interception & auto-repair...")
            st.write("📦 **Node 6: Exporter** — Packaging model code and executive PDF report...")
            
            # Execute LangGraph Workflow
            app = build_mlops_graph()
            initial_state = {
                "dataset_path": temp_path,
                "target_column": target_column,
                "retry_count": 0
            }
            final_output = app.invoke(initial_state)

        if final_output.get("execution_status"):
            st.success("🎉 Pipeline executed and verified successfully!")
            
            tab1, tab2, tab3 = st.tabs(["📄 Generated Pipeline Script", "📈 Execution Output", "📦 Download Deliverables"])
            
            with tab1:
                st.code(final_output.get("generated_code", ""), language="python")
                
            with tab2:
                st.text(final_output.get("stdout", "No output recorded."))
                
            with tab3:
                c1, c2 = st.columns(2)
                if os.path.exists("artifacts/pipeline.py"):
                    with open("artifacts/pipeline.py", "rb") as f:
                        c1.download_button(
                            label="⬇️ Download pipeline.py",
                            data=f,
                            file_name="pipeline.py",
                            mime="text/x-python",
                            use_container_width=True
                        )
                if os.path.exists("artifacts/summary_report.pdf"):
                    with open("artifacts/summary_report.pdf", "rb") as f:
                        c2.download_button(
                            label="⬇️ Download Executive Report (PDF)",
                            data=f,
                            file_name="summary_report.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
        else:
            st.error("❌ Pipeline failed after maximum retry attempts.")
            if final_output.get("stderr"):
                st.code(final_output.get("stderr"), language="bash")
else:
    st.info("👈 Upload a CSV file in the sidebar to begin.")