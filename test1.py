"""
Streamlit App for CrewAI Mathematical Proof System
Displays agent conversations in real-time and shows results with expandable sections
"""

import streamlit as st
import sys
import io
import re
import json
from datetime import datetime
from contextlib import redirect_stdout
from threading import Thread
import time
import queue
from typing import Optional

# Import the proof system from main.py
from main import ProofSystem

# Page configuration
st.set_page_config(
    page_title="MathMind: AI Mathematical Proof System",
    page_icon="🧮",
    layout="wide",

    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-theorem {
        color: #28a745;
        font-weight: bold;
    }
    .status-conjecture {
        color: #ffc107;
        font-weight: bold;
    }
    .agent-message {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        font-family: monospace;
        font-size: 0.9rem;
        white-space: pre-wrap;
        max-height: 400px;
        overflow-y: auto;
    }
    .summary-card {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .progress-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'conversation_log' not in st.session_state:
    st.session_state.conversation_log = []
if 'current_phase' not in st.session_state:
    st.session_state.current_phase = ""
if 'progress' not in st.session_state:
    st.session_state.progress = 0.0

# Header
st.markdown('<div class="main-header">🧮 MathMind: AI Mathematical Proof System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Collaborative AI Agents for Mathematical Conjecture Generation & Verification</div>', unsafe_allow_html=True)

# Sidebar - Input Section
with st.sidebar:
    st.header("")

    st.markdown("### 📝 Mathematical Query")
    user_query = st.text_area(
        "Enter your mathematical question:",
        height=150,
        placeholder="e.g., distribution of zeros of Ramanujan tau(n) modulo small primes",
        help="Ask a question about number theory or algebraic topology"
    )

    st.markdown("### 🎯 Agent Type")
    agent_type = st.radio(
        "Select mathematical domain:",
        options=["number_theory", "algebraic_topology"],
        format_func=lambda x: "📐 Number Theory" if x == "number_theory" else "🔷 Algebraic Topology",
        help="Choose the type of mathematical agents to use"
    )

    st.markdown("### 🔄 Verification Settings")
    max_iterations = st.slider(
        "Maximum verification iterations:",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of times the verifier and experimenter will refine the proof"
    )

    st.markdown("---")

    # Submit button
    submit_button = st.button(
        "🚀 Start Proof Generation",
        type="primary",
        disabled=st.session_state.processing or not user_query,
        use_container_width=True
    )

    # Example queries
    with st.expander("💡 Example Queries"):
        st.markdown("**Number Theory:**")
        st.code("distribution of zeros of Ramanujan tau(n) modulo small primes", language=None)
        st.code("density of primes p where tau(p) ≡ 0 (mod p)", language=None)

        st.markdown("**Algebraic Topology:**")
        st.code("homology groups of product spaces", language=None)
        st.code("homotopy groups of spheres", language=None)


class StreamCapture:
    """Captures stdout in real-time and updates Streamlit using a queue"""
    def __init__(self, message_queue: Optional[queue.Queue] = None):
        self.buffer = io.StringIO()
        self.log = []
        self.message_queue = message_queue

    def write(self, text):
        if text.strip():
            self.log.append(text)
            # Add to session state
            if 'conversation_log' in st.session_state:
                st.session_state.conversation_log.append(text)
            # Add to queue if available
            if self.message_queue:
                try:
                    self.message_queue.put_nowait(text)
                except queue.Full:
                    pass
        return len(text)

    def flush(self):
        pass

    def get_log(self):
        return "".join(self.log)


def update_progress(phase: str, progress: float):
    """Update progress tracking"""
    st.session_state.current_phase = phase
    st.session_state.progress = progress


def run_proof_system(query: str, agent_type: str, max_iter: int, message_queue: queue.Queue):
    """Run the proof system and capture output"""
    try:
        # Initialize proof system
        update_progress("Initializing agents...", 0.1)
        message_queue.put(("progress", "Initializing agents...", 0.1))

        proof_system = ProofSystem(max_iterations=max_iter, agent_type=agent_type)

        # Capture stdout
        stream_capture = StreamCapture(message_queue=message_queue)

        update_progress("Generating conjectures...", 0.2)
        message_queue.put(("progress", "Generating conjectures...", 0.2))

        # Run the proof system with captured output
        with redirect_stdout(stream_capture):
            results = proof_system.process_query(query)

        update_progress("Complete!", 1.0)
        message_queue.put(("progress", "Complete!", 1.0))
        message_queue.put(("results", results))

        st.session_state.results = results
        st.session_state.processing = False

    except Exception as e:
        st.session_state.processing = False
        message_queue.put(("error", str(e)))
        import traceback
        error_trace = traceback.format_exc()
        message_queue.put(("error_trace", error_trace))


# Initialize message queue in session state
if 'message_queue' not in st.session_state:
    st.session_state.message_queue = None

# Main content area
if submit_button and user_query:
    # Reset state
    st.session_state.processing = True
    st.session_state.conversation_log = []
    st.session_state.results = None
    st.session_state.current_phase = "Starting..."
    st.session_state.progress = 0.0
    st.session_state.message_queue = queue.Queue(maxsize=1000)

    # Start processing in a thread
    thread = Thread(
        target=run_proof_system,
        args=(user_query, agent_type, max_iterations, st.session_state.message_queue),
        daemon=True
    )
    thread.start()

    # Trigger rerun to show processing UI
    st.rerun()

# Show processing status
if st.session_state.processing:
    st.markdown('<div class="progress-section">', unsafe_allow_html=True)
    st.subheader("⚡ Processing")

    # Progress bar
    progress_bar = st.progress(st.session_state.progress)
    status_text = st.empty()
    status_text.text(st.session_state.current_phase)

    # Poll the message queue for updates
    if st.session_state.message_queue:
        updates_received = False
        try:
            while True:
                try:
                    message = st.session_state.message_queue.get_nowait()
                    updates_received = True

                    if message[0] == "progress":
                        _, phase, progress = message
                        st.session_state.current_phase = phase
                        st.session_state.progress = progress
                    elif message[0] == "results":
                        st.session_state.results = message[1]
                        st.session_state.processing = False
                    elif message[0] == "error":
                        st.error(f"Error: {message[1]}")
                        st.session_state.processing = False
                    elif message[0] == "error_trace":
                        st.code(message[1])
                except queue.Empty:
                    break
        except Exception as e:
            st.warning(f"Queue error: {e}")

    # Real-time conversation display
    st.markdown("### 💬 Agent Conversations")
    conversation_container = st.container()

    with conversation_container:
        if st.session_state.conversation_log:
            # Show last 100 lines to avoid overwhelming the UI
            recent_log = st.session_state.conversation_log[-100:]
            conversation_text = "".join(recent_log)
            st.markdown(f'<div class="agent-message">{conversation_text}</div>', unsafe_allow_html=True)
        else:
            st.info("Waiting for agents to start conversing...")

    st.markdown('</div>', unsafe_allow_html=True)

    # Auto-refresh while processing (every 1 second)
    if st.session_state.processing:
        time.sleep(1)
        st.rerun()

# Display results
if st.session_state.results and not st.session_state.processing:
    results = st.session_state.results

    st.success("✅ Processing Complete!")

    # Summary Card
    st.markdown("## 📊 Summary")
    summary = results.get("summary", {})

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Conjectures", summary.get("total_conjectures", 0))
    with col2:
        st.metric("✓ Theorems Proven", summary.get("theorems_proven", 0))
    with col3:
        st.metric("⚠️ Conjectures Remaining", summary.get("conjectures_remaining", 0))

    # Display conjectures with expandable sections
    st.markdown("## 🔍 Detailed Results")

    theorems = results.get("theorems", [])
    remaining_conjectures = results.get("remaining_conjectures", [])

    # Theorems section
    if theorems:
        st.markdown("### ✅ Proven Theorems")
        for idx, theorem in enumerate(theorems, 1):
            with st.expander(f"**Theorem {idx}** - {theorem.get('LaTeX', 'No title')[:100]}...", expanded=False):
                st.markdown("#### Statement")
                latex_statement = theorem.get("LaTeX", "")
                if latex_statement:
                    try:
                        st.latex(latex_statement)
                    except:
                        st.code(latex_statement)

                st.markdown("#### Explanation")
                st.write(theorem.get("explanation", "No explanation provided"))

                if "proof" in theorem:
                    st.markdown("#### Proof")
                    st.markdown(theorem["proof"])

                st.markdown(f"**Verification Iterations:** {theorem.get('verification_iterations', 'N/A')}")

                if "verification_result" in theorem:
                    st.markdown("#### Verification Result")
                    st.info(theorem["verification_result"][:500] + "..." if len(theorem.get("verification_result", "")) > 500 else theorem.get("verification_result", ""))

    # Remaining conjectures section
    if remaining_conjectures:
        st.markdown("### ⚠️ Unproven Conjectures")
        for idx, conj in enumerate(remaining_conjectures, 1):
            with st.expander(f"**Conjecture {idx}** - {conj.get('LaTeX', 'No title')[:100]}...", expanded=False):
                st.markdown("#### Statement")
                latex_statement = conj.get("LaTeX", "")
                if latex_statement:
                    try:
                        st.latex(latex_statement)
                    except:
                        st.code(latex_statement)

                st.markdown("#### Explanation")
                st.write(conj.get("explanation", "No explanation provided"))

                st.markdown(f"**Verification Iterations:** {conj.get('verification_iterations', 'N/A')}")

                # Show verification history
                if "verification_history" in conj:
                    st.markdown("#### Verification History")
                    for iter_data in conj["verification_history"]:
                        iteration_num = iter_data.get("iteration", "?")
                        st.markdown(f"**Iteration {iteration_num}:**")

                        with st.expander(f"View Iteration {iteration_num} Details"):
                            st.markdown("**Proof Attempt:**")
                            st.markdown(iter_data.get("proof", "")[:1000] + "..." if len(iter_data.get("proof", "")) > 1000 else iter_data.get("proof", ""))

                            st.markdown("**Verifier Feedback:**")
                            st.markdown(iter_data.get("verification", "")[:1000] + "..." if len(iter_data.get("verification", "")) > 1000 else iter_data.get("verification", ""))

    # Final synthesis report
    if "final_output" in results:
        st.markdown("## 📄 Final Synthesis Report")
        with st.expander("View Full Report", expanded=False):
            st.markdown(results["final_output"])

    # Download button
    st.markdown("## 💾 Export Results")
    col1, col2 = st.columns(2)

    with col1:
        # JSON download
        json_str = json.dumps(results, indent=2)
        st.download_button(
            label="📥 Download as JSON",
            data=json_str,
            file_name=f"proof_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

    with col2:
        # Markdown download
        md_content = f"""# Mathematical Proof Results

## Query
{results.get('query', '')}

## Summary
- Total Conjectures: {summary.get('total_conjectures', 0)}
- Theorems Proven: {summary.get('theorems_proven', 0)}
- Conjectures Remaining: {summary.get('conjectures_remaining', 0)}

## Results
{results.get('final_output', '')}
"""
        st.download_button(
            label="📥 Download as Markdown",
            data=md_content,
            file_name=f"proof_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>MathMind: Collaborative AI Mathematical Proof System</p>
    <p>Powered by CrewAI and GPT-4</p>
</div>
""", unsafe_allow_html=True)

# Show conversation log in sidebar when not processing
if not st.session_state.processing and st.session_state.conversation_log:
    with st.sidebar:
        st.markdown("---")
        with st.expander("📜 View Full Conversation Log"):
            log_text = "".join(st.session_state.conversation_log)
            st.text_area("", value=log_text, height=300, disabled=True)
