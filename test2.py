"""
Streamlit App for CrewAI Mathematical Proof System
Displays agent conversations in real-time and shows results with expandable sections
"""

import os
# Disable CrewAI telemetry BEFORE importing CrewAI
os.environ["OTEL_SDK_DISABLED"] = "true"

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
from typing import Optional, List, Dict, Any
import logging

# Suppress CrewAI logging noise
logging.getLogger("crewai").setLevel(logging.ERROR)
logging.getLogger("crewai.telemetry").setLevel(logging.CRITICAL)
logging.getLogger("crewai.events").setLevel(logging.ERROR)

# Import the proof system from main.py
from main import ProofSystem


# ============================================================================
# ANSI Code Parsing and Conversation Parsing Utilities
# ============================================================================

def strip_ansi_codes(text: str) -> str:
    """Remove ANSI color codes and escape sequences from text"""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def extract_box_content(content_text: str) -> str:
    """
    Extract actual content from inside CrewAI box formatting.
    Removes box characters (╭, │, ╰, ─) and extracts the text.
    """
    # Remove box drawing characters
    lines = content_text.split('\n')
    content_lines = []

    for line in lines:
        # Skip lines that are just box borders
        if line.strip() and not all(c in '╭╮│╰╯─═' for c in line.strip()):
            # Remove leading/trailing box characters
            cleaned = line.strip()
            # Remove leading │ and whitespace
            if cleaned.startswith('│'):
                cleaned = cleaned[1:].strip()
            # Remove trailing │
            if cleaned.endswith('│'):
                cleaned = cleaned[:-1].strip()

            if cleaned:
                content_lines.append(cleaned)

    return '\n'.join(content_lines)


def parse_crewai_output(log_text: str) -> List[Dict[str, Any]]:
    """
    Parse CrewAI output into structured conversation steps.
    Returns a list of conversation events with type, agent, content, etc.
    """
    # Strip ANSI codes first
    clean_text = strip_ansi_codes(log_text)

    events = []

    # Split into lines for easier parsing
    lines = clean_text.split('\n')

    current_event = None
    buffer = []
    in_box = False

    for i, line in enumerate(lines):
        # Detect start of a box
        if '╭' in line:
            in_box = True
            if current_event and buffer:
                # Save previous event
                events.append(current_event)
            buffer = [line]
            current_event = None

        # Detect end of a box
        elif '╰' in line:
            in_box = False
            if buffer:
                buffer.append(line)
                # Determine event type from buffer content
                full_content = '\n'.join(buffer)

                if '🤖 Agent Started' in full_content or 'Agent Started' in full_content:
                    current_event = {'type': 'agent_started', 'content': buffer, 'line_num': i}
                elif 'Crew Execution Started' in full_content:
                    current_event = {'type': 'crew_started', 'content': buffer, 'line_num': i}
                elif '🔧 Agent Tool Execution' in full_content or 'Agent Tool Execution' in full_content:
                    current_event = {'type': 'tool_execution', 'content': buffer, 'line_num': i}
                elif 'Tool Input' in full_content and 'Tool Output' not in full_content:
                    current_event = {'type': 'tool_input', 'content': buffer, 'line_num': i}
                elif 'Tool Output' in full_content:
                    current_event = {'type': 'tool_output', 'content': buffer, 'line_num': i}
                elif '✅ Agent Final Answer' in full_content or 'Agent Final Answer' in full_content:
                    current_event = {'type': 'final_answer', 'content': buffer, 'line_num': i}
                elif 'Task Completion' in full_content or 'Task Completed' in full_content:
                    current_event = {'type': 'task_completion', 'content': buffer, 'line_num': i}
                elif 'Crew Completion' in full_content or 'Crew Execution Completed' in full_content:
                    current_event = {'type': 'crew_completion', 'content': buffer, 'line_num': i}
                else:
                    # Generic event
                    current_event = {'type': 'unknown', 'content': buffer, 'line_num': i}

                events.append(current_event)
                buffer = []
                current_event = None
        elif in_box:
            buffer.append(line)

    # Add final event if exists
    if current_event and current_event not in events:
        events.append(current_event)

    # Parse details from each event
    parsed_events = []
    for event in events:
        content_text = '\n'.join(event['content'])
        # Extract clean content without box characters
        clean_content = extract_box_content(content_text)

        parsed_event = {
            'type': event['type'],
            'raw_content': content_text,
            'clean_content': clean_content,
            'line_num': event['line_num']
        }

        # Extract agent name if present
        agent_match = re.search(r'Agent:\s*([^\n│]+)', content_text)
        if agent_match:
            parsed_event['agent'] = agent_match.group(1).strip()

        # Extract task if present
        task_match = re.search(r'Task:\s*(.*?)(?:│|╰|$)', content_text, re.DOTALL)
        if task_match:
            task_text = task_match.group(1).strip()
            # Clean up the task text
            task_lines = [line.strip('│ ') for line in task_text.split('\n') if line.strip() and not all(c in '│ ─' for c in line.strip())]
            parsed_event['task'] = '\n'.join(task_lines)

        # Extract thought if present
        thought_match = re.search(r'Thought:\s*(.*?)(?:│|Using Tool|$)', content_text, re.DOTALL)
        if thought_match:
            thought_text = thought_match.group(1).strip()
            thought_lines = [line.strip('│ ') for line in thought_text.split('\n') if line.strip()]
            parsed_event['thought'] = '\n'.join(thought_lines)

        # Extract tool name if present
        tool_match = re.search(r'Using Tool:\s*([^\n│]+)', content_text)
        if tool_match:
            parsed_event['tool'] = tool_match.group(1).strip()

        # For tool input/output, extract the actual data
        if event['type'] == 'tool_input':
            # Try to extract JSON or text content
            json_match = re.search(r'\{[^}]*\}', clean_content, re.DOTALL)
            if json_match:
                parsed_event['input_data'] = json_match.group(0)
            else:
                parsed_event['input_data'] = clean_content

        if event['type'] == 'tool_output':
            # Extract everything after "Tool Output" header
            output_match = re.search(r'Tool Output.*?(?:╮|─+)(.*?)(?:╰|$)', content_text, re.DOTALL)
            if output_match:
                output_text = output_match.group(1).strip()
                output_lines = [line.strip('│ ') for line in output_text.split('\n') if line.strip() and not all(c in '│ ─╰' for c in line.strip())]
                parsed_event['output_data'] = '\n'.join(output_lines)
            else:
                parsed_event['output_data'] = clean_content

        # Extract final answer if present
        answer_match = re.search(r'Final Answer:\s*(.*?)(?:╰|$)', content_text, re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()
            answer_lines = [line.strip('│ ') for line in answer_text.split('\n') if line.strip() and not all(c in '│ ─╰' for c in line.strip())]
            parsed_event['answer'] = '\n'.join(answer_lines)

        parsed_events.append(parsed_event)

    return parsed_events


def format_event_for_display(event: Dict[str, Any], index: int) -> str:
    """Format a parsed event for nice display in Streamlit"""
    event_type = event.get('type', 'unknown')

    if event_type == 'agent_started':
        agent_name = event.get('agent', 'Unknown Agent')
        task = event.get('task', 'No task description')
        return f"""
**🤖 Agent Started: {agent_name}**

**Task:** {task[:500]}{'...' if len(task) > 500 else ''}
"""

    elif event_type == 'crew_started':
        return "**🚀 Crew Execution Started**\n\nInitializing agent workflow..."

    elif event_type == 'tool_execution':
        agent_name = event.get('agent', 'Unknown Agent')
        thought = event.get('thought', 'No thought provided')
        tool = event.get('tool', 'Unknown Tool')
        return f"""
**🔧 Tool Execution**

**Agent:** {agent_name}

**Thought:** {thought[:400]}{'...' if len(thought) > 400 else ''}

**Tool:** {tool}
"""

    elif event_type == 'tool_input':
        input_data = event.get('input_data', event.get('clean_content', 'No input data'))
        # Try to format JSON nicely
        try:
            if input_data.strip().startswith('{'):
                import json
                parsed = json.loads(input_data)
                formatted = json.dumps(parsed, indent=2)
                return f"**📥 Tool Input**\n\n```json\n{formatted}\n```"
        except:
            pass
        return f"**📥 Tool Input**\n\n```\n{input_data[:500]}{'...' if len(input_data) > 500 else ''}\n```"

    elif event_type == 'tool_output':
        output_data = event.get('output_data', event.get('clean_content', 'No output data'))
        return f"**📤 Tool Output**\n\n```\n{output_data[:800]}{'...' if len(output_data) > 800 else ''}\n```"

    elif event_type == 'final_answer':
        agent_name = event.get('agent', 'Unknown Agent')
        answer = event.get('answer', 'No answer provided')
        return f"""
**✅ Final Answer from {agent_name}**

{answer[:1500]}{'...' if len(answer) > 1500 else ''}
"""

    elif event_type == 'task_completion':
        return "**✓ Task Completed**"

    elif event_type == 'crew_completion':
        return "**🎉 Crew Execution Completed**"

    else:
        clean_content = event.get('clean_content', event.get('raw_content', 'No content'))
        return f"**Unknown Event**\n\n{clean_content[:300]}"

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
        background-color: #ffffff;
        color: #333333;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
        font-size: 0.9rem;
        line-height: 1.6;
    }
    .conversation-event {
        background-color: #f8f9fa;
        color: #212529;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.75rem 0;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .conversation-event.agent-started {
        border-left-color: #28a745;
    }
    .conversation-event.tool-execution {
        border-left-color: #ffc107;
    }
    .conversation-event.final-answer {
        border-left-color: #dc3545;
    }
    .conversation-container {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        max-height: 600px;
        overflow-y: auto;
        border: 1px solid #dee2e6;
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
    .event-step-number {
        background-color: #1f77b4;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
        font-size: 0.8rem;
        margin-right: 0.5rem;
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
    st.header("⚙️ Configuration")

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
            # Only use queue to communicate with main thread
            # DO NOT access st.session_state from background thread
            if self.message_queue:
                try:
                    self.message_queue.put_nowait(("log", text))
                except queue.Full:
                    pass
        return len(text)

    def flush(self):
        pass

    def get_log(self):
        return "".join(self.log)


def update_progress(phase: str, progress: float):
    """Update progress tracking - DEPRECATED, use message queue instead"""
    # Do not access st.session_state from background thread
    # This function is kept for compatibility but does nothing
    pass


def run_proof_system(query: str, agent_type: str, max_iter: int, message_queue: queue.Queue):
    """Run the proof system and capture output"""
    try:
        # Initialize proof system
        # DO NOT access st.session_state from this background thread
        message_queue.put(("progress", "Initializing agents...", 0.1))

        proof_system = ProofSystem(max_iterations=max_iter, agent_type=agent_type)

        # Capture stdout
        stream_capture = StreamCapture(message_queue=message_queue)

        message_queue.put(("progress", "Generating conjectures...", 0.2))

        # Run the proof system with captured output
        with redirect_stdout(stream_capture):
            results = proof_system.process_query(query)

        message_queue.put(("progress", "Complete!", 1.0))
        message_queue.put(("results", results))

    except Exception as e:
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
                    elif message[0] == "log":
                        # Add log message from background thread
                        st.session_state.conversation_log.append(message[1])
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

    # Create tabs for different views
    tab1, tab2 = st.tabs(["📊 Structured View", "📜 Raw Log"])

    with tab1:
        st.markdown('<div class="conversation-container">', unsafe_allow_html=True)
        if st.session_state.conversation_log:
            # Parse and display structured conversation
            full_log = "".join(st.session_state.conversation_log)
            parsed_events = parse_crewai_output(full_log)

            if parsed_events:
                for idx, event in enumerate(parsed_events, 1):
                    event_type = event.get('type', 'unknown')

                    # Create a container for each event
                    with st.container():
                        # Display step number and event type indicator
                        col1, col2 = st.columns([1, 20])
                        with col1:
                            st.markdown(f"**{idx}**")

                        with col2:
                            if event_type == 'agent_started':
                                agent_name = event.get('agent', 'Unknown Agent')
                                task = event.get('task', 'No task description')
                                st.markdown(f"**🤖 Agent Started: {agent_name}**")
                                st.info(task)

                            elif event_type == 'crew_started':
                                st.markdown("**🚀 Crew Execution Started**")
                                st.caption("Initializing agent workflow...")

                            elif event_type == 'tool_execution':
                                agent_name = event.get('agent', 'Unknown Agent')
                                thought = event.get('thought', 'No thought provided')
                                tool = event.get('tool', 'Unknown Tool')
                                st.markdown(f"**🔧 {agent_name}** is using **{tool}**")
                                st.write("**Thought:**", thought)

                            elif event_type == 'tool_input':
                                input_data = event.get('input_data', event.get('clean_content', 'No input data'))
                                st.markdown("**📥 Tool Input**")
                                # Try to display as JSON
                                try:
                                    if input_data.strip().startswith('{'):
                                        parsed_json = json.loads(input_data)
                                        st.json(parsed_json)
                                    else:
                                        st.code(input_data[:500], language="text")
                                except:
                                    st.code(input_data[:500], language="text")

                            elif event_type == 'tool_output':
                                output_data = event.get('output_data', event.get('clean_content', 'No output data'))
                                st.markdown("**📤 Tool Output**")
                                st.text(output_data[:1000])

                            elif event_type == 'final_answer':
                                agent_name = event.get('agent', 'Unknown Agent')
                                answer = event.get('answer', 'No answer provided')
                                st.markdown(f"**✅ Final Answer from {agent_name}**")
                                st.markdown(answer[:2000])

                            elif event_type == 'task_completion':
                                st.markdown("**✓ Task Completed**")

                            elif event_type == 'crew_completion':
                                st.markdown("**🎉 Crew Execution Completed**")

                            else:
                                clean_content = event.get('clean_content', 'No content')
                                st.markdown(f"**Event:** {event_type}")
                                if clean_content:
                                    st.text(clean_content[:200])

                        st.markdown("---")
            else:
                st.info("Parsing agent conversations...")
        else:
            st.info("Waiting for agents to start conversing...")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        # Show raw log with ANSI codes stripped
        if st.session_state.conversation_log:
            full_log = "".join(st.session_state.conversation_log)
            clean_log = strip_ansi_codes(full_log)
            st.text_area("Raw Log", value=clean_log, height=400, disabled=True)

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
            clean_log_text = strip_ansi_codes(log_text)
            st.text_area("", value=clean_log_text, height=300, disabled=True)
