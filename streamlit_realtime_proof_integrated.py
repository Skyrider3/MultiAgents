"""
Real-time Streamlit App for  Proof System
INTEGRATED VERSION - Actually runs the  agents
"""

import streamlit as st
import json
import time
import sys
import io
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List, Any, Optional
import threading
import queue
from dataclasses import dataclass, asdict
import re
from contextlib import redirect_stdout, redirect_stderr

# Import the actual  proof system
from crewai_proof_system import ProofSystem, IterativeProofVerifier
from crewai import Agent, Task, Crew

# Custom agent tracker for intercepting  outputs
@dataclass
class AgentMessage:
    timestamp: str
    agent: str
    action: str
    content: str
    evidence: List[Dict] = None
    verdict: Optional[str] = None
    iteration: Optional[int] = None
    phase: str = "unknown"


class CrewAIOutputInterceptor:
    """Intercepts console output and converts to structured messages"""

    def __init__(self, message_queue: queue.Queue):
        self.message_queue = message_queue
        self.buffer = ""
        self.current_agent = None
        self.current_phase = "initialization"
        self.current_iteration = 0

    def write(self, text):
        """Intercept stdout writes """
        # Store in buffer
        self.buffer += text

        # Parse  output patterns
        if "Expert Number Theory Mathematician" in text:
            self.current_agent = "Number Theorist"
        elif "Mathematical Experimenter" in text:
            self.current_agent = "Experimenter"
        elif "Mathematical Proof Verifier" in text:
            self.current_agent = "Verifier"
        elif "Research Coordinator" in text:
            self.current_agent = "Coordinator"

        # Detect phases
        if "PHASE 1:" in text:
            self.current_phase = "conjecture_generation"
        elif "PHASE 2:" in text:
            self.current_phase = "verification"
        elif "PHASE 3:" in text:
            self.current_phase = "synthesis"
        elif "ITERATION" in text:
            # Extract iteration number
            match = re.search(r"ITERATION (\d+)", text)
            if match:
                self.current_iteration = int(match.group(1))

        # Create message when we have meaningful content
        if self.current_agent and len(text.strip()) > 0:
            # Extract evidence if present
            evidence = self.extract_evidence(text)

            # Determine action type
            action = "Processing"
            if "Retrieving" in text or "retrieve" in text.lower():
                action = "Retrieving Evidence"
            elif "Generating" in text or "generate" in text.lower():
                action = "Generating Conjectures"
            elif "Verifying" in text or "verify" in text.lower():
                action = "Verification Check"
            elif "Refining" in text or "refine" in text.lower():
                action = "Proof Refinement"
            elif "Synthesizing" in text:
                action = "Final Synthesis"

            # Extract verdict for verifier
            verdict = None
            if self.current_agent == "Verifier":
                verdict = self.parse_verdict(text)

            # Create and queue message
            msg = AgentMessage(
                timestamp=datetime.now().isoformat(),
                agent=self.current_agent,
                action=action,
                content=text.strip()[:500],  # Limit content length
                evidence=evidence,
                verdict=verdict,
                iteration=self.current_iteration if self.current_phase == "verification" else None,
                phase=self.current_phase
            )

            self.message_queue.put(msg)

        # Also write to actual stdout for debugging
        sys.__stdout__.write(text)

    def flush(self):
        pass

    def extract_evidence(self, text: str) -> List[Dict]:
        """Extract evidence citations from text"""
        evidence = []

        # Look for Source: patterns
        source_pattern = r"Source:\s*([^\n]+)"
        matches = re.findall(source_pattern, text)
        for match in matches:
            evidence.append({
                'source': match.strip(),
                'text': "Evidence extracted from  output",
                'confidence': 0.85
            })

        # Look for citations
        citation_patterns = [
            r'\[([^\]]+__\d+)\]',
            r'\(([A-Z][a-z]+ \d{4})\)',
        ]

        for pattern in citation_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                evidence.append({
                    'source': match,
                    'text': f"Citation: {match}",
                    'confidence': 0.85
                })

        return evidence

    def parse_verdict(self, text: str) -> str:
        """Extract verdict from verifier output"""
        text_upper = text.upper()
        if "VALID" in text_upper:
            return "VALID"
        elif "INVALID" in text_upper:
            return "INVALID"
        elif "INCOMPLETE" in text_upper:
            return "INCOMPLETE"
        return "UNKNOWN"


class RealTimeProofRunner:
    """Manages the actual  proof system execution with live updates"""

    def __init__(self):
        self.message_queue = queue.Queue()
        self.messages = []
        self.is_running = False
        self.proof_system = None

    def run_proof_system(self, query: str, max_iterations: int = 5):
        """Run the actual  proof system with output interception"""

        try:
            self.is_running = True

            # Create interceptor
            interceptor = CrewAIOutputInterceptor(self.message_queue)

            # Initialize the proof system
            self.proof_system = ProofSystem(max_iterations=max_iterations)

            # Redirect stdout to capture output
            with redirect_stdout(interceptor):
                # Run the actual proof system
                results = self.proof_system.process_query(query)

            # Add final message with results
            final_msg = AgentMessage(
                timestamp=datetime.now().isoformat(),
                agent="System",
                action="Proof System Complete",
                content=f"Final Status: {results.get('status', 'Unknown')}",
                phase="complete"
            )
            self.message_queue.put(final_msg)

            return results

        except Exception as e:
            error_msg = AgentMessage(
                timestamp=datetime.now().isoformat(),
                agent="System",
                action="Error",
                content=f"Error running proof system: {str(e)}",
                phase="error"
            )
            self.message_queue.put(error_msg)
            raise e

        finally:
            self.is_running = False


def create_agent_network_graph(messages: List[AgentMessage]) -> go.Figure:
    """Create an interactive network graph of agent interactions"""

    # Count interactions between agents
    interactions = {}
    for i in range(len(messages) - 1):
        source = messages[i].agent
        target = messages[i + 1].agent
        if source != target:
            key = f"{source}->{target}"
            interactions[key] = interactions.get(key, 0) + 1

    # Create Plotly Sankey diagram
    agents = list(set([m.agent for m in messages]))
    agent_indices = {agent: i for i, agent in enumerate(agents)}

    sources = []
    targets = []
    values = []
    labels = []

    for interaction, count in interactions.items():
        source, target = interaction.split("->")
        if source in agent_indices and target in agent_indices:
            sources.append(agent_indices[source])
            targets.append(agent_indices[target])
            values.append(count)
            labels.append(f"{count} messages")

    if sources:  # Only create figure if we have interactions
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=agents,
                color=["#2196F3", "#FF9800", "#E91E63", "#4CAF50", "#9C27B0"][:len(agents)]
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                label=labels
            )
        )])

        fig.update_layout(
            title="Agent Interaction Flow",
            font_size=10,
            height=300
        )
    else:
        # Create empty figure if no interactions yet
        fig = go.Figure()
        fig.update_layout(title="Waiting for agent interactions...")

    return fig


def display_message_timeline(messages: List[AgentMessage]):
    """Display messages in a timeline format"""

    for msg in messages:
        # Determine styling based on agent
        agent_colors = {
            "Number Theorist": ("🧠", "#2196F3"),
            "Experimenter": ("🔬", "#FF9800"),
            "Verifier": ("🔍", "#E91E63"),
            "Coordinator": ("📊", "#4CAF50"),
            "System": ("💻", "#9C27B0")
        }

        emoji, color = agent_colors.get(msg.agent, ("🤖", "#666666"))

        # Create columns for timeline display
        col1, col2, col3 = st.columns([1, 1, 8])

        with col1:
            timestamp = datetime.fromisoformat(msg.timestamp)
            st.caption(timestamp.strftime("%H:%M:%S"))

        with col2:
            st.markdown(f"<h3>{emoji}</h3>", unsafe_allow_html=True)

        with col3:
            # Message container with styling
            container = st.container()
            with container:
                if msg.iteration:
                    st.caption(f"Iteration {msg.iteration}")

                st.markdown(f"""
                <div style="border-left: 3px solid {color}; padding-left: 10px; margin-bottom: 10px;">
                    <b>{msg.agent}</b> - {msg.action}<br>
                    <span style="color: #666; font-size: 0.9em;">{msg.content[:200]}...</span>
                </div>
                """, unsafe_allow_html=True)

                # Show evidence if available
                if msg.evidence:
                    with st.expander(f"📚 Evidence ({len(msg.evidence)} sources)"):
                        for e in msg.evidence:
                            st.caption(f"**{e['source']}**: {e.get('text', 'N/A')[:100]}...")

                # Show verdict for verifier
                if msg.verdict:
                    verdict_colors = {
                        "VALID": "✅",
                        "INVALID": "❌",
                        "INCOMPLETE": "⚠️",
                        "UNKNOWN": "❓"
                    }
                    st.markdown(f"**Verdict**: {verdict_colors.get(msg.verdict, '')} {msg.verdict}")


def main():
    st.set_page_config(
        page_title="Proof System",
        page_icon="🔬",
        layout="wide"
    )

    st.title("🔬  Mathematical conjucture discovery  ")
    # st.markdown("")

    # Initialize session state
    if 'runner' not in st.session_state:
        st.session_state.runner = RealTimeProofRunner()

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Control Panel")

        query = st.text_area(
            "Mathematical Query",
            "Distribution of zeros of Ramanujan tau(n) modulo small primes",
            height=100
        )

        max_iterations = st.slider("Max Iterations", 1, 5, 3)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Start Proof", type="primary", disabled=st.session_state.runner.is_running):
                st.session_state.messages.clear()
                st.session_state.runner.messages.clear()
                st.session_state.runner.is_running = True

                # Show initial status
                with st.spinner("Initializing  agents..."):
                    # Run in thread to avoid blocking
                    thread = threading.Thread(
                        target=st.session_state.runner.run_proof_system,
                        args=(query, max_iterations)
                    )
                    thread.daemon = True
                    thread.start()

                    # Start message collection
                    st.rerun()

        with col2:
            if st.button("🔄 Clear", disabled=st.session_state.runner.is_running):
                st.session_state.messages.clear()
                st.session_state.runner.messages.clear()
                st.rerun()

        st.divider()

        # Statistics
        st.header("📊 Statistics")

        if st.session_state.messages:
            st.metric("Total Messages", len(st.session_state.messages))

            # Count by agent
            agent_counts = {}
            for msg in st.session_state.messages:
                agent_counts[msg.agent] = agent_counts.get(msg.agent, 0) + 1

            for agent, count in agent_counts.items():
                st.caption(f"{agent}: {count}")

            # Current phase
            if st.session_state.messages:
                st.metric("Current Phase", st.session_state.messages[-1].phase.replace("_", " ").title())

        # System status
        st.divider()
        st.header("🔧 System Status")
        if st.session_state.runner.is_running:
            st.success("🟢  System Running")
        else:
            st.info("⚪  System Idle")

    # Main content area
    if st.session_state.runner.is_running:
        # Auto-refresh placeholder for live updates
        placeholder = st.empty()
        message_container = st.container()

        # Collect messages from queue
        while not st.session_state.runner.message_queue.empty():
            try:
                msg = st.session_state.runner.message_queue.get_nowait()
                st.session_state.messages.append(msg)
                st.session_state.runner.messages.append(msg)
            except queue.Empty:
                break

        # Display current state
        with message_container:
            display_current_state()

        # Auto-refresh while running
        if st.session_state.runner.is_running:
            time.sleep(1)
            st.rerun()
    else:
        display_current_state()


def display_current_state():
    """Display the current state of the proof system"""

    messages = st.session_state.messages

    if not messages:
        st.info("👈 Click 'Start Proof' to run the actual  proof system")

        # Show instructions
        with st.expander("ℹ️ About This Integration"):
            st.markdown("""
            This version **actually runs the  proof system** in real-time:

            - ✅ **Real Agent Execution**: Runs actual  agents (Number Theorist, Experimenter, Verifier, Coordinator)
            - ✅ **Live Output Capture**: Intercepts and displays  console output in real-time
            - ✅ **Evidence Retrieval**: Uses the actual evidence retrieval from your paper database
            - ✅ **Iterative Verification**: Runs the real iterative proof verification loop
            - ✅ **Real Results**: Shows actual proof attempts and verification verdicts

            **Note**: This requires your OpenAI API key to be configured in the .env file.
            """)
        return

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📜 Timeline View",
        "🔗 Network Graph",
        "📊 Verification Progress",
        "📝 Evidence Log"
    ])

    with tab1:
        st.header("Agent Communication Timeline")
        display_message_timeline(messages)

    with tab2:
        st.header("Agent Interaction Network")
        if len(messages) > 1:
            fig = create_agent_network_graph(messages)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Collecting agent interactions...")

    with tab3:
        st.header("Verification Progress")

        # Extract verification messages
        verification_msgs = [m for m in messages if m.phase == "verification"]

        if verification_msgs:
            # Progress bar
            iterations_complete = max([m.iteration for m in verification_msgs if m.iteration] or [0])
            if iterations_complete > 0:
                progress = iterations_complete / 5
                st.progress(progress)
                st.caption(f"Iteration {iterations_complete} of 5")

            # Show iteration details
            for iteration in range(1, iterations_complete + 1):
                iteration_msgs = [m for m in verification_msgs if m.iteration == iteration]

                with st.expander(f"Iteration {iteration}", expanded=(iteration == iterations_complete)):
                    for msg in iteration_msgs:
                        if msg.agent == "Verifier":
                            if msg.verdict == "VALID":
                                st.success(f"**Verifier**: {msg.verdict} - {msg.content}")
                            elif msg.verdict in ["INVALID", "INCOMPLETE"]:
                                st.error(f"**Verifier**: {msg.verdict} - {msg.content}")
                            else:
                                st.info(f"**Verifier**: {msg.content}")
                        elif msg.agent == "Experimenter":
                            st.info(f"**Experimenter**: {msg.content}")

            # Final status
            last_verdict = next((m.verdict for m in reversed(verification_msgs) if m.verdict), None)
            if last_verdict == "VALID":
                st.success("✅ Proof Validated!")
            elif iterations_complete >= 5:
                st.warning("⚠️ Maximum iterations reached - Remains Conjecture")
        else:
            st.info("Waiting for verification phase to begin...")

    with tab4:
        st.header("Evidence and Citations")

        # Collect all evidence
        all_evidence = []
        for msg in messages:
            if msg.evidence:
                for e in msg.evidence:
                    all_evidence.append({
                        'Timestamp': msg.timestamp,
                        'Agent': msg.agent,
                        'Source': e.get('source', 'Unknown'),
                        'Evidence': e.get('text', '')[:100] + '...'
                    })

        if all_evidence:
            df = pd.DataFrame(all_evidence)
            st.dataframe(df, use_container_width=True)

            # Evidence summary
            st.subheader("📚 Unique Sources Referenced")
            unique_sources = df['Source'].unique()
            for source in unique_sources:
                st.caption(f"• {source}")
        else:
            st.info("No evidence collected yet")


if __name__ == "__main__":
    main()