"""
Script to visualize the LangGraph flow structure.
"""
from conjecture_flow import ConjectureFlow
import os
from dotenv import load_dotenv


def visualize_graph():
    """Visualize the graph structure."""
    load_dotenv()
    
    # Initialize flow (this builds the graph)
    flow = ConjectureFlow(index_dir="index")
    
    # Get the graph structure
    graph = flow.graph
    
    # Print graph structure
    print("=" * 60)
    print("LangGraph Flow Structure")
    print("=" * 60)
    print("\nNodes:")
    for node in graph.nodes:
        print(f"  - {node}")
    
    print("\nEdges:")
    for edge in graph.edges:
        print(f"  - {edge}")
    
    print("\nEntry point:", graph.entry_point)
    print("\n" + "=" * 60)
    
    # Try to generate a visual representation
    try:
        # This will create a visual representation if graphviz is available
        graph_image = graph.get_graph().draw_mermaid_png()
        with open("conjecture_flow_graph.png", "wb") as f:
            f.write(graph_image)
        print("\nGraph visualization saved to: conjecture_flow_graph.png")
    except Exception as e:
        print(f"\nCould not generate graph visualization: {e}")
        print("Install graphviz to enable visualization: pip install pygraphviz")


if __name__ == "__main__":
    visualize_graph()

