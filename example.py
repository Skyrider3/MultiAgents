"""
Example script showing how to use the conjecture generation flow programmatically.
"""
import os
from dotenv import load_dotenv
from conjecture_flow import ConjectureFlow


def example_usage():
    """Example of using the conjecture flow."""
    # Load environment variables
    load_dotenv()
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found.")
        return
    
    # Initialize the flow
    print("Initializing conjecture generation flow...\n")
    flow = ConjectureFlow(
        index_dir="index",
        model_name="gpt-4",
        temperature=0.7
    )
    
    # Example query
    query = "prime number distribution"
    
    print(f"Query: {query}")
    print("Running conjecture generation flow...\n")
    
    # Option 1: Run and get final result
    result = flow.run(query)
    
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\nQuery: {result['query']}")
    print(f"\nRetrieved {len(result['retrieved_docs'])} documents")
    print("\nTop 3 retrieved documents:")
    for i, doc in enumerate(result['retrieved_docs'][:3], 1):
        print(f"  {i}. {doc['source']} (page {doc['page']}, distance: {doc['distance']:.4f})")
    
    print(f"\nGenerated {len(result['conjectures'])} conjectures:")
    print("-" * 60)
    for i, conjecture in enumerate(result['conjectures'], 1):
        print(f"{i}. {conjecture}")
    
    # Option 2: Stream the process (uncomment to use)
    # print("\n" + "=" * 60)
    # print("STREAMING PROCESS")
    # print("=" * 60)
    # for state_update in flow.stream(query):
    #     for node_name, node_state in state_update.items():
    #         print(f"\n[{node_name}]")
    #         if 'conjectures' in node_state:
    #             print(f"  Conjectures so far: {len(node_state['conjectures'])}")
    #         if 'retrieved_docs' in node_state:
    #             print(f"  Retrieved docs: {len(node_state['retrieved_docs'])}")


if __name__ == "__main__":
    example_usage()

