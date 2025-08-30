#!/usr/bin/env python3
"""
Example usage of the F1 agent with different architectures.

This demonstrates how to initialize and use the F1 agent with the 5 different
architectural variants for advanced biomedical AI agent research.
"""

from biomni.agent.architecture.f1 import F1

def test_f1_architectures():
    """Test all F1 architectures with a simple biomedical query."""
    
    # Example biomedical query
    query = "Find information about the CFTR gene and its role in cystic fibrosis."
    
    # Test each architecture
    architectures = [
        'baseline',
        'hierarchical_expert', 
        'cognitive_corrector',
        'exploratory_sandbox',
        'tool_augmented_graph'
    ]
    
    print("🧬 F1 Agent Architecture Comparison 🧬\n")
    print("=" * 50)
    
    for arch in architectures:
        print(f"\n🤖 Testing {arch.upper().replace('_', ' ')} Architecture")
        print("-" * 40)
        
        try:
            # Initialize F1 agent with specific architecture
            agent = F1(
                path="./data",
                llm="claude-sonnet-4-20250514",
                architecture=arch,
                timeout_seconds=300
            )
            
            print(f"✅ Successfully initialized {arch} architecture")
            print(f"📊 Architecture features:")
            
            # Display architecture-specific features
            if arch == 'baseline':
                print("   - Standard execute/solution workflow")
                print("   - Basic error handling")
            elif arch == 'hierarchical_expert':
                print("   - Tool benchmarking system") 
                print("   - 5-minute cooldown for failed tools")
                print("   - Dynamic tool availability tracking")
            elif arch == 'cognitive_corrector':
                print("   - Intention-based execution")
                print("   - Built-in tool correction knowledge base")
                print("   - JSON intention parsing")
            elif arch == 'exploratory_sandbox':
                print("   - Pre-flight environment scanning")
                print("   - Environment-aware execution")
                print("   - Available tool detection")
            elif arch == 'tool_augmented_graph':
                print("   - Tool interrogation capabilities")
                print("   - Sandbox code testing")
                print("   - Enhanced tool exploration")
            
            # Note: Uncomment the following line to actually run the agent
            # log, result = agent.go(query)
            # print(f"📝 Result: {result[:100]}...")
            
        except Exception as e:
            print(f"❌ Error initializing {arch}: {e}")
        
        print()

def architecture_comparison_guide():
    """Print a guide explaining when to use each architecture."""
    
    guide = """
🎯 F1 ARCHITECTURE SELECTION GUIDE
==================================

📊 BASELINE
-----------
• Use for: Standard biomedical queries, reliable execution
• Benefits: Proven workflow, minimal overhead
• Best for: Production environments, simple tasks

🏗️ HIERARCHICAL EXPERT  
-----------------------
• Use for: Unreliable tool environments, fault tolerance needed
• Benefits: Automatic tool failure recovery, adaptive execution
• Best for: Research with experimental tools, unstable APIs

🧠 COGNITIVE CORRECTOR
-----------------------
• Use for: Known tool issues, systematic error correction
• Benefits: Pre-defined error corrections, intention clarity
• Best for: Well-studied domains with known failure patterns

🔬 EXPLORATORY SANDBOX
-----------------------
• Use for: Unknown environments, dynamic tool discovery
• Benefits: Environment profiling, availability-aware execution  
• Best for: New deployments, heterogeneous environments

🔧 TOOL-AUGMENTED GRAPH (TAG)
-----------------------------
• Use for: Complex tool interactions, experimental workflows
• Benefits: Tool interrogation, safe sandbox testing
• Best for: Research requiring deep tool understanding

📈 RESEARCH APPLICATIONS
========================
• Compare architecture performance on same biomedical tasks
• Measure error recovery rates and execution success
• Analyze tool usage patterns across architectures
• Study agent self-correction capabilities

💡 USAGE EXAMPLE
================
    from biomni.agent.architecture.f1 import F1
    
    # Initialize with specific architecture
    agent = F1(architecture='hierarchical_expert')
    
    # Execute biomedical query
    log, result = agent.go("Analyze BRCA1 mutations in breast cancer")
    
    # Compare results across architectures for research
"""
    
    print(guide)

if __name__ == "__main__":
    print("🧬 F1 Advanced Agent Architecture Framework 🧬\n")
    
    # Show the guide first
    architecture_comparison_guide()
    
    # Test all architectures (initialization only)
    test_f1_architectures()
    
    print("\n🎉 F1 Agent Setup Complete!")
    print("📚 Ready for biomedical AI agent research and comparison studies.") 