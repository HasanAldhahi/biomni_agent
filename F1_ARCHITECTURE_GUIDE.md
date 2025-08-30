# F1 Advanced Agent Architecture Framework 🧬

## Overview

The **F1 Agent** is an advanced biomedical AI agent framework that extends the baseline A1 architecture with **five distinct architectural patterns** designed for robust, self-correcting agent behavior in biomedical research environments.

## 🏗️ Architecture Variants

### 1. **Baseline Architecture** (`baseline`)
- **Purpose**: Standard biomedical queries with reliable execution
- **Features**: 
  - Standard execute/solution workflow
  - Basic error handling
  - Proven stability
- **Best for**: Production environments, simple tasks, reliable execution

### 2. **Hierarchical Expert Architecture** (`hierarchical_expert`)
- **Purpose**: Fault-tolerant execution with automatic tool failure recovery
- **Features**:
  - Tool benchmarking system with 5-minute cooldown
  - Dynamic tool availability tracking
  - Automatic fallback to alternative tools
  - Persistent failure memory across execution cycles
- **Best for**: Unreliable tool environments, experimental APIs, research with unstable tools

### 3. **Cognitive Corrector Architecture** (`cognitive_corrector`)
- **Purpose**: Systematic error correction with intention-based execution
- **Features**:
  - Intention-based code generation via JSON format
  - Built-in knowledge base for known tool corrections
  - Pre-defined error pattern matching
  - Structured tool parameter validation
- **Best for**: Well-studied domains with known failure patterns, systematic debugging

### 4. **Exploratory Sandbox Architecture** (`exploratory_sandbox`)
- **Purpose**: Environment-aware execution with dynamic capability discovery
- **Features**:
  - Pre-flight environment scanning
  - Tool availability detection
  - Environment profile generation
  - Adaptive execution based on available resources
- **Best for**: New deployments, heterogeneous environments, unknown tool landscapes

### 5. **Tool-Augmented Graph (TAG) Architecture** (`tool_augmented_graph`)
- **Purpose**: Advanced tool interaction with deep exploration capabilities
- **Features**:
  - Tool interrogation for detailed documentation
  - Safe sandbox code testing
  - Enhanced tool understanding workflow
  - Multi-modal execution paths (interrogate/sandbox/execute)
- **Best for**: Complex tool interactions, research requiring deep tool understanding

## 📊 Complete Feature Set

### Core Methods (inherited from A1)
- ✅ **Tool Management**: `add_tool()`, `get_custom_tool()`, `list_custom_tools()`, `remove_custom_tool()`  
- ✅ **Data Management**: `add_data()`, `get_custom_data()`, `list_custom_data()`, `remove_custom_data()`
- ✅ **Software Management**: `add_software()`, `get_custom_software()`, `list_custom_software()`, `remove_custom_software()`
- ✅ **Execution Control**: `_execute_code()`, `_parse_tool_call_from_code()`, `_inject_custom_functions_to_repl()`
- ✅ **System Management**: `_generate_system_prompt()`, `configure()`, `go()`, `result_formatting()`
- ✅ **Monitoring**: `count_tokens()`, `print_token_info()`

### Advanced Features
- ✅ **Dynamic Tool Retrieval**: Context-aware tool selection based on user prompts
- ✅ **Architecture-Specific State Management**: Specialized state handling for each architecture
- ✅ **Multi-Language Execution**: Python, R, and Bash script support
- ✅ **Custom Resource Integration**: Seamless integration of user-defined tools, data, and software

## 🚀 Usage Examples

### Basic Usage
```python
from biomni.agent.architecture.f1 import F1

# Initialize with specific architecture
agent = F1(
    path="./data",
    llm="claude-sonnet-4-20250514",
    architecture='hierarchical_expert',  # Choose your architecture
    timeout_seconds=300
)

# Execute biomedical query
log, result = agent.go("Analyze BRCA1 mutations and their impact on breast cancer risk")
```

### Architecture Comparison Study
```python
# Compare different architectures on the same task
architectures = ['baseline', 'hierarchical_expert', 'cognitive_corrector', 
                'exploratory_sandbox', 'tool_augmented_graph']

query = "Find protein-protein interactions for TP53 and analyze their functional significance"
results = {}

for arch in architectures:
    agent = F1(architecture=arch)
    log, result = agent.go(query)
    results[arch] = {
        'log': log,
        'result': result,
        'execution_steps': len(log),
        'success': 'solution' in result.lower()
    }
```

### Custom Tool Integration
```python
def custom_analysis_tool(gene_name: str, analysis_type: str):
    """Custom tool for specialized gene analysis."""
    # Your custom analysis logic here
    return f"Analysis of {gene_name} using {analysis_type}"

# Add custom tool to agent
agent = F1(architecture='tool_augmented_graph')
agent.add_tool(custom_analysis_tool)

# The tool is now available for use
log, result = agent.go("Use the custom analysis tool to analyze CFTR gene")
```

## 🔬 Research Applications

### Performance Comparison Studies
- **Error Recovery Rates**: Compare how different architectures handle tool failures
- **Execution Efficiency**: Measure steps required to complete identical tasks
- **Tool Usage Patterns**: Analyze which tools each architecture prefers
- **Success Rates**: Compare task completion rates across architectures

### Biomedical Use Cases
- **Gene Analysis Pipelines**: Compare architectures on multi-step genomic workflows
- **Drug Discovery Workflows**: Test robustness in pharmaceutical research tasks  
- **Protein Structure Analysis**: Evaluate performance on structural biology tasks
- **Clinical Data Processing**: Compare handling of medical datasets

### Architecture Selection Guidelines

| **Task Complexity** | **Environment Stability** | **Recommended Architecture** |
|---------------------|---------------------------|------------------------------|
| Simple, well-defined | Stable tools | `baseline` |
| Complex, multi-step | Unstable APIs | `hierarchical_expert` |
| Known failure patterns | Mixed stability | `cognitive_corrector` |
| Unknown environment | Variable tools | `exploratory_sandbox` |
| Deep tool exploration | Complex interactions | `tool_augmented_graph` |

## 📈 Technical Specifications

### AgentState Structure
```python
class AgentState(TypedDict):
    messages: List[BaseMessage]              # Conversation history
    next_step: str | None                    # Next workflow step
    
    # Hierarchical Expert
    benched_tools: dict[str, float]          # Failed tools with cooldown timestamps
    
    # Cognitive Corrector  
    tool_intention: dict                     # Structured tool intentions
    corrected_code: str                      # Corrected code after review
    
    # Tool-Augmented Graph
    interrogation_result: str                # Tool documentation results
    sandboxed_code: str                      # Safe code testing results
```

### Architecture-Specific Workflows

#### Baseline: `Generate → Execute → End`
Simple linear workflow with basic error handling.

#### Hierarchical Expert: `Generate → Execute → (Bench on failure) → Generate`
Dynamic tool management with failure tracking and recovery.

#### Cognitive Corrector: `Generate → Corrector → Execute → Generate`
Intention-driven execution with built-in error correction.

#### Exploratory Sandbox: `Environment Scan → Generate → Execute → End`
Environment-aware execution with pre-flight capability detection.

#### Tool-Augmented Graph: `Generate → (Interrogate|Sandbox|Execute) → Generate`
Multi-modal tool interaction with deep exploration capabilities.

## 🎯 Master's Thesis Applications

### Research Questions
1. **Which architecture performs best for different biomedical task types?**
2. **How do error recovery mechanisms affect overall task success rates?**
3. **What is the trade-off between execution time and reliability across architectures?**
4. **How does tool failure frequency impact architecture selection?**

### Evaluation Metrics
- **Task Completion Rate**: Percentage of successfully completed biomedical queries
- **Error Recovery Time**: Time to recover from tool failures  
- **Tool Utilization Efficiency**: Optimal tool selection patterns
- **Execution Step Count**: Efficiency measured by workflow steps
- **Failure Pattern Analysis**: Types and frequencies of failures per architecture

### Experimental Design
```python
# Example experimental framework
def run_architecture_experiment(tasks, architectures, iterations=10):
    results = {}
    
    for task in tasks:
        for arch in architectures:
            arch_results = []
            
            for i in range(iterations):
                agent = F1(architecture=arch)
                start_time = time.time()
                
                try:
                    log, result = agent.go(task)
                    execution_time = time.time() - start_time
                    
                    arch_results.append({
                        'success': True,
                        'execution_time': execution_time,
                        'steps': len(log),
                        'tools_used': extract_tools_from_log(log)
                    })
                except Exception as e:
                    arch_results.append({
                        'success': False,
                        'error': str(e),
                        'execution_time': time.time() - start_time
                    })
            
            results[f"{task}_{arch}"] = arch_results
    
    return results
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+
- Required dependencies (see biomni environment setup)
- Proper API keys in `.env` file

### Quick Start
1. **Clone Repository**: `git clone <biomni-repo>`
2. **Set Environment Variables**: Configure `.env` with your API keys
3. **Import F1**: `from biomni.agent.architecture.f1 import F1`
4. **Initialize Agent**: `agent = F1(architecture='your_choice')`
5. **Execute Query**: `log, result = agent.go("Your biomedical query")`

## 🎉 Conclusion

The **F1 Advanced Agent Architecture Framework** provides a comprehensive platform for:

✅ **Biomedical AI Research**: Five distinct architectures for different research needs  
✅ **Robustness Testing**: Built-in error recovery and fault tolerance mechanisms  
✅ **Performance Comparison**: Direct architecture comparison on identical tasks  
✅ **Custom Integration**: Seamless addition of domain-specific tools and data  
✅ **Master's Thesis Ready**: Complete framework for academic research and evaluation  

**Ready for advanced biomedical AI agent research and comparison studies!** 🧬🚀 