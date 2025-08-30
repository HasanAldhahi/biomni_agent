import glob
import inspect
import os
import re
from typing import Literal, TypedDict
from IPython.display import Image, display
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph


from biomni.env_desc import data_lake_dict, library_content_dict
from biomni.llm import get_llm
from biomni.model.retriever import ToolRetriever
from biomni.tool.support_tools import run_python_repl
from biomni.tool.tool_registry import ToolRegistry
from biomni.utils import (
    check_and_download_s3_files,
    download_and_unzip,
    function_to_api_schema,
    pretty_print,
    read_module2api,
    run_bash_script,
    run_r_code,
    run_with_timeout,
    textify_api_dict,
)

if os.path.exists(".env"):
    load_dotenv(".env", override=False)
    print("Loaded environment variables from .env")


class AgentState(TypedDict):
    messages: list[BaseMessage]
    next_step: str | None
    high_level_plan: list | None
    current_step_index: int | None
    plan_complete: bool | None
    relevant_tool_nodes: list | None
    interrogated_tool_info: dict | None
    generated_code_snippet: str | None
    snippet_ok: bool | None
    full_code: str | None
    execution_ok: bool | None


class E1:
    def __init__(
        self,
        path="./data",
        llm="claude-sonnet-4-20250514",
        use_tool_retriever=True,
        timeout_seconds=600,
        base_url: str | None = None,
        api_key: str = "EMPTY",
        custom_model_name: str = "",
    ):
        """Initialize the biomni agent.

        Args:
            path: Path to the data
            llm: LLM to use for the agent
            use_tool_retriever: If True, use a tool retriever
            timeout_seconds: Timeout for code execution in seconds
            base_url: Base URL for custom model serving (e.g., "http://localhost:8000/v1")
            api_key: API key for the custom LLM

        """
        self.path = path

        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")

        # --- Begin custom folder/file checks ---
        benchmark_dir = os.path.join(path, "biomni_data", "benchmark")
        data_lake_dir = os.path.join(path, "biomni_data", "data_lake")

        # Create the biomni_data directory structure
        os.makedirs(benchmark_dir, exist_ok=True)
        os.makedirs(data_lake_dir, exist_ok=True)

        expected_data_lake_files = list(data_lake_dict.keys())

        # Check and download missing data lake files
        print("Checking and downloading missing data lake files...")
        check_and_download_s3_files(
            s3_bucket_url="https://biomni-release.s3.amazonaws.com",
            local_data_lake_path=data_lake_dir,
            expected_files=expected_data_lake_files,
            folder="data_lake",
        )

        # Check if benchmark directory structure is complete
        benchmark_ok = False
        if os.path.isdir(benchmark_dir):
            patient_gene_detection_dir = os.path.join(benchmark_dir, "hle")
            if os.path.isdir(patient_gene_detection_dir):
                benchmark_ok = True

        if not benchmark_ok:
            print("Checking and downloading benchmark files...")
            check_and_download_s3_files(
                s3_bucket_url="https://biomni-release.s3.amazonaws.com",
                local_data_lake_path=benchmark_dir,
                expected_files=[],  # Empty list - will download entire folder
                folder="benchmark",
            )

        self.path = os.path.join(path, "biomni_data")
        module2api = read_module2api()

        self.llm = get_llm(llm, stop_sequences=["</execute>", "</solution>"], base_url=base_url, api_key=api_key, custom_model_name=custom_model_name)
        self.module2api = module2api
        self.use_tool_retriever = use_tool_retriever

        if self.use_tool_retriever:
            self.tool_registry = ToolRegistry(module2api)
            self.retriever = ToolRetriever()

        # Add timeout parameter
        self.timeout_seconds = timeout_seconds  # 10 minutes default timeout
        
        # Enhanced Tool Knowledge Graph for TAG model
        self.tool_knowledge_graph = {}
        self._build_tool_knowledge_graph()
        
        self.configure()

    def _build_tool_knowledge_graph(self):
        """Build an enhanced tool knowledge graph with examples and dependencies."""
        if not hasattr(self, "module2api") or not self.module2api:
            return
            
        for module_name, tools in self.module2api.items():
            for tool in tools:
                tool_name = tool.get("name", "")
                if tool_name:
                    self.tool_knowledge_graph[tool_name] = {
                        "name": tool_name,
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {}),
                        "module": module_name,
                        "examples": [
                            f"from {module_name} import {tool_name}",
                            f"result = {tool_name}()"
                        ],
                        "full_docstring": tool.get("description", ""),
                        "dependencies": [],
                        "common_errors": []
                    }

    def count_tokens(self, text):
        """Simple token counting approximation (roughly 4 characters = 1 token)."""
        if isinstance(text, str):
            return len(text) // 4
        return 0
    
    def print_token_info(self, messages, stage=""):
        """Print token information for debugging."""
        system_tokens = 0
        user_tokens = 0
        total_tokens = 0
        
        for msg in messages:
            content = msg.content if hasattr(msg, 'content') else str(msg)
            tokens = self.count_tokens(content)
            total_tokens += tokens
            
            if isinstance(msg, SystemMessage):
                system_tokens += tokens
            elif isinstance(msg, HumanMessage):
                user_tokens += tokens
        
        print(f"\n=== TOKEN TRACKING {stage} ===")
        print(f"System prompt tokens: {system_tokens:,}")
        print(f"User messages tokens: {user_tokens:,}")
        print(f"Total input tokens: {total_tokens:,}")
        print(f"================================\n")

    # Include all the helper methods (shortened for space)
    def add_tool(self, api):
        """Add a new tool to the agent's tool registry and make it available for retrieval."""
        try:
            function_code = inspect.getsource(api)
            module_name = api.__module__ if hasattr(api, "__module__") else "custom_tools"
            function_name = api.__name__ if hasattr(api, "__name__") else str(api)
            schema = function_to_api_schema(function_code, self.llm)
            
            if not isinstance(schema, dict):
                raise ValueError("Generated schema is not a dictionary")
            
            if "name" not in schema:
                schema["name"] = function_name
            if "description" not in schema:
                schema["description"] = f"Custom tool: {function_name}"
            if "required_parameters" not in schema:
                schema["required_parameters"] = []

            schema["module"] = module_name

            if hasattr(self, "tool_registry") and self.tool_registry is not None:
                try:
                    self.tool_registry.register_tool(schema)
                    print(f"Successfully registered tool '{schema['name']}' in tool registry")
                except Exception as e:
                    print(f"Warning: Failed to register tool in registry: {e}")

            if not hasattr(self, "module2api") or self.module2api is None:
                self.module2api = {}

            if module_name not in self.module2api:
                self.module2api[module_name] = []

            existing_tool = None
            for existing in self.module2api[module_name]:
                if existing.get("name") == schema["name"]:
                    existing_tool = existing
                    break

            if existing_tool:
                existing_tool.update(schema)
                print(f"Updated existing tool '{schema['name']}' in module '{module_name}'")
            else:
                self.module2api[module_name].append(schema)
                print(f"Added new tool '{schema['name']}' to module '{module_name}'")

            # Update tool knowledge graph
            self.tool_knowledge_graph[schema["name"]] = {
                "name": schema["name"],
                "description": schema["description"],
                "parameters": schema.get("parameters", {}),
                "module": module_name,
                "examples": [
                    f"from {module_name} import {schema['name']}",
                    f"result = {schema['name']}()"
                ],
                "full_docstring": schema["description"],
                "dependencies": [],
                "common_errors": []
            }

            if not hasattr(self, "_custom_functions"):
                self._custom_functions = {}
            self._custom_functions[schema["name"]] = api

            if not hasattr(self, "_custom_tools"):
                self._custom_tools = {}
            self._custom_tools[schema["name"]] = {
                "name": schema["name"],
                "description": schema["description"],
                "module": module_name,
            }

            import builtins
            if not hasattr(builtins, "_biomni_custom_functions"):
                builtins._biomni_custom_functions = {}
            builtins._biomni_custom_functions[schema["name"]] = api

            print(f"Tool '{schema['name']}' successfully added and ready for use in both direct execution and retrieval")
            self.configure()
            return schema

        except Exception as e:
            print(f"Error adding tool: {e}")
            import traceback
            traceback.print_exc()
            raise

    def get_custom_tool(self, name):
        if hasattr(self, "_custom_functions") and name in self._custom_functions:
            return self._custom_functions[name]
        return None

    def list_custom_tools(self):
        if hasattr(self, "_custom_functions"):
            return list(self._custom_functions.keys())
        return []

    def remove_custom_tool(self, name):
        removed = False
        if hasattr(self, "_custom_functions") and name in self._custom_functions:
            del self._custom_functions[name]
            removed = True
        if hasattr(self, "_custom_tools") and name in self._custom_tools:
            del self._custom_tools[name]
            removed = True
        if name in self.tool_knowledge_graph:
            del self.tool_knowledge_graph[name]
            removed = True
        # Additional cleanup logic...
        return removed

    def add_data(self, data):
        try:
            if not isinstance(data, dict):
                raise ValueError("Data must be a dictionary with file path as key and description as value")
            if not hasattr(self, "_custom_data"):
                self._custom_data = {}
            for file_path, description in data.items():
                if not isinstance(file_path, str) or not isinstance(description, str):
                    continue
                filename = os.path.basename(file_path) if "/" in file_path else file_path
                self._custom_data[filename] = {"path": file_path, "description": description}
                self.data_lake_dict[filename] = description
            self.configure()
            return True
        except Exception:
            return False

    def get_custom_data(self, name):
        if hasattr(self, "_custom_data") and name in self._custom_data:
            return self._custom_data[name]
        return None

    def list_custom_data(self):
        if hasattr(self, "_custom_data"):
            return [(name, info["description"]) for name, info in self._custom_data.items()]
        return []

    def remove_custom_data(self, name):
        removed = False
        if hasattr(self, "_custom_data") and name in self._custom_data:
            del self._custom_data[name]
            removed = True
        if hasattr(self, "data_lake_dict") and name in self.data_lake_dict:
            del self.data_lake_dict[name]
            removed = True
        return removed

    def add_software(self, software):
        try:
            if not isinstance(software, dict):
                raise ValueError("Software must be a dictionary with software name as key and description as value")
            if not hasattr(self, "_custom_software"):
                self._custom_software = {}
            for software_name, description in software.items():
                if not isinstance(software_name, str) or not isinstance(description, str):
                    continue
                self._custom_software[software_name] = {"name": software_name, "description": description}
                self.library_content_dict[software_name] = description
            self.configure()
            return True
        except Exception:
            return False

    def get_custom_software(self, name):
        if hasattr(self, "_custom_software") and name in self._custom_software:
            return self._custom_software[name]
        return None

    def list_custom_software(self):
        if hasattr(self, "_custom_software"):
            return [(name, info["description"]) for name, info in self._custom_software.items()]
        return []

    def remove_custom_software(self, name):
        removed = False
        if hasattr(self, "_custom_software") and name in self._custom_software:
            del self._custom_software[name]
            removed = True
        if hasattr(self, "library_content_dict") and name in self.library_content_dict:
            del self.library_content_dict[name]
            removed = True
        return removed

    def _generate_system_prompt(
        self,
        tool_desc,
        data_lake_content,
        library_content_list,
        self_critic=False,
        is_retrieval=False,
        custom_tools=None,
        custom_data=None,
        custom_software=None,
    ):
        """Generate the system prompt based on the provided resources."""

        def format_item_with_description(name, description):
            if not description:
                description = f"Data lake item: {name}"
            if isinstance(name, str) and ": " in name:
                return name
            max_line_length = 80
            if len(description) > max_line_length:
                wrapped_desc = []
                words = description.split()
                current_line = ""
                for word in words:
                    if len(current_line) + len(word) + 1 <= max_line_length:
                        if current_line:
                            current_line += " " + word
                        else:
                            current_line = word
                    else:
                        wrapped_desc.append(current_line)
                        current_line = word
                if current_line:
                    wrapped_desc.append(current_line)
                formatted_desc = f"{name}:\n  " + "\n  ".join(wrapped_desc)
                return formatted_desc
            else:
                return f"{name}: {description}"

        # Base prompt
        prompt_modifier = """
You are a helpful biomedical assistant assigned with the task of problem-solving.
To achieve this, you will be using an interactive coding environment equipped with a variety of tool functions, data, and softwares to assist you throughout the process.

You have access to special tags for advanced tool interaction:
- <interrogate_tool tool_name="tool_name">: Get detailed information about a specific tool
- <sandbox_code>: Test a small code snippet in isolation before full execution
- <execute>: Execute the main code
- <solution>: Provide the final answer

Given a task, make a plan first. The plan should be a numbered list of steps that you will take to solve the task. Be specific and detailed.
Format your plan as a checklist with empty checkboxes like this:
1. [ ] First step
2. [ ] Second step
3. [ ] Third step

Follow the plan step by step. After completing each step, update the checklist by replacing the empty checkbox with a checkmark:
1. [✓] First step (completed)
2. [ ] Second step
3. [ ] Third step

You can interrogate tools to understand them better before using them:
<interrogate_tool tool_name="query_pubmed">

You can test small code snippets in a sandbox before full execution:
<sandbox_code>
from biomni.tool.query_tools import query_pubmed
result = query_pubmed("CFTR gene")
print(type(result))
</sandbox_code>

Then execute your main code:
<execute>
from biomni.tool.query_tools import query_pubmed
result = query_pubmed("CFTR gene mutations")
print(result)
</execute>

When ready, provide your solution:
<solution>Your answer here</solution>

In each response, you must include ONE of: <interrogate_tool>, <sandbox_code>, <execute>, or <solution>. Do not respond with messages without any tags.
"""

        # Add environment resources
        prompt_modifier += """

Environment Resources:

- Function Dictionary:
{function_intro}
---
{tool_desc}
---

{import_instruction}

- Biological data lake
You can access a biological data lake at the following path: {data_lake_path}.
{data_lake_intro}
Each item is listed with its description to help you understand its contents.
----
{data_lake_content}
----

- Software Library:
{library_intro}
Each library is listed with its description to help you understand its functionality.
----
{library_content_formatted}
----

- Note on using R packages and Bash scripts:
  - R packages: Use subprocess.run(['Rscript', '-e', 'your R code here']) in Python, or use the #!R marker in your execute block.
  - Bash scripts and commands: Use the #!BASH marker in your execute block for both simple commands and complex shell scripts with variables, loops, conditionals, etc.
        """

        # Set appropriate text based on whether this is initial configuration or after retrieval
        if is_retrieval:
            function_intro = "Based on your query, I've identified the following most relevant functions that you can use in your code:"
            data_lake_intro = "Based on your query, I've identified the following most relevant datasets:"
            library_intro = "Based on your query, I've identified the following most relevant libraries that you can use:"
            import_instruction = "IMPORTANT: When using any function, you MUST first import it from its module. For example:\nfrom [module_name] import [function_name]"
        else:
            function_intro = "In your code, you will need to import the function location using the following dictionary of functions:"
            data_lake_intro = "You can write code to understand the data, process and utilize it for the task. Here is the list of datasets:"
            library_intro = "The environment supports a list of libraries that can be directly used. Do not forget the import statement:"
            import_instruction = ""

        # Format content
        data_lake_formatted = []
        for item in data_lake_content:
            if isinstance(item, dict):
                name = item.get("name", "")
                description = self.data_lake_dict.get(name, f"Data lake item: {name}")
                data_lake_formatted.append(format_item_with_description(name, description))
            else:
                description = self.data_lake_dict.get(item, f"Data lake item: {item}")
                data_lake_formatted.append(format_item_with_description(item, description))

        libraries_formatted = []
        for lib in library_content_list:
            if isinstance(lib, dict):
                name = lib.get("name", "")
                description = self.library_content_dict.get(name, f"Software library: {name}")
                libraries_formatted.append(format_item_with_description(name, description))
            else:
                description = self.library_content_dict.get(lib, f"Software library: {lib}")
                libraries_formatted.append(format_item_with_description(lib, description))

        library_content_formatted = "\n".join(libraries_formatted)
        data_lake_content_formatted = "\n".join(data_lake_formatted)

        # Format the prompt with the appropriate values
        format_dict = {
            "function_intro": function_intro,
            "tool_desc": textify_api_dict(tool_desc) if isinstance(tool_desc, dict) else tool_desc,
            "import_instruction": import_instruction,
            "data_lake_path": self.path + "/data_lake",
            "data_lake_intro": data_lake_intro,
            "data_lake_content": data_lake_content_formatted,
            "library_intro": library_intro,
            "library_content_formatted": library_content_formatted,
        }

        formatted_prompt = prompt_modifier.format(**format_dict)
        return formatted_prompt

    def configure(self, self_critic=False, test_time_scale_round=0):
        """Configure the agent with the Tool-Augmented Graph (TAG) Model workflow.

        Args:
            self_critic: Whether to enable self-critic mode
            test_time_scale_round: Number of rounds for test time scaling

        """
        # Store self_critic for later use
        self.self_critic = self_critic

        # Get data lake content
        data_lake_path = self.path + "/data_lake"
        data_lake_content = glob.glob(data_lake_path + "/*")
        data_lake_items = [x.split("/")[-1] for x in data_lake_content]

        # Store data_lake_dict as instance variable for use in retrieval
        self.data_lake_dict = data_lake_dict
        # Store library_content_dict directly without library_content
        self.library_content_dict = library_content_dict

        # Prepare tool descriptions
        tool_desc = {i: [x for x in j if x["name"] != "run_python_repl"] for i, j in self.module2api.items()}

        # Prepare data lake items with descriptions
        data_lake_with_desc = []
        for item in data_lake_items:
            description = self.data_lake_dict.get(item, f"Data lake item: {item}")
            data_lake_with_desc.append({"name": item, "description": description})

        # Add custom data items if they exist
        if hasattr(self, "_custom_data") and self._custom_data:
            for name, info in self._custom_data.items():
                data_lake_with_desc.append({"name": name, "description": info["description"]})

        # Prepare library content list including custom software
        library_content_list = list(self.library_content_dict.keys())
        if hasattr(self, "_custom_software") and self._custom_software:
            for name in self._custom_software:
                if name not in library_content_list:  # Avoid duplicates
                    library_content_list.append(name)

        # Generate the system prompt for initial configuration (is_retrieval=False)
        # Prepare custom resources for highlighting
        custom_tools = []
        if hasattr(self, "_custom_tools") and self._custom_tools:
            for name, info in self._custom_tools.items():
                custom_tools.append({
                    "name": name,
                    "description": info["description"],
                    "module": info["module"],
                })

        custom_data = []
        if hasattr(self, "_custom_data") and self._custom_data:
            for name, info in self._custom_data.items():
                custom_data.append({"name": name, "description": info["description"]})

        custom_software = []
        if hasattr(self, "_custom_software") and self._custom_software:
            for name, info in self._custom_software.items():
                custom_software.append({"name": name, "description": info["description"]})

        self.system_prompt = self._generate_system_prompt(
            tool_desc=tool_desc,
            data_lake_content=data_lake_with_desc,
            library_content_list=library_content_list,
            self_critic=self_critic,
            is_retrieval=False,
            custom_tools=custom_tools if custom_tools else None,
            custom_data=custom_data if custom_data else None,
            custom_software=custom_software if custom_software else None,
        )
        
        # Track system prompt size
        system_prompt_tokens = self.count_tokens(self.system_prompt)
        print(f"\n=== SYSTEM PROMPT GENERATED (INITIAL) ===")
        print(f"System prompt length: {len(self.system_prompt):,} characters")
        print(f"Estimated tokens: {system_prompt_tokens:,}")
        print(f"=========================================\n")

        # Architecture 4: Tool-Augmented Graph (TAG) Model - Define the nodes
        def planner_generate_high_level_plan(state: AgentState) -> AgentState:
            messages = [SystemMessage(content=self.system_prompt)] + state["messages"]
            
            # Track and print token usage before sending to LLM
            self.print_token_info(messages, "BEFORE LLM INVOKE")
            
            response = self.llm.invoke(messages)

            # Parse the response
            msg = str(response.content)

            # Check for incomplete tags and fix them
            if "<interrogate_tool" in msg and ">" not in msg[msg.find("<interrogate_tool"):]:
                msg += ">"
            if "<sandbox_code>" in msg and "</sandbox_code>" not in msg:
                msg += "</sandbox_code>"
            if "<execute>" in msg and "</execute>" not in msg:
                msg += "</execute>"
            if "<solution>" in msg and "</solution>" not in msg:
                msg += "</solution>"

            interrogate_match = re.search(r"<interrogate_tool tool_name=\"([^\"]+)\">", msg)
            sandbox_match = re.search(r"<sandbox_code>(.*?)</sandbox_code>", msg, re.DOTALL)
            execute_match = re.search(r"<execute>(.*?)</execute>", msg, re.DOTALL)
            answer_match = re.search(r"<solution>(.*?)</solution>", msg, re.DOTALL)

            # Add the message to the state before checking for errors
            state["messages"].append(AIMessage(content=msg.strip()))

            if answer_match:
                state["next_step"] = "end"
            elif interrogate_match:
                # Extract tool name for interrogation
                tool_name = interrogate_match.group(1)
                state["relevant_tool_nodes"] = [tool_name]
                state["next_step"] = "interrogator_query_tool_node"
            elif sandbox_match:
                state["generated_code_snippet"] = sandbox_match.group(1)
                state["next_step"] = "sandbox_test_generated_code_snippet"
            elif execute_match:
                state["full_code"] = execute_match.group(1)
                state["next_step"] = "executor_run_full_code"
            else:
                # Create a simple high-level plan
                state["high_level_plan"] = ["step1", "step2", "step3"]
                state["current_step_index"] = 0
                state["plan_complete"] = False
                state["next_step"] = "retriever_find_relevant_tool_nodes"

            return state

        def retriever_find_relevant_tool_nodes(state: AgentState) -> AgentState:
            # Find relevant tools based on the current step
            user_prompt = state["messages"][0].content if state["messages"] else ""
            
            # Simple keyword-based tool retrieval
            relevant_tools = []
            if "query" in user_prompt.lower() or "search" in user_prompt.lower():
                relevant_tools.extend(["query_pubmed", "query_ensembl", "search_google"])
            if "gene" in user_prompt.lower() or "protein" in user_prompt.lower():
                relevant_tools.extend(["query_uniprot", "query_ensembl"])
            if "analysis" in user_prompt.lower():
                relevant_tools.extend(["understand_scRNA", "analyze_data"])
            
            # Default tools if none found
            if not relevant_tools:
                relevant_tools = list(self.tool_knowledge_graph.keys())[:3]
            
            state["relevant_tool_nodes"] = relevant_tools[:2]  # Limit to 2 tools
            state["next_step"] = "interrogator_query_tool_node"
            return state

        def interrogator_query_tool_node(state: AgentState) -> AgentState:
            # Get detailed information about the first relevant tool
            relevant_tools = state.get("relevant_tool_nodes", [])
            if relevant_tools:
                tool_name = relevant_tools[0]
                tool_info = self.tool_knowledge_graph.get(tool_name, {
                    "name": tool_name,
                    "description": "Tool information not available",
                    "examples": [f"# Example usage of {tool_name}"],
                    "full_docstring": "No detailed documentation available"
                })
                
                state["interrogated_tool_info"] = tool_info
                
                # Add tool information as observation
                tool_details = f"""
Tool Information for {tool_name}:
Description: {tool_info.get('description', 'N/A')}
Module: {tool_info.get('module', 'N/A')}
Examples:
{chr(10).join(tool_info.get('examples', []))}
Full Documentation: {tool_info.get('full_docstring', 'N/A')}
"""
                observation = f"\n<observation>{tool_details}</observation>"
                state["messages"].append(AIMessage(content=observation.strip()))
            
            state["next_step"] = "generate_code_with_examples"
            return state

        def generate_code_with_examples(state: AgentState) -> AgentState:
            # Generate code using the interrogated tool information
            messages = [SystemMessage(content=self.system_prompt)] + state["messages"]
            
            response = self.llm.invoke(messages)
            msg = str(response.content)

            # Check for incomplete tags and fix them
            if "<sandbox_code>" in msg and "</sandbox_code>" not in msg:
                msg += "</sandbox_code>"
            if "<execute>" in msg and "</execute>" not in msg:
                msg += "</execute>"
            if "<solution>" in msg and "</solution>" not in msg:
                msg += "</solution>"

            sandbox_match = re.search(r"<sandbox_code>(.*?)</sandbox_code>", msg, re.DOTALL)
            execute_match = re.search(r"<execute>(.*?)</execute>", msg, re.DOTALL)
            answer_match = re.search(r"<solution>(.*?)</solution>", msg, re.DOTALL)

            state["messages"].append(AIMessage(content=msg.strip()))

            if answer_match:
                state["next_step"] = "end"
            elif sandbox_match:
                state["generated_code_snippet"] = sandbox_match.group(1)
                state["next_step"] = "sandbox_test_generated_code_snippet"
            elif execute_match:
                state["full_code"] = execute_match.group(1)
                state["next_step"] = "executor_run_full_code"
            else:
                state["next_step"] = "generate_code_with_examples"

            return state

        def sandbox_test_generated_code_snippet(state: AgentState) -> AgentState:
            # Test the generated code snippet in a safe environment
            code_snippet = state.get("generated_code_snippet", "")
            
            if code_snippet:
                timeout = 30  # Short timeout for snippet testing
                
                try:
                    # Inject custom functions into the Python execution environment
                    self._inject_custom_functions_to_repl()
                    result = run_with_timeout(run_python_repl, [code_snippet], timeout=timeout)
                    
                    # Check if snippet execution was successful
                    if "Error" in result or "Exception" in result or "Traceback" in result:
                        state["snippet_ok"] = False
                        state["next_step"] = "critic_analyze_snippet_error"
                    else:
                        state["snippet_ok"] = True
                        state["next_step"] = "executor_run_full_code"
                    
                    observation = f"\n<observation>Snippet Test Result:\n{result}</observation>"
                    state["messages"].append(AIMessage(content=observation.strip()))
                
                except Exception as e:
                    state["snippet_ok"] = False
                    state["next_step"] = "critic_analyze_snippet_error"
                    observation = f"\n<observation>Snippet Test Failed: {str(e)}</observation>"
                    state["messages"].append(AIMessage(content=observation.strip()))
            
            return state

        def critic_analyze_snippet_error(state: AgentState) -> AgentState:
            # Analyze the snippet error and try to fix it
            error_msg = "The code snippet failed. Let me analyze the error and generate a better version."
            state["messages"].append(HumanMessage(content=error_msg))
            state["next_step"] = "generate_code_with_examples"
            return state

        def executor_run_full_code(state: AgentState) -> AgentState:
            # Execute the full code
            code = state.get("full_code", state.get("generated_code_snippet", ""))
            
            if code:
                timeout = self.timeout_seconds

                # Check if the code is R code
                if (code.strip().startswith("#!R") or code.strip().startswith("# R code") or code.strip().startswith("# R script")):
                    r_code = re.sub(r"^#!R|^# R code|^# R script", "", code, 1).strip()
                    result = run_with_timeout(run_r_code, [r_code], timeout=timeout)
                # Check if the code is a Bash script or CLI command
                elif (code.strip().startswith("#!BASH") or code.strip().startswith("# Bash script") or code.strip().startswith("#!CLI")):
                    if code.strip().startswith("#!CLI"):
                        cli_command = re.sub(r"^#!CLI", "", code, 1).strip()
                        cli_command = cli_command.replace("\n", " ")
                        result = run_with_timeout(run_bash_script, [cli_command], timeout=timeout)
                    else:
                        bash_script = re.sub(r"^#!BASH|^# Bash script", "", code, 1).strip()
                        result = run_with_timeout(run_bash_script, [bash_script], timeout=timeout)
                else:
                    # Inject custom functions into the Python execution environment
                    self._inject_custom_functions_to_repl()
                    result = run_with_timeout(run_python_repl, [code], timeout=timeout)

                if len(result) > 10000:
                    result = "The output is too long to be added to context. Here are the first 10K characters...\n" + result[:10000]
                
                # Check if execution was successful
                if "Error" in result or "Exception" in result or "Traceback" in result:
                    state["execution_ok"] = False
                    state["next_step"] = "critic_analyze_full_error"
                else:
                    state["execution_ok"] = True
                    state["next_step"] = "end"
                
                observation = f"\n<observation>{result}</observation>"
                state["messages"].append(AIMessage(content=observation.strip()))

            return state

        def critic_analyze_full_error(state: AgentState) -> AgentState:
            # Analyze the full execution error
            error_msg = "The full code execution failed. Let me analyze the error and find relevant tool nodes to try a different approach."
            state["messages"].append(HumanMessage(content=error_msg))
            state["next_step"] = "retriever_find_relevant_tool_nodes"
            return state

        def routing_function(
            state: AgentState,
        ) -> Literal["planner_generate_high_level_plan", "retriever_find_relevant_tool_nodes", "interrogator_query_tool_node", "generate_code_with_examples", "sandbox_test_generated_code_snippet", "critic_analyze_snippet_error", "executor_run_full_code", "critic_analyze_full_error", "end"]:
            next_step = state.get("next_step")
            return next_step

        # Create the workflow
        workflow = StateGraph(AgentState)

        # Add nodes for Architecture 4: Tool-Augmented Graph (TAG) Model
        workflow.add_node("planner_generate_high_level_plan", planner_generate_high_level_plan)
        workflow.add_node("retriever_find_relevant_tool_nodes", retriever_find_relevant_tool_nodes)
        workflow.add_node("interrogator_query_tool_node", interrogator_query_tool_node)
        workflow.add_node("generate_code_with_examples", generate_code_with_examples)
        workflow.add_node("sandbox_test_generated_code_snippet", sandbox_test_generated_code_snippet)
        workflow.add_node("critic_analyze_snippet_error", critic_analyze_snippet_error)
        workflow.add_node("executor_run_full_code", executor_run_full_code)
        workflow.add_node("critic_analyze_full_error", critic_analyze_full_error)

        # Add conditional edges
        workflow.add_conditional_edges(
            "planner_generate_high_level_plan",
            routing_function,
            path_map={
                "retriever_find_relevant_tool_nodes": "retriever_find_relevant_tool_nodes",
                "interrogator_query_tool_node": "interrogator_query_tool_node",
                "sandbox_test_generated_code_snippet": "sandbox_test_generated_code_snippet",
                "executor_run_full_code": "executor_run_full_code",
                "end": END,
            },
        )
        
        workflow.add_conditional_edges(
            "retriever_find_relevant_tool_nodes",
            routing_function,
            path_map={"interrogator_query_tool_node": "interrogator_query_tool_node"},
        )
        
        workflow.add_conditional_edges(
            "interrogator_query_tool_node",
            routing_function,
            path_map={"generate_code_with_examples": "generate_code_with_examples"},
        )
        
        workflow.add_conditional_edges(
            "generate_code_with_examples",
            routing_function,
            path_map={
                "sandbox_test_generated_code_snippet": "sandbox_test_generated_code_snippet",
                "executor_run_full_code": "executor_run_full_code",
                "generate_code_with_examples": "generate_code_with_examples",
                "end": END,
            },
        )
        
        workflow.add_conditional_edges(
            "sandbox_test_generated_code_snippet",
            routing_function,
            path_map={
                "critic_analyze_snippet_error": "critic_analyze_snippet_error",
                "executor_run_full_code": "executor_run_full_code",
            },
        )
        
        workflow.add_conditional_edges(
            "critic_analyze_snippet_error",
            routing_function,
            path_map={"generate_code_with_examples": "generate_code_with_examples"},
        )
        
        workflow.add_conditional_edges(
            "executor_run_full_code",
            routing_function,
            path_map={
                "critic_analyze_full_error": "critic_analyze_full_error",
                "end": END,
            },
        )
        
        workflow.add_conditional_edges(
            "critic_analyze_full_error",
            routing_function,
            path_map={"retriever_find_relevant_tool_nodes": "retriever_find_relevant_tool_nodes"},
        )

        workflow.add_edge(START, "planner_generate_high_level_plan")

        # Compile the workflow
        self.app = workflow.compile()
        self.checkpointer = MemorySaver()
        self.app.checkpointer = self.checkpointer
        graph_png = self.app.get_graph().draw_mermaid_png()
        with open('workflow_graph_e1_edited.png', 'wb') as f:
            f.write(graph_png)
        display(Image(graph_png))

    def go(self, prompt):
        """Execute the agent with the given prompt."""
        self.critic_count = 0
        self.user_task = prompt

        if self.use_tool_retriever:
            # Gather all available resources
            all_tools = self.tool_registry.tools if hasattr(self, "tool_registry") else []
            data_lake_path = self.path + "/data_lake"
            data_lake_content = glob.glob(data_lake_path + "/*")
            data_lake_items = [x.split("/")[-1] for x in data_lake_content]

            data_lake_descriptions = []
            for item in data_lake_items:
                description = self.data_lake_dict.get(item, f"Data lake item: {item}")
                data_lake_descriptions.append({"name": item, "description": description})

            if hasattr(self, "_custom_data") and self._custom_data:
                for name, info in self._custom_data.items():
                    data_lake_descriptions.append({"name": name, "description": info["description"]})

            library_descriptions = []
            for lib_name, lib_desc in self.library_content_dict.items():
                library_descriptions.append({"name": lib_name, "description": lib_desc})

            if hasattr(self, "_custom_software") and self._custom_software:
                for name, info in self._custom_software.items():
                    if not any(lib["name"] == name for lib in library_descriptions):
                        library_descriptions.append({"name": name, "description": info["description"]})

            resources = {
                "tools": all_tools,
                "data_lake": data_lake_descriptions,
                "libraries": library_descriptions,
            }

            selected_resources = self.retriever.prompt_based_retrieval(prompt, resources, llm=self.llm)
            print("Using prompt-based retrieval with the agent's LLM")

            selected_resources_names = {
                "tools": selected_resources["tools"],
                "data_lake": [],
                "libraries": [lib["name"] if isinstance(lib, dict) else lib for lib in selected_resources["libraries"]],
            }

            for item in selected_resources["data_lake"]:
                if isinstance(item, dict):
                    selected_resources_names["data_lake"].append(item["name"])
                elif isinstance(item, str) and ": " in item:
                    name = item.split(": ")[0]
                    selected_resources_names["data_lake"].append(name)
                else:
                    selected_resources_names["data_lake"].append(item)

            self.update_system_prompt_with_selected_resources(selected_resources_names)

        inputs = {"messages": [HumanMessage(content=prompt)], "next_step": None}
        config = {"recursion_limit": 500, "configurable": {"thread_id": 42}}
        self.log = []

        for s in self.app.stream(inputs, stream_mode="values", config=config):
            message = s["messages"][-1]
            out = pretty_print(message)
            self.log.append(out)

        return self.log, message.content

    def update_system_prompt_with_selected_resources(self, selected_resources):
        """Update the system prompt with the selected resources."""
        tool_desc = {}
        for tool in selected_resources["tools"]:
            if isinstance(tool, dict):
                module_name = tool.get("module", None)
                if not module_name and hasattr(self, "module2api"):
                    for mod, apis in self.module2api.items():
                        for api in apis:
                            if api.get("name") == tool.get("name"):
                                module_name = mod
                                tool["module"] = module_name
                                break
                        if module_name:
                            break
                if not module_name:
                    module_name = "biomni.tool.scRNA_tools"
                    tool["module"] = module_name
            else:
                module_name = getattr(tool, "module_name", None)
                if not module_name and hasattr(self, "module2api"):
                    tool_name = getattr(tool, "name", str(tool))
                    for mod, apis in self.module2api.items():
                        for api in apis:
                            if api.get("name") == tool_name:
                                module_name = mod
                                tool.module_name = module_name
                                break
                        if module_name:
                            break
                if not module_name:
                    module_name = "biomni.tool.scRNA_tools"
                    tool.module_name = module_name

            if module_name not in tool_desc:
                tool_desc[module_name] = []

            if isinstance(tool, dict):
                if "module" not in tool:
                    tool["module"] = module_name
                tool_desc[module_name].append(tool)
            else:
                tool_dict = {
                    "name": getattr(tool, "name", str(tool)),
                    "description": getattr(tool, "description", ""),
                    "parameters": getattr(tool, "parameters", {}),
                    "module": module_name,
                }
                tool_desc[module_name].append(tool_dict)

        data_lake_with_desc = []
        for item in selected_resources["data_lake"]:
            description = self.data_lake_dict.get(item, f"Data lake item: {item}")
            data_lake_with_desc.append({"name": item, "description": description})

        custom_tools = []
        if hasattr(self, "_custom_tools") and self._custom_tools:
            for name, info in self._custom_tools.items():
                custom_tools.append({
                    "name": name,
                    "description": info["description"],
                    "module": info["module"],
                })

        custom_data = []
        if hasattr(self, "_custom_data") and self._custom_data:
            for name, info in self._custom_data.items():
                custom_data.append({"name": name, "description": info["description"]})

        custom_software = []
        if hasattr(self, "_custom_software") and self._custom_software:
            for name, info in self._custom_software.items():
                custom_software.append({"name": name, "description": info["description"]})

        self.system_prompt = self._generate_system_prompt(
            tool_desc=tool_desc,
            data_lake_content=data_lake_with_desc,
            library_content_list=selected_resources["libraries"],
            self_critic=getattr(self, "self_critic", False),
            is_retrieval=True,
            custom_tools=custom_tools if custom_tools else None,
            custom_data=custom_data if custom_data else None,
            custom_software=custom_software if custom_software else None,
        )
        
        system_prompt_tokens = self.count_tokens(self.system_prompt)
        print(f"\n=== SYSTEM PROMPT UPDATED (AFTER RETRIEVAL) ===")
        print(f"System prompt length: {len(self.system_prompt):,} characters")
        print(f"Estimated tokens: {system_prompt_tokens:,}")
        print(f"===============================================\n")

    def result_formatting(self, output_class, task_intention):
        self.format_check_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are evaluateGPT, tasked with extract and parse the task output based on the history of an agent. "
                "Review the entire history of messages provided. "
                "Here is the task output requirement: \n"
                f"'{task_intention.replace('{', '{{').replace('}', '}}')}'.\n"
            )),
            ("placeholder", "{messages}"),
        ])

        checker_llm = self.format_check_prompt | self.llm.with_structured_output(output_class)
        result = checker_llm.invoke({"messages": [("user", str(self.log))]}).dict()
        return result

    def _inject_custom_functions_to_repl(self):
        """Inject custom functions into the Python REPL execution environment."""
        if hasattr(self, "_custom_functions") and self._custom_functions:
            from biomni.tool.support_tools import _persistent_namespace
            for name, func in self._custom_functions.items():
                _persistent_namespace[name] = func
            import builtins
            if not hasattr(builtins, "_biomni_custom_functions"):
                builtins._biomni_custom_functions = {}
            builtins._biomni_custom_functions.update(self._custom_functions) 