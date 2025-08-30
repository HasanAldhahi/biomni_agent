import glob
import inspect
import os
import re
import json
import time
import uuid
from typing import Literal, TypedDict, List, NotRequired, Any

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from IPython.display import Image, display

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
    """A comprehensive state dictionary that supports all implemented agent architectures."""
    messages: List[BaseMessage]
    next_step: str | None

    # --- For Hierarchical Expert ---
    benched_tools: NotRequired[dict[str, float]]
    """Stores {'tool_name': cooldown_timestamp} for failed tools."""

    # --- For Cognitive Corrector ---
    tool_intention: NotRequired[dict]
    """The agent's abstract intention (eg: {'tool': 'query_ensembl', 'prompt': '...'})."""
    corrected_code: NotRequired[str]
    """The corrected code to be executed after review by the corrector."""

    # --- For Tool-Augmented Graph (TAG) ---
    interrogation_result: NotRequired[str]
    sandboxed_code: NotRequired[str]


class F1:
    def __init__(
        self,
        path="./data",
        llm="claude-sonnet-4-20250514",
        use_tool_retriever=True,
        timeout_seconds=600,
        base_url: str | None = None,
        api_key: str = "EMPTY",
        custom_model_name: str = "",
        architecture: Literal[
            'baseline',
            'hierarchical_expert',
            'cognitive_corrector',
            'exploratory_sandbox',
            'tool_augmented_graph'
        ] = 'baseline',
    ):
        """Initialize the F1 biomni agent with advanced architectures and dynamic tool retrieval."""
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")

        # --- Data setup from A1 ---
        benchmark_dir = os.path.join(path, "biomni_data", "benchmark")
        data_lake_dir = os.path.join(path, "biomni_data", "data_lake")
        os.makedirs(benchmark_dir, exist_ok=True)
        os.makedirs(data_lake_dir, exist_ok=True)
        
        expected_data_lake_files = list(data_lake_dict.keys())
        print("Checking and downloading missing data lake files...")
        check_and_download_s3_files(
            s3_bucket_url="https://biomni-release.s3.amazonaws.com",
            local_data_lake_path=data_lake_dir,
            expected_files=expected_data_lake_files,
            folder="data_lake",
        )
        
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
                expected_files=[],
                folder="benchmark",
            )
        
        self.path = os.path.join(path, "biomni_data")
        module2api = read_module2api()

        self.llm = get_llm(llm, stop_sequences=["</execute>", "</solution>", "</intention>", "</interrogate_tool>", "</sandbox_code>"], base_url=base_url, api_key=api_key, custom_model_name=custom_model_name)
        self.module2api = module2api
        self.use_tool_retriever = use_tool_retriever

        if self.use_tool_retriever:
            self.tool_registry = ToolRegistry(module2api)
            self.retriever = ToolRetriever()

        self.tool_name_to_module_map = {
            tool['name']: mod
            for mod, tools in self.module2api.items()
            for tool in tools
        }
        self.timeout_seconds = timeout_seconds
        self.architecture = architecture
        print(f"Initializing F1 agent with '{self.architecture}' architecture.")
        self.configure()

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

    def _execute_code(self, code: str) -> str:
        """A centralized code execution function."""
        timeout = self.timeout_seconds
        
        # Check if the code is R code
        if (
            code.strip().startswith("#!R")
            or code.strip().startswith("# R code")
            or code.strip().startswith("# R script")
        ):
            # Remove the R marker and run as R code
            r_code = re.sub(r"^#!R|^# R code|^# R script", "", code, 1).strip()
            result = run_with_timeout(run_r_code, [r_code], timeout=timeout)
        # Check if the code is a Bash script or CLI command
        elif (
            code.strip().startswith("#!BASH")
            or code.strip().startswith("# Bash script")
            or code.strip().startswith("#!CLI")
        ):
            # Handle both Bash scripts and CLI commands with the same function
            if code.strip().startswith("#!CLI"):
                # For CLI commands, extract the command and run it as a simple bash script
                cli_command = re.sub(r"^#!CLI", "", code, 1).strip()
                # Remove any newlines to ensure it's a single command
                cli_command = cli_command.replace("\n", " ")
                result = run_with_timeout(run_bash_script, [cli_command], timeout=timeout)
            else:
                # For Bash scripts, remove the marker and run as a bash script
                bash_script = re.sub(r"^#!BASH|^# Bash script", "", code, 1).strip()
                result = run_with_timeout(run_bash_script, [bash_script], timeout=timeout)
        # Otherwise, run as Python code
        else:
            # Inject custom functions into the Python execution environment
            self._inject_custom_functions_to_repl()
            result = run_with_timeout(run_python_repl, [code], timeout=timeout)
        
        if len(result) > 10000:
            result = "The output is too long to be added to context. Here are the first 10K characters...\n" + result[:10000]
        return result

    def _parse_tool_call_from_code(self, code: str) -> str | None:
        """Extracts the first function call from a code string."""
        match = re.search(r"(\w+)\(", code)
        if match:
            return match.group(1)
        return None

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

    def add_tool(self, api):
        """Add a new tool to the agent's tool registry and make it available for retrieval."""
        try:
            # Get function information
            function_code = inspect.getsource(api)
            module_name = api.__module__ if hasattr(api, "__module__") else "custom_tools"
            function_name = api.__name__ if hasattr(api, "__name__") else str(api)

            # Generate API schema using the existing utility function
            schema = function_to_api_schema(function_code, self.llm)

            # Ensure the schema has all required fields for the tool registry
            if not isinstance(schema, dict):
                raise ValueError("Generated schema is not a dictionary")

            # Set default values if missing
            if "name" not in schema:
                schema["name"] = function_name
            if "description" not in schema:
                schema["description"] = f"Custom tool: {function_name}"
            if "required_parameters" not in schema:
                # Try to extract from parameters if available
                if "parameters" in schema and isinstance(schema["parameters"], dict):
                    required_params = []
                    params = schema["parameters"]
                    if "properties" in params:
                        for param_name in params["properties"]:
                            if param_name in params.get("required", []):
                                required_params.append(param_name)
                    schema["required_parameters"] = required_params
                else:
                    schema["required_parameters"] = []

            # Add module information to the schema
            schema["module"] = module_name

            # Add the tool to the tool registry if it exists
            if hasattr(self, "tool_registry") and self.tool_registry is not None:
                try:
                    self.tool_registry.register_tool(schema)
                    print(f"Successfully registered tool '{schema['name']}' in tool registry")
                except Exception as e:
                    print(f"Warning: Failed to register tool in registry: {e}")

            # Add the tool to module2api structure for system prompt generation
            if not hasattr(self, "module2api") or self.module2api is None:
                self.module2api = {}

            if module_name not in self.module2api:
                self.module2api[module_name] = []

            # Check if tool already exists in module2api to avoid duplicates
            existing_tool = None
            for existing in self.module2api[module_name]:
                if existing.get("name") == schema["name"]:
                    existing_tool = existing
                    break

            if existing_tool:
                # Update existing tool
                existing_tool.update(schema)
                print(f"Updated existing tool '{schema['name']}' in module '{module_name}'")
            else:
                # Add new tool
                self.module2api[module_name].append(schema)
                print(f"Added new tool '{schema['name']}' to module '{module_name}'")

            # Store the original function for potential future use
            if not hasattr(self, "_custom_functions"):
                self._custom_functions = {}
            self._custom_functions[schema["name"]] = api

            # Also store in _custom_tools for highlighting
            if not hasattr(self, "_custom_tools"):
                self._custom_tools = {}
            self._custom_tools[schema["name"]] = {
                "name": schema["name"],
                "description": schema["description"],
                "module": module_name,
            }

            # Make the function available in the global namespace for execution
            import builtins
            if not hasattr(builtins, "_biomni_custom_functions"):
                builtins._biomni_custom_functions = {}
            builtins._biomni_custom_functions[schema["name"]] = api

            print(f"Tool '{schema['name']}' successfully added and ready for use")
            self.configure()
            return schema

        except Exception as e:
            print(f"Error adding tool: {e}")
            import traceback
            traceback.print_exc()
            raise

    def get_custom_tool(self, name):
        """Get a custom tool by name."""
        if hasattr(self, "_custom_functions") and name in self._custom_functions:
            return self._custom_functions[name]
        return None

    def list_custom_tools(self):
        """List all custom tools that have been added."""
        if hasattr(self, "_custom_functions"):
            return list(self._custom_functions.keys())
        return []

    def remove_custom_tool(self, name):
        """Remove a custom tool."""
        removed = False

        # Remove from custom functions
        if hasattr(self, "_custom_functions") and name in self._custom_functions:
            del self._custom_functions[name]
            removed = True

        # Remove from custom tools (for highlighting)
        if hasattr(self, "_custom_tools") and name in self._custom_tools:
            del self._custom_tools[name]
            removed = True

        # Remove from global namespace
        import builtins
        if hasattr(builtins, "_biomni_custom_functions") and name in builtins._biomni_custom_functions:
            del builtins._biomni_custom_functions[name]

        # Remove from tool registry
        if hasattr(self, "tool_registry") and self.tool_registry is not None:
            if self.tool_registry.remove_tool_by_name(name):
                removed = True

        # Remove from module2api
        if hasattr(self, "module2api"):
            for tools in self.module2api.values():
                for i, tool in enumerate(tools):
                    if tool.get("name") == name:
                        del tools[i]
                        removed = True
                        break

        if removed:
            print(f"Custom tool '{name}' has been removed")
        else:
            print(f"Custom tool '{name}' was not found")

        return removed

    def add_data(self, data):
        """Add new data to the data lake."""
        try:
            if not isinstance(data, dict):
                raise ValueError("Data must be a dictionary with file path as key and description as value")

            # Initialize custom data storage if it doesn't exist
            if not hasattr(self, "_custom_data"):
                self._custom_data = {}

            # Add each data item
            for file_path, description in data.items():
                if not isinstance(file_path, str) or not isinstance(description, str):
                    print("Warning: Skipping invalid data entry - file_path and description must be strings")
                    continue

                # Extract filename from path for storage
                filename = os.path.basename(file_path) if "/" in file_path else file_path

                # Store the data with both the full path and description
                self._custom_data[filename] = {
                    "path": file_path,
                    "description": description,
                }

                # Also add to the data_lake_dict for consistency
                if not hasattr(self, "data_lake_dict"):
                    self.data_lake_dict = data_lake_dict.copy()
                self.data_lake_dict[filename] = description

                print(f"Added data item '{filename}': {description}")
            
            self.configure()
            print(f"Successfully added {len(data)} data item(s) to the data lake")
            return True

        except Exception as e:
            print(f"Error adding data: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_custom_data(self, name):
        """Get a custom data item by name."""
        if hasattr(self, "_custom_data") and name in self._custom_data:
            return self._custom_data[name]
        return None

    def list_custom_data(self):
        """List all custom data items that have been added."""
        if hasattr(self, "_custom_data"):
            return [(name, info["description"]) for name, info in self._custom_data.items()]
        return []

    def remove_custom_data(self, name):
        """Remove a custom data item."""
        removed = False

        # Remove from custom data
        if hasattr(self, "_custom_data") and name in self._custom_data:
            del self._custom_data[name]
            removed = True

        # Remove from data_lake_dict
        if hasattr(self, "data_lake_dict") and name in self.data_lake_dict:
            del self.data_lake_dict[name]
            removed = True

        if removed:
            print(f"Custom data item '{name}' has been removed")
        else:
            print(f"Custom data item '{name}' was not found")

        return removed

    def add_software(self, software):
        """Add new software to the software library."""
        try:
            if not isinstance(software, dict):
                raise ValueError("Software must be a dictionary with software name as key and description as value")

            # Initialize custom software storage if it doesn't exist
            if not hasattr(self, "_custom_software"):
                self._custom_software = {}

            # Add each software item
            for software_name, description in software.items():
                if not isinstance(software_name, str) or not isinstance(description, str):
                    print("Warning: Skipping invalid software entry - software_name and description must be strings")
                    continue

                # Store the software with description
                self._custom_software[software_name] = {
                    "name": software_name,
                    "description": description,
                }

                # Also add to the library_content_dict for consistency
                if not hasattr(self, "library_content_dict"):
                    self.library_content_dict = library_content_dict.copy()
                self.library_content_dict[software_name] = description

                print(f"Added software '{software_name}': {description}")

            print(f"Successfully added {len(software)} software item(s) to the library")
            self.configure()
            return True

        except Exception as e:
            print(f"Error adding software: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_custom_software(self, name):
        """Get a custom software item by name."""
        if hasattr(self, "_custom_software") and name in self._custom_software:
            return self._custom_software[name]
        return None

    def list_custom_software(self):
        """List all custom software items that have been added."""
        if hasattr(self, "_custom_software"):
            return [(name, info["description"]) for name, info in self._custom_software.items()]
        return []

    def remove_custom_software(self, name):
        """Remove a custom software item."""
        removed = False

        # Remove from custom software
        if hasattr(self, "_custom_software") and name in self._custom_software:
            del self._custom_software[name]
            removed = True

        # Remove from library_content_dict
        if hasattr(self, "library_content_dict") and name in self.library_content_dict:
            del self.library_content_dict[name]
            removed = True

        if removed:
            print(f"Custom software item '{name}' has been removed")
        else:
            print(f"Custom software item '{name}' was not found")

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
            """Format an item with its description in a readable way."""
            if not description:
                description = f"Data lake item: {name}"
            if isinstance(name, str) and ": " in name:
                return name
            return f"{name}: {description}"

        # Format the content consistently for both initial and retrieval cases
        if isinstance(data_lake_content, list):
            data_lake_formatted = []
            for item in data_lake_content:
                if isinstance(item, dict):
                    name = item.get("name", "")
                    description = item.get("description", f"Data lake item: {name}")
                    data_lake_formatted.append(format_item_with_description(name, description))
                else:
                    data_lake_formatted.append(str(item))
        else:
            data_lake_formatted = [str(data_lake_content)]

        library_content_formatted = "\n".join(f"- {lib}" for lib in library_content_list)
        data_lake_content_formatted = "\n".join(f"- {item}" for item in data_lake_formatted)

        # Base prompt
        prompt_modifier = """
You are a helpful biomedical assistant assigned with the task of problem-solving.
To achieve this, you will be using an interactive coding environment equipped with a variety of tool functions, data, and softwares to assist you throughout the process.

Given a task, make a plan first. The plan should be a numbered list of steps that you will take to solve the task. Be specific and detailed.
Format your plan as a checklist with empty checkboxes like this:
1. [ ] First step
2. [ ] Second step
3. [ ] Third step

Follow the plan step by step. After completing each step, update the checklist by replacing the empty checkbox with a checkmark:
1. [✓] First step (completed)
2. [ ] Second step
3. [ ] Third step

At each turn, you should first provide your thinking and reasoning given the conversation history.
After that, you have two options:

1) Interact with a programming environment and receive the corresponding output within <observe></observe>. Your code should be enclosed using "<execute>" tag, for example: <execute> print("Hello World!") </execute>. IMPORTANT: You must end the code block with </execute> tag.

2) When you think it is ready, directly provide a solution that adheres to the required format for the given task to the user. Your solution should be enclosed using "<solution>" tag, for example: The answer is <solution> A </solution>. IMPORTANT: You must end the solution block with </solution> tag.

Environment Resources:

- Function Dictionary:
{function_intro}
{tool_desc}

- Biological data lake
You can access a biological data lake at the following path: {data_lake_path}.
{data_lake_intro}
{data_lake_content}

- Software Library:
{library_intro}
{library_content_formatted}
"""

        # Add self-critic instructions if needed
        if self_critic:
            prompt_modifier += """
You may or may not receive feedbacks from human. If so, address the feedbacks by following the same procedure of multiple rounds of thinking, execution, and then coming up with a new solution.
"""

        # Set appropriate text based on whether this is initial configuration or after retrieval
        if is_retrieval:
            function_intro = "Based on your query, I've identified the following most relevant functions that you can use in your code:"
            data_lake_intro = "Based on your query, I've identified the following most relevant datasets:"
            library_intro = "Based on your query, I've identified the following most relevant libraries that you can use:"
        else:
            function_intro = "In your code, you will need to import the function location using the following dictionary of functions:"
            data_lake_intro = "You can write code to understand the data, process and utilize it for the task. Here is the list of datasets:"
            library_intro = "The environment supports a list of libraries that can be directly used. Do not forget the import statement:"

        # Format the prompt with the appropriate values
        format_dict = {
            "function_intro": function_intro,
            "tool_desc": textify_api_dict(tool_desc) if isinstance(tool_desc, dict) else tool_desc,
            "data_lake_path": self.path + "/data_lake",
            "data_lake_intro": data_lake_intro,
            "data_lake_content": data_lake_content_formatted,
            "library_intro": library_intro,
            "library_content_formatted": library_content_formatted,
        }

        formatted_prompt = prompt_modifier.format(**format_dict)
        return formatted_prompt

    def configure(self, self_critic=False, test_time_scale_round=0):
        """Configure the agent with the initial system prompt and workflow, based on the selected architecture."""
        self.self_critic = self_critic
        
        # Common setup for all architectures
        data_lake_path = self.path + "/data_lake"
        data_lake_content = glob.glob(data_lake_path + "/*")
        data_lake_items = [x.split("/")[-1] for x in data_lake_content]
        
        if not hasattr(self, "data_lake_dict"):
            self.data_lake_dict = data_lake_dict
        if not hasattr(self, "library_content_dict"):
            self.library_content_dict = library_content_dict
            
        tool_desc = {i: [x for x in j if x["name"] != "run_python_repl"] for i, j in self.module2api.items()}
        data_lake_with_desc = [{"name": item, "description": self.data_lake_dict.get(item, "")} for item in data_lake_items]

        # Add custom data items if they exist
        if hasattr(self, "_custom_data") and self._custom_data:
            for name, info in self._custom_data.items():
                data_lake_with_desc.append({"name": name, "description": info["description"]})

        # Prepare library content list including custom software
        library_content_list = list(self.library_content_dict.keys())
        if hasattr(self, "_custom_software") and self._custom_software:
            for name in self._custom_software:
                if name not in library_content_list:
                    library_content_list.append(name)

        # Initialize the workflow graph
        workflow = StateGraph(AgentState)

        # Architecture-Specific Graph Construction
        if self.architecture == 'baseline':
            print("Configuring: Baseline Architecture")
            self.system_prompt = self._generate_system_prompt(
                tool_desc=tool_desc,
                data_lake_content=data_lake_with_desc,
                library_content_list=library_content_list,
                self_critic=self_critic
            )

            def generate(state: AgentState) -> AgentState:
                messages = [SystemMessage(content=self.system_prompt)] + state["messages"]
                self.print_token_info(messages, "BEFORE LLM INVOKE")
                
                response = self.llm.invoke(messages)
                msg = str(response.content)
                
                # Check for incomplete tags and fix them
                if "<execute>" in msg and "</execute>" not in msg:
                    msg += "</execute>"
                if "<solution>" in msg and "</solution>" not in msg:
                    msg += "</solution>"
                
                state["messages"].append(AIMessage(content=msg.strip()))
                
                if "<solution>" in msg:
                    state["next_step"] = "end"
                elif "<execute>" in msg:
                    state["next_step"] = "execute"
                else:
                    state["next_step"] = "generate"
                return state

            def execute(state: AgentState) -> AgentState:
                last_message = state["messages"][-1].content
                execute_match = re.search(r"<execute>(.*?)</execute>", last_message, re.DOTALL)
                if execute_match:
                    code = execute_match.group(1)
                    result = self._execute_code(code)
                    observation = f"\n<observation>{result}</observation>"
                    state["messages"].append(AIMessage(content=observation.strip()))
                return state

            def routing_function(state: AgentState) -> Literal["execute", "generate", "end"]:
                next_step = state.get("next_step")
                if next_step == "execute":
                    return "execute"
                elif next_step == "generate":
                    return "generate"
                elif next_step == "end":
                    return "end"
                else:
                    return "generate"
            
            workflow.add_node("generate", generate)
            workflow.add_node("execute", execute)
            workflow.add_conditional_edges("generate", routing_function, {"execute": "execute", "generate": "generate", "end": END})
            workflow.add_edge("execute", "generate")
            workflow.add_edge(START, "generate")

        elif self.architecture == 'hierarchical_expert':
            print("Configuring: Hierarchical Expert Architecture")
            BENCH_COOLDOWN_SECONDS = 300  # 5 minutes

            base_system_prompt = self._generate_system_prompt(
                tool_desc=tool_desc,
                data_lake_content=data_lake_with_desc,
                library_content_list=library_content_list,
                self_critic=self_critic
            )
            
            self.system_prompt_template = base_system_prompt + """

IMPORTANT: Some tools may be temporarily unavailable or "benched" if they have recently failed.
Currently benched tools: {benched_tools_str}
If your first choice of tool is benched, please select an alternative tool to accomplish the task.
"""

            def generate(state: AgentState) -> AgentState:
                benched_tools = state.get("benched_tools", {})
                active_benched_tools = {tool: ts for tool, ts in benched_tools.items() if time.time() < ts}
                benched_str = ", ".join(active_benched_tools.keys()) or "None"
                
                current_system_prompt = self.system_prompt_template.format(benched_tools_str=benched_str)
                messages = [SystemMessage(content=current_system_prompt)] + state["messages"]
                
                response = self.llm.invoke(messages)
                msg = str(response.content)
                if "<execute>" in msg and "</execute>" not in msg:
                    msg += "</execute>"
                if "<solution>" in msg and "</solution>" not in msg:
                    msg += "</solution>"
                    
                state["messages"].append(AIMessage(content=msg.strip()))
                
                if "<solution>" in msg:
                    state["next_step"] = "end"
                elif "<execute>" in msg:
                    state["next_step"] = "execute"
                else:
                    state["next_step"] = "generate"
                return state

            def execute(state: AgentState) -> AgentState:
                last_message = state["messages"][-1].content
                execute_match = re.search(r"<execute>(.*?)</execute>", last_message, re.DOTALL)
                if execute_match:
                    code = execute_match.group(1)
                    tool_name = self._parse_tool_call_from_code(code)
                    try:
                        result = self._execute_code(code)
                        observation = f"\n<observation>{result}</observation>"
                        state["messages"].append(AIMessage(content=observation.strip()))
                        state["next_step"] = "generate"
                    except Exception as e:
                        print(f"Execution failed for tool '{tool_name}': {e}")
                        benched_tools = state.get("benched_tools", {})
                        if tool_name:
                            benched_tools[tool_name] = time.time() + BENCH_COOLDOWN_SECONDS
                        state["benched_tools"] = benched_tools
                        state["messages"].append(HumanMessage(content=f"TOOL ERROR: The tool '{tool_name}' failed with error: {e}. Please try a different tool or approach."))
                        state["next_step"] = "generate"
                return state

            def router(state: AgentState) -> Literal["execute", "generate", "end"]:
                next_step = state.get("next_step", "generate")
                if next_step == "end":
                    return "end"
                elif next_step == "execute":
                    return "execute"
                else:
                    return "generate"
            
            workflow.add_node("generate", generate)
            workflow.add_node("execute", execute)
            workflow.add_conditional_edges("generate", router, {"execute": "execute", "generate": "generate", "end": END})
            workflow.add_edge("execute", "generate")
            workflow.add_edge(START, "generate")

        elif self.architecture == 'cognitive_corrector':
            print("Configuring: Cognitive Corrector Architecture")
            
            # Knowledge base of known tool issues and their corrections
            self.tool_knowledge_base = {
                'query_ensembl': {
                    'error_pattern': "get_llm() got an unexpected keyword argument 'llm'",
                    'correction_template': 'query_ensembl(endpoint="lookup/symbol/homo_sapiens/{gene_symbol}")'
                },
                'query_stringdb': {
                    'error_pattern': "get_llm() got an unexpected keyword argument 'llm'",
                    'correction_template': 'query_stringdb(endpoint="https://string-db.org/api/json/interaction_partners?identifiers={protein}&species=9606")'
                }
            }
            
            base_system_prompt = self._generate_system_prompt(
                tool_desc=tool_desc,
                data_lake_content=data_lake_with_desc,
                library_content_list=library_content_list,
                self_critic=self_critic
            )
            
            self.system_prompt = base_system_prompt + """

Instead of writing code directly in an <execute> tag, please first state your intention in an <intention> tag.
The intention should be a JSON object with 'tool' and 'parameters' keys.
Example: <intention>{"tool": "query_ensembl", "parameters": {"gene_symbol": "CFTR"}}</intention>
"""

            def generate(state: AgentState) -> AgentState:
                messages = [SystemMessage(content=self.system_prompt)] + state["messages"]
                response = self.llm.invoke(messages)
                msg = str(response.content)
                if "<intention>" in msg and "</intention>" not in msg:
                    msg += "</intention>"
                if "<solution>" in msg and "</solution>" not in msg:
                    msg += "</solution>"
                    
                state["messages"].append(AIMessage(content=msg.strip()))
                
                if "<solution>" in msg:
                    state["next_step"] = "end"
                elif "<intention>" in msg:
                    state["next_step"] = "cognitive_corrector"
                else:
                    state["next_step"] = "generate"
                return state

            def cognitive_corrector(state: AgentState) -> AgentState:
                last_message = state["messages"][-1].content
                intention_match = re.search(r"<intention>(.*?)</intention>", last_message, re.DOTALL)
                if intention_match:
                    try:
                        intention = json.loads(intention_match.group(1))
                        state["tool_intention"] = intention
                        tool_name = intention.get("tool")
                        params = intention.get("parameters", {})

                        # Check knowledge base for known issues
                        if tool_name in self.tool_knowledge_base:
                            print(f"Corrector: Found known issue for tool '{tool_name}'. Applying correction.")
                            correction_rule = self.tool_knowledge_base[tool_name]
                            corrected_code = correction_rule['correction_template'].format(**params)
                        else:
                            # If no known issue, generate standard code
                            params_str = ', '.join(f'{k}="{v}"' for k, v in params.items())
                            corrected_code = f"{tool_name}({params_str})"
                        
                        state["corrected_code"] = corrected_code
                        state["next_step"] = "execute"
                    except (json.JSONDecodeError, KeyError) as e:
                        state["messages"].append(HumanMessage(content=f"CORRECTOR ERROR: Invalid intention format. {e}"))
                        state["next_step"] = "generate"
                else:
                    state["messages"].append(HumanMessage(content="CORRECTOR ERROR: No <intention> tag found."))
                    state["next_step"] = "generate"
                return state

            def execute(state: AgentState) -> AgentState:
                code = state.get("corrected_code")
                if code:
                    try:
                        result = self._execute_code(code)
                        observation = f"\n<observation>{result}</observation>"
                        state["messages"].append(AIMessage(content=observation.strip()))
                    except Exception as e:
                        state["messages"].append(HumanMessage(content=f"EXECUTION ERROR: {e}"))
                state["next_step"] = "generate"
                return state

            def router(state: AgentState) -> Literal["generate", "cognitive_corrector", "execute", "end"]:
                next_step = state.get("next_step", "generate")
                if next_step == "end":
                    return "end"
                elif next_step == "cognitive_corrector":
                    return "cognitive_corrector"
                elif next_step == "execute":
                    return "execute"
                else:
                    return "generate"

            workflow.add_node("generate", generate)
            workflow.add_node("cognitive_corrector", cognitive_corrector)
            workflow.add_node("execute", execute)
            workflow.add_edge(START, "generate")
            workflow.add_conditional_edges("generate", router, {
                "cognitive_corrector": "cognitive_corrector",
                "generate": "generate",
                "end": END
            })
            workflow.add_conditional_edges("cognitive_corrector", router, {
                "execute": "execute",
                "generate": "generate"
            })
            workflow.add_edge("execute", "generate")

        elif self.architecture == 'exploratory_sandbox':
            print("Configuring: Exploratory Sandbox Architecture")
            
            print("Running environment exploration pre-flight check...")
            sanity_checks = [
                '#!BASH\ncommand -v mafft',
                '#!BASH\ncommand -v clustalo',
                'import biomni.tool.database'
            ]
            check_results = []
            for check in sanity_checks:
                try:
                    result = self._execute_code(check)
                    if "not found" in result or (isinstance(result, str) and "Error" in result):
                        check_results.append(f"- {check.splitlines()[-1]}: UNAVAILABLE")
                    else:
                        check_results.append(f"- {check.splitlines()[-1]}: AVAILABLE")
                except Exception as e:
                    check_results.append(f"- {check.splitlines()[-1]}: UNAVAILABLE ({e})")
            
            environment_profile = "\n".join(check_results)
            print(f"Environment Profile:\n{environment_profile}")

            base_system_prompt = self._generate_system_prompt(
                tool_desc=tool_desc,
                data_lake_content=data_lake_with_desc,
                library_content_list=library_content_list,
                self_critic=self_critic
            )
            
            self.system_prompt = base_system_prompt + f"""

IMPORTANT ENVIRONMENT PROFILE:
Your execution environment has been scanned. Please use ONLY the available tools and libraries listed below.
{environment_profile}
"""
            
            def generate(state: AgentState) -> AgentState:
                messages = [SystemMessage(content=self.system_prompt)] + state["messages"]
                response = self.llm.invoke(messages)
                msg = str(response.content)
                
                if "<execute>" in msg and "</execute>" not in msg:
                    msg += "</execute>"
                if "<solution>" in msg and "</solution>" not in msg:
                    msg += "</solution>"
                    
                state["messages"].append(AIMessage(content=msg.strip()))
                if "<solution>" in msg:
                    state["next_step"] = "end"
                elif "<execute>" in msg:
                    state["next_step"] = "execute"
                else:
                    state["next_step"] = "generate"
                return state

            def execute(state: AgentState) -> AgentState:
                last_message = state["messages"][-1].content
                execute_match = re.search(r"<execute>(.*?)</execute>", last_message, re.DOTALL)
                if execute_match:
                    code = execute_match.group(1)
                    result = self._execute_code(code)
                    observation = f"\n<observation>{result}</observation>"
                    state["messages"].append(AIMessage(content=observation.strip()))
                return state
            
            def router(state: AgentState) -> Literal["execute", "generate", "end"]:
                next_step = state.get("next_step", "generate")
                if next_step == "end":
                    return "end"
                elif next_step == "execute":
                    return "execute"
                else:
                    return "generate"

            workflow.add_node("generate", generate)
            workflow.add_node("execute", execute)
            workflow.add_conditional_edges("generate", router, {"execute": "execute", "generate": "generate", "end": END})
            workflow.add_edge("execute", "generate")
            workflow.add_edge(START, "generate")

        elif self.architecture == 'tool_augmented_graph':
            print("Configuring: Tool-Augmented Graph (TAG) Architecture")
            
            base_system_prompt = self._generate_system_prompt(
                tool_desc=tool_desc,
                data_lake_content=data_lake_with_desc,
                library_content_list=library_content_list,
                self_critic=self_critic
            )
            
            self.system_prompt = base_system_prompt + """

In addition to <execute> and <solution>, you can use two new actions:
1. <interrogate_tool tool_name="..."/>: To get detailed documentation and examples for a specific tool.
2. <sandbox_code>...</sandbox_code>: To test a small, non-critical code snippet safely.
Use these to understand tools better before executing a final plan.
"""

            def generate(state: AgentState) -> AgentState:
                messages = [SystemMessage(content=self.system_prompt)] + state["messages"]
                response = self.llm.invoke(messages)
                msg = str(response.content)
                
                if "<execute>" in msg and "</execute>" not in msg:
                    msg += "</execute>"
                if "<solution>" in msg and "</solution>" not in msg:
                    msg += "</solution>"
                if "<sandbox_code>" in msg and "</sandbox_code>" not in msg:
                    msg += "</sandbox_code>"
                    
                state["messages"].append(AIMessage(content=msg.strip()))
                
                if "<solution>" in msg:
                    state["next_step"] = "end"
                elif "<execute>" in msg:
                    state["next_step"] = "execute"
                elif "<interrogate_tool" in msg:
                    state["next_step"] = "interrogate"
                elif "<sandbox_code>" in msg:
                    state["next_step"] = "sandbox"
                else:
                    state["next_step"] = "generate"
                return state

            def interrogate_tool(state: AgentState) -> AgentState:
                last_message = state["messages"][-1].content
                match = re.search(r'<interrogate_tool tool_name="(\w+)"\s*/>', last_message)
                if match:
                    tool_name = match.group(1)
                    tool_info = "No detailed documentation found."
                    for mod, tools in self.module2api.items():
                        for t in tools:
                            if t['name'] == tool_name:
                                tool_info = f"Documentation for {tool_name}:\nDescription: {t.get('description', 'N/A')}\nParameters: {t.get('parameters', 'N/A')}\nExample: {t.get('example', 'No example available.')}"
                                break
                    observation = f"\n<observation>{tool_info}</observation>"
                    state["messages"].append(AIMessage(content=observation))
                state["next_step"] = "generate"
                return state
            
            def sandbox_code(state: AgentState) -> AgentState:
                last_message = state["messages"][-1].content
                match = re.search(r"<sandbox_code>(.*?)</sandbox_code>", last_message, re.DOTALL)
                if match:
                    code = match.group(1)
                    result = self._execute_code(code)
                    observation = f"\n<observation>[SANDBOX] Result: {result}</observation>"
                    state["messages"].append(AIMessage(content=observation))
                state["next_step"] = "generate"
                return state

            def execute(state: AgentState) -> AgentState:
                last_message = state["messages"][-1].content
                match = re.search(r"<execute>(.*?)</execute>", last_message, re.DOTALL)
                if match:
                    code = match.group(1)
                    result = self._execute_code(code)
                    observation = f"\n<observation>{result}</observation>"
                    state["messages"].append(AIMessage(content=observation.strip()))
                state["next_step"] = "generate"
                return state

            def router(state: AgentState) -> Literal["interrogate", "sandbox", "execute", "generate", "end"]:
                next_step = state.get("next_step", "generate")
                if next_step == "end":
                    return "end"
                elif next_step == "interrogate":
                    return "interrogate"
                elif next_step == "sandbox":
                    return "sandbox"
                elif next_step == "execute":
                    return "execute"
                else:
                    return "generate"

            workflow.add_node("generate", generate)
            workflow.add_node("interrogate", interrogate_tool)
            workflow.add_node("sandbox", sandbox_code)
            workflow.add_node("execute", execute)
            workflow.add_edge(START, "generate")
            workflow.add_conditional_edges("generate", router, {
                "interrogate": "interrogate",
                "sandbox": "sandbox",
                "execute": "execute",
                "generate": "generate",
                "end": END
            })
            workflow.add_edge("interrogate", "generate")
            workflow.add_edge("sandbox", "generate")
            workflow.add_edge("execute", "generate")

        # Final Compilation Step for all architectures
        self.app = workflow.compile()
        self.checkpointer = MemorySaver()
        self.app.checkpointer = self.checkpointer
        graph_png = self.app.get_graph().draw_mermaid_png()
        with open('workflow_graph_f1_edited.png', 'wb') as f:
            f.write(graph_png)
        display(Image(graph_png))

        # Track system prompt size
        system_prompt_tokens = self.count_tokens(getattr(self, 'system_prompt', ''))
        print(f"\n=== SYSTEM PROMPT GENERATED ({self.architecture.upper()}) ===")
        print(f"System prompt length: {len(getattr(self, 'system_prompt', '')):,} characters")
        print(f"Estimated tokens: {system_prompt_tokens:,}")
        print(f"=============================================\n")

    def go(self, prompt):
        """Execute the agent with the given prompt."""
        self.critic_count = 0
        self.user_task = prompt

        if self.use_tool_retriever:
            self.update_system_prompt_with_selected_resources(prompt)

        inputs = {"messages": [HumanMessage(content=prompt)], "next_step": None}
        config = {"recursion_limit": 500, "configurable": {"thread_id": str(uuid.uuid4())}}
        self.log = []

        for s in self.app.stream(inputs, stream_mode="values", config=config):
            message = s["messages"][-1]
            out = pretty_print(message)
            self.log.append(out)

        return self.log, message.content

    def update_system_prompt_with_selected_resources(self, prompt):
        """Update the system prompt with the selected resources."""
        # Gather all available resources
        all_tools = self.tool_registry.tools if hasattr(self, "tool_registry") else []

        # Data lake items with descriptions
        data_lake_path = self.path + "/data_lake"
        data_lake_content = glob.glob(data_lake_path + "/*")
        data_lake_items = [x.split("/")[-1] for x in data_lake_content]

        # Create data lake descriptions for retrieval
        data_lake_descriptions = []
        for item in data_lake_items:
            description = self.data_lake_dict.get(item, f"Data lake item: {item}")
            data_lake_descriptions.append({"name": item, "description": description})

        # Add custom data items to retrieval if they exist
        if hasattr(self, "_custom_data") and self._custom_data:
            for name, info in self._custom_data.items():
                data_lake_descriptions.append({"name": name, "description": info["description"]})

        # Libraries with descriptions
        library_descriptions = []
        for lib_name, lib_desc in self.library_content_dict.items():
            library_descriptions.append({"name": lib_name, "description": lib_desc})

        # Add custom software items to retrieval if they exist
        if hasattr(self, "_custom_software") and self._custom_software:
            for name, info in self._custom_software.items():
                if not any(lib["name"] == name for lib in library_descriptions):
                    library_descriptions.append({"name": name, "description": info["description"]})

        # Use retrieval to get relevant resources
        resources = {
            "tools": all_tools,
            "data_lake": data_lake_descriptions,
            "libraries": library_descriptions,
        }

        # Use prompt-based retrieval with the agent's LLM
        selected_resources = self.retriever.prompt_based_retrieval(prompt, resources, llm=self.llm)
        print("Using prompt-based retrieval with the agent's LLM")

        # Extract the names from the selected resources for the system prompt
        selected_resources_names = {
            "tools": selected_resources["tools"],
            "data_lake": [],
            "libraries": [lib["name"] if isinstance(lib, dict) else lib for lib in selected_resources["libraries"]],
        }

        # Process data lake items to extract just the names
        for item in selected_resources["data_lake"]:
            if isinstance(item, dict):
                selected_resources_names["data_lake"].append(item["name"])
            elif isinstance(item, str) and ": " in item:
                name = item.split(": ")[0]
                selected_resources_names["data_lake"].append(name)
            else:
                selected_resources_names["data_lake"].append(item)

        # Extract tool descriptions for the selected tools
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
                module_name = getattr(tool, "module_name", "biomni.tool.scRNA_tools")

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

        # Prepare data lake items with descriptions
        data_lake_with_desc = []
        for item in selected_resources_names["data_lake"]:
            description = self.data_lake_dict.get(item, f"Data lake item: {item}")
            data_lake_with_desc.append({"name": item, "description": description})

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
            library_content_list=selected_resources_names["libraries"],
            self_critic=getattr(self, "self_critic", False),
            is_retrieval=True,
            custom_tools=custom_tools if custom_tools else None,
            custom_data=custom_data if custom_data else None,
            custom_software=custom_software if custom_software else None,
        )
        
        # Track system prompt size after retrieval
        system_prompt_tokens = self.count_tokens(self.system_prompt)
        print(f"\n=== SYSTEM PROMPT UPDATED (AFTER RETRIEVAL) ===")
        print(f"System prompt length: {len(self.system_prompt):,} characters")
        print(f"Estimated tokens: {system_prompt_tokens:,}")
        print(f"===============================================\n")

    def result_formatting(self, output_class, task_intention):
        """Format the result using a structured output class."""
        self.format_check_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                (
                    "You are evaluateGPT, tasked with extract and parse the task output based on the history of an agent. "
                    "Review the entire history of messages provided. "
                    "Here is the task output requirement: \n"
                    f"'{task_intention.replace('{', '{{').replace('}', '}}')}'.\n"
                ),
            ),
            ("placeholder", "{messages}"),
        ])

        checker_llm = self.format_check_prompt | self.llm.with_structured_output(output_class)
        result = checker_llm.invoke({"messages": [("user", str(self.log))]}).dict()
        return result