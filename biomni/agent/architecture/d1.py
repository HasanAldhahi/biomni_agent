import glob
import inspect
import os
import re
from typing import Literal, TypedDict

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
    exploration_complete: bool | None
    environment_profile: str | None
    task_success: bool | None


class D1:
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

    # Include all the helper methods from original class (shortened for space)
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

            if hasattr(self, "tool_registry") and self.tool_registry is not None:
                try:
                    docs = []
                    for tool_id in range(len(self.tool_registry.tools)):
                        docs.append([int(tool_id), self.tool_registry.get_tool_by_id(int(tool_id))])
                    self.tool_registry.document_df = pd.DataFrame(docs, columns=["docid", "document_content"])
                except Exception as e:
                    print(f"Warning: Failed to update tool registry document dataframe: {e}")

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
        import builtins
        if hasattr(builtins, "_biomni_custom_functions") and name in builtins._biomni_custom_functions:
            del builtins._biomni_custom_functions[name]
        if hasattr(self, "tool_registry") and self.tool_registry is not None:
            if self.tool_registry.remove_tool_by_name(name):
                removed = True
                try:
                    docs = []
                    for tool_id in range(len(self.tool_registry.tools)):
                        docs.append([int(tool_id), self.tool_registry.get_tool_by_id(int(tool_id))])
                    self.tool_registry.document_df = pd.DataFrame(docs, columns=["docid", "document_content"])
                except Exception as e:
                    print(f"Warning: Failed to update tool registry document dataframe: {e}")
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
        try:
            if not isinstance(data, dict):
                raise ValueError("Data must be a dictionary with file path as key and description as value")
            if not hasattr(self, "_custom_data"):
                self._custom_data = {}
            for file_path, description in data.items():
                if not isinstance(file_path, str) or not isinstance(description, str):
                    print("Warning: Skipping invalid data entry - file_path and description must be strings")
                    continue
                filename = os.path.basename(file_path) if "/" in file_path else file_path
                self._custom_data[filename] = {"path": file_path, "description": description}
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
        if removed:
            print(f"Custom data item '{name}' has been removed")
        else:
            print(f"Custom data item '{name}' was not found")
        return removed

    def add_software(self, software):
        try:
            if not isinstance(software, dict):
                raise ValueError("Software must be a dictionary with software name as key and description as value")
            if not hasattr(self, "_custom_software"):
                self._custom_software = {}
            for software_name, description in software.items():
                if not isinstance(software_name, str) or not isinstance(description, str):
                    print("Warning: Skipping invalid software entry - software_name and description must be strings")
                    continue
                self._custom_software[software_name] = {"name": software_name, "description": description}
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

        # Separate custom and default resources
        default_data_lake_content = []
        default_library_content_list = []

        custom_data_names = set()
        custom_software_names = set()

        if custom_data:
            custom_data_names = {item.get("name") if isinstance(item, dict) else item for item in custom_data}
        if custom_software:
            custom_software_names = {item.get("name") if isinstance(item, dict) else item for item in custom_software}

        for item in data_lake_content:
            if isinstance(item, dict):
                name = item.get("name", "")
                if name not in custom_data_names:
                    default_data_lake_content.append(item)
            elif item not in custom_data_names:
                default_data_lake_content.append(item)

        for lib in library_content_list:
            if isinstance(lib, dict):
                name = lib.get("name", "")
                if name not in custom_software_names:
                    default_library_content_list.append(lib)
            elif lib not in custom_software_names:
                default_library_content_list.append(lib)

        # Format the default data lake content
        if isinstance(default_data_lake_content, list) and all(isinstance(item, str) for item in default_data_lake_content):
            data_lake_formatted = []
            for item in default_data_lake_content:
                if ": " in item:
                    data_lake_formatted.append(item)
                else:
                    description = self.data_lake_dict.get(item, f"Data lake item: {item}")
                    data_lake_formatted.append(format_item_with_description(item, description))
        else:
            data_lake_formatted = []
            for item in default_data_lake_content:
                if isinstance(item, dict):
                    name = item.get("name", "")
                    description = self.data_lake_dict.get(name, f"Data lake item: {name}")
                    data_lake_formatted.append(format_item_with_description(name, description))
                elif isinstance(item, str) and ": " in item:
                    data_lake_formatted.append(item)
                else:
                    description = self.data_lake_dict.get(item, f"Data lake item: {item}")
                    data_lake_formatted.append(format_item_with_description(item, description))

        # Format the default library content
        if isinstance(default_library_content_list, list) and all(isinstance(item, str) for item in default_library_content_list):
            if (len(default_library_content_list) > 0 and isinstance(default_library_content_list[0], str) and "," not in default_library_content_list[0]):
                libraries_formatted = []
                for lib in default_library_content_list:
                    description = self.library_content_dict.get(lib, f"Software library: {lib}")
                    libraries_formatted.append(format_item_with_description(lib, description))
            else:
                libraries_formatted = default_library_content_list
        else:
            libraries_formatted = []
            for lib in default_library_content_list:
                if isinstance(lib, dict):
                    name = lib.get("name", "")
                    description = self.library_content_dict.get(name, f"Software library: {name}")
                    libraries_formatted.append(format_item_with_description(name, description))
                else:
                    description = self.library_content_dict.get(lib, f"Software library: {lib}")
                    libraries_formatted.append(format_item_with_description(lib, description))

        # Format custom resources with highlighting
        custom_tools_formatted = []
        if custom_tools:
            for tool in custom_tools:
                if isinstance(tool, dict):
                    name = tool.get("name", "Unknown")
                    desc = tool.get("description", "")
                    module = tool.get("module", "custom_tools")
                    custom_tools_formatted.append(f"🔧 {name} (from {module}): {desc}")
                else:
                    custom_tools_formatted.append(f"🔧 {str(tool)}")

        custom_data_formatted = []
        if custom_data:
            for item in custom_data:
                if isinstance(item, dict):
                    name = item.get("name", "Unknown")
                    desc = item.get("description", "")
                    custom_data_formatted.append(f"📊 {format_item_with_description(name, desc)}")
                else:
                    desc = self.data_lake_dict.get(item, f"Custom data: {item}")
                    custom_data_formatted.append(f"📊 {format_item_with_description(item, desc)}")

        custom_software_formatted = []
        if custom_software:
            for item in custom_software:
                if isinstance(item, dict):
                    name = item.get("name", "Unknown")
                    desc = item.get("description", "")
                    custom_software_formatted.append(f"⚙️ {format_item_with_description(name, desc)}")
                else:
                    desc = self.library_content_dict.get(item, f"Custom software: {item}")
                    custom_software_formatted.append(f"⚙️ {format_item_with_description(item, desc)}")

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

If a step fails or needs modification, mark it with an X and explain why:
1. [✓] First step (completed)
2. [✗] Second step (failed because...)
3. [ ] Modified second step
4. [ ] Third step

Always show the updated plan after each step so the user can track progress.

At each turn, you should first provide your thinking and reasoning given the conversation history.
After that, you have two options:

1) Interact with a programming environment and receive the corresponding output within <observe></observe>. Your code should be enclosed using "<execute>" tag, for example: <execute> print("Hello World!") </execute>. IMPORTANT: You must end the code block with </execute> tag.
   - For Python code (default): <execute> print("Hello World!") </execute>
   - For R code: <execute> #!R\nlibrary(ggplot2)\nprint("Hello from R") </execute>
   - For Bash scripts and commands: <execute> #!BASH\necho "Hello from Bash"\nls -la </execute>
   - For CLI softwares, use Bash scripts.

2) When you think it is ready, directly provide a solution that adheres to the required format for the given task to the user. Your solution should be enclosed using "<solution>" tag, for example: The answer is <solution> A </solution>. IMPORTANT: You must end the solution block with </solution> tag.

You have many chances to interact with the environment to receive the observation. So you can decompose your code into multiple steps.
Don't overcomplicate the code. Keep it simple and easy to understand.
When writing the code, please print out the steps and results in a clear and concise manner, like a research log.
When calling the existing python functions in the function dictionary, YOU MUST SAVE THE OUTPUT and PRINT OUT the result.
For example, result = understand_scRNA(XXX) print(result)
Otherwise the system will not be able to know what has been done.

For R code, use the #!R marker at the beginning of your code block to indicate it's R code.
For Bash scripts and commands, use the #!BASH marker at the beginning of your code block. This allows for both simple commands and multi-line scripts with variables, loops, conditionals, loops, and other Bash features.

In each response, you must include EITHER <execute> or <solution> tag. Not both at the same time. Do not respond with messages without any tags. No empty messages.
"""

        # Add self-critic instructions if needed
        if self_critic:
            prompt_modifier += """
You may or may not receive feedbacks from human. If so, address the feedbacks by following the same procedure of multiple rounds of thinking, execution, and then coming up with a new solution.
"""

        # Add custom resources section first (highlighted)
        has_custom_resources = any([custom_tools_formatted, custom_data_formatted, custom_software_formatted])

        if has_custom_resources:
            prompt_modifier += """

PRIORITY CUSTOM RESOURCES
===============================
IMPORTANT: The following custom resources have been specifically added for your use.
    PRIORITIZE using these resources as they are directly relevant to your task.
    Always consider these FIRST and in the meantime using default resources.

"""

            if custom_tools_formatted:
                prompt_modifier += """
CUSTOM TOOLS (USE THESE FIRST):
{custom_tools}

"""

            if custom_data_formatted:
                prompt_modifier += """
CUSTOM DATA (PRIORITIZE THESE DATASETS):
{custom_data}

"""

            if custom_software_formatted:
                prompt_modifier += """
⚙️ CUSTOM SOFTWARE (USE THESE LIBRARIES):
{custom_software}

"""

            prompt_modifier += """===============================
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

        # Format the content consistently for both initial and retrieval cases
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

        # Add custom resources to format dict if they exist
        if custom_tools_formatted:
            format_dict["custom_tools"] = "\n".join(custom_tools_formatted)
        if custom_data_formatted:
            format_dict["custom_data"] = "\n".join(custom_data_formatted)
        if custom_software_formatted:
            format_dict["custom_software"] = "\n".join(custom_software_formatted)

        formatted_prompt = prompt_modifier.format(**format_dict)

        return formatted_prompt

    def configure(self, self_critic=False, test_time_scale_round=0):
        """Configure the agent with the Exploratory Sandbox Model workflow.

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
                custom_tools.append(
                    {
                        "name": name,
                        "description": info["description"],
                        "module": info["module"],
                    }
                )

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

        # Architecture 3: Exploratory Sandbox Model - Define the nodes
        # EXPLORATION MODE
        def generate_exploration_plan(state: AgentState) -> AgentState:
            # Generate exploration plan based on user prompt
            user_prompt = state["messages"][0].content if state["messages"] else ""
            
            # Simple heuristic to generate exploration checks based on keywords
            exploration_checks = []
            if "alignment" in user_prompt.lower():
                exploration_checks.extend(["command -v mafft", "command -v clustalo", "command -v muscle"])
            if "python" in user_prompt.lower() or "import" in user_prompt.lower():
                exploration_checks.extend(["python -c 'import numpy; print(\"numpy available\")'", 
                                         "python -c 'import pandas; print(\"pandas available\")'",
                                         "python -c 'import scipy; print(\"scipy available\")'"])
            if "r" in user_prompt.lower() or "ggplot" in user_prompt.lower():
                exploration_checks.extend(["Rscript --version", "R -e 'library(ggplot2)'"])
            
            # Default checks
            if not exploration_checks:
                exploration_checks = ["python --version", "which python", "ls -la /usr/bin/python*"]
            
            # Create exploration plan message
            plan_msg = "Starting environment exploration with the following checks:\n" + "\n".join(f"- {check}" for check in exploration_checks)
            state["messages"].append(AIMessage(content=plan_msg))
            
            state["next_step"] = "execute_sanity_checks"
            return state

        def execute_sanity_checks(state: AgentState) -> AgentState:
            # Execute the sanity checks
            exploration_checks = [
                "python --version",
                "which python", 
                "python -c 'import sys; print(sys.version)'",
                "python -c 'import numpy; print(\"numpy:\", numpy.__version__)' 2>/dev/null || echo 'numpy: not available'",
                "python -c 'import pandas; print(\"pandas:\", pandas.__version__)' 2>/dev/null || echo 'pandas: not available'",
                "command -v mafft 2>/dev/null || echo 'mafft: not available'",
                "command -v clustalo 2>/dev/null || echo 'clustalo: not available'",
                "Rscript --version 2>/dev/null || echo 'R: not available'"
            ]
            
            results = []
            timeout = 30  # Short timeout for exploration
            
            for check in exploration_checks:
                try:
                    result = run_with_timeout(run_bash_script, [check], timeout=timeout)
                    results.append(f"{check}: {result.strip()}")
                except Exception as e:
                    results.append(f"{check}: ERROR - {str(e)}")
            
            exploration_results = "\n".join(results)
            observation = f"\n<observation>Exploration Results:\n{exploration_results}</observation>"
            state["messages"].append(AIMessage(content=observation))
            
            state["next_step"] = "build_environment_profile"
            return state

        def build_environment_profile(state: AgentState) -> AgentState:
            # Build environment profile from exploration results
            last_message = state["messages"][-1].content if state["messages"] else ""
            
            # Parse results and build profile
            profile_lines = []
            if "not available" in last_message:
                if "mafft: not available" in last_message:
                    profile_lines.append("- Command-line tool 'mafft' is not available")
                if "clustalo: not available" in last_message:
                    profile_lines.append("- Command-line tool 'clustalo' is not available")
                if "numpy: not available" in last_message:
                    profile_lines.append("- Python package 'numpy' is not installed")
                if "pandas: not available" in last_message:
                    profile_lines.append("- Python package 'pandas' is not installed")
                if "R: not available" in last_message:
                    profile_lines.append("- R environment is not available")
            
            # Add available tools
            if "python" in last_message and "not available" not in last_message:
                profile_lines.append("- Python environment is available")
            if "numpy:" in last_message and "not available" not in last_message:
                profile_lines.append("- Python package 'numpy' is available")
            if "pandas:" in last_message and "not available" not in last_message:
                profile_lines.append("- Python package 'pandas' is available")
            
            environment_profile = "\n".join(profile_lines) if profile_lines else "- Standard Python environment detected"
            state["environment_profile"] = environment_profile
            
            profile_msg = f"Environment Profile Built:\n{environment_profile}\n\nNow proceeding to main task execution..."
            state["messages"].append(AIMessage(content=profile_msg))
            
            state["exploration_complete"] = True
            state["next_step"] = "generate_task_plan"
            return state

        # EXECUTION MODE  
        def generate_task_plan(state: AgentState) -> AgentState:
            # Update system prompt with environment constraints
            if state.get("environment_profile"):
                enhanced_prompt = self.system_prompt + f"\n\nIMPORTANT ENVIRONMENT CONSTRAINTS:\n{state['environment_profile']}\nPlease formulate your plan using ONLY the available tools."
                messages = [SystemMessage(content=enhanced_prompt)] + state["messages"]
            else:
                messages = [SystemMessage(content=self.system_prompt)] + state["messages"]
            
            # Track and print token usage before sending to LLM
            self.print_token_info(messages, "BEFORE LLM INVOKE")
            
            response = self.llm.invoke(messages)

            # Parse the response
            msg = str(response.content)

            # Check for incomplete tags and fix them
            if "<execute>" in msg and "</execute>" not in msg:
                msg += "</execute>"
            if "<solution>" in msg and "</solution>" not in msg:
                msg += "</solution>"

            execute_match = re.search(r"<execute>(.*?)</execute>", msg, re.DOTALL)
            answer_match = re.search(r"<solution>(.*?)</solution>", msg, re.DOTALL)

            # Add the message to the state before checking for errors
            state["messages"].append(AIMessage(content=msg.strip()))

            if answer_match:
                state["next_step"] = "end"
            elif execute_match:
                state["next_step"] = "execute_task_code"
            else:
                state["next_step"] = "generate_task_plan"

            return state

        def execute_task_code(state: AgentState) -> AgentState:
            last_message = state["messages"][-1].content
            
            execute_match = re.search(r"<execute>(.*?)</execute>", last_message, re.DOTALL)
            if execute_match:
                code = execute_match.group(1)

                timeout = self.timeout_seconds

                # Check if the code is R code
                if (
                    code.strip().startswith("#!R")
                    or code.strip().startswith("# R code")
                    or code.strip().startswith("# R script")
                ):
                    r_code = re.sub(r"^#!R|^# R code|^# R script", "", code, 1).strip()
                    result = run_with_timeout(run_r_code, [r_code], timeout=timeout)
                # Check if the code is a Bash script or CLI command
                elif (
                    code.strip().startswith("#!BASH")
                    or code.strip().startswith("# Bash script")
                    or code.strip().startswith("#!CLI")
                ):
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
                    result = (
                        "The output is too long to be added to context. Here are the first 10K characters...\n"
                        + result[:10000]
                    )
                
                # Check if execution was successful
                if "Error" in result or "Exception" in result or "Traceback" in result:
                    state["task_success"] = False
                    state["next_step"] = "self_critic_and_replan"
                else:
                    state["task_success"] = True
                    state["next_step"] = "end"
                
                observation = f"\n<observation>{result}</observation>"
                state["messages"].append(AIMessage(content=observation.strip()))

            return state

        def self_critic_and_replan(state: AgentState) -> AgentState:
            # Provide feedback and re-plan
            feedback_msg = "The previous execution failed. Let me analyze the error and try a different approach."
            state["messages"].append(HumanMessage(content=feedback_msg))
            state["next_step"] = "generate_task_plan"
            return state

        def routing_function(
            state: AgentState,
        ) -> Literal["generate_exploration_plan", "execute_sanity_checks", "build_environment_profile", "generate_task_plan", "execute_task_code", "self_critic_and_replan", "end"]:
            next_step = state.get("next_step")
            return next_step

        # Create the workflow
        workflow = StateGraph(AgentState)

        # Add nodes for Architecture 3: Exploratory Sandbox Model
        workflow.add_node("generate_exploration_plan", generate_exploration_plan)
        workflow.add_node("execute_sanity_checks", execute_sanity_checks)
        workflow.add_node("build_environment_profile", build_environment_profile)
        workflow.add_node("generate_task_plan", generate_task_plan)
        workflow.add_node("execute_task_code", execute_task_code)
        workflow.add_node("self_critic_and_replan", self_critic_and_replan)

        # Add conditional edges
        workflow.add_conditional_edges(
            "generate_exploration_plan",
            routing_function,
            path_map={"execute_sanity_checks": "execute_sanity_checks"},
        )
        
        workflow.add_conditional_edges(
            "execute_sanity_checks",
            routing_function,
            path_map={"build_environment_profile": "build_environment_profile"},
        )
        
        workflow.add_conditional_edges(
            "build_environment_profile",
            routing_function,
            path_map={"generate_task_plan": "generate_task_plan"},
        )
        
        workflow.add_conditional_edges(
            "generate_task_plan",
            routing_function,
            path_map={
                "execute_task_code": "execute_task_code",
                "generate_task_plan": "generate_task_plan",
                "end": END,
            },
        )
        
        workflow.add_conditional_edges(
            "execute_task_code",
            routing_function,
            path_map={
                "self_critic_and_replan": "self_critic_and_replan",
                "end": END,
            },
        )
        
        workflow.add_conditional_edges(
            "self_critic_and_replan",
            routing_function,
            path_map={"generate_task_plan": "generate_task_plan"},
        )

        workflow.add_edge(START, "generate_exploration_plan")

        # Compile the workflow
        self.app = workflow.compile()
        self.checkpointer = MemorySaver()
        self.app.checkpointer = self.checkpointer

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