import glob
import inspect
import os
import re
from typing import Literal, TypedDict, Annotated, Sequence

import pandas as pd
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
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # This handles multiple messages properly
    next_step: str | None


class A1:
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
        # get the tool description from the tool_description folder using importlib
        module2api = read_module2api()
        # get the llm 
        self.llm = get_llm(llm, stop_sequences=["</execute>", "</solution>"], base_url=base_url, api_key=api_key, custom_model_name=custom_model_name)
        
        self.module2api = module2api
        # reteriver tool 
        self.use_tool_retriever = use_tool_retriever

        if self.use_tool_retriever:
            self.tool_registry = ToolRegistry(module2api)
            self.retriever = ToolRetriever()

        # Add timeout parameter
        self.timeout_seconds = timeout_seconds  # 10 minutes default timeout
        # get the system prompt for the agent to start with  which is task independent default system prompt 
        self.configure()

    def add_tool(self, api):
        """Add a new tool to the agent's tool registry and make it available for retrieval.

        Args:
            api: A callable function to be added as a tool

        """
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

            # Validate and enhance the schema

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
                    # Continue with adding to module2api even if registry fails

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

            # Update the tool registry's document dataframe if it exists
            if hasattr(self, "tool_registry") and self.tool_registry is not None:
                try:
                    # Rebuild the document dataframe
                    docs = []
                    for tool_id in range(len(self.tool_registry.tools)):
                        docs.append(
                            [
                                int(tool_id),
                                self.tool_registry.get_tool_by_id(int(tool_id)),
                            ]
                        )
                    self.tool_registry.document_df = pd.DataFrame(docs, columns=["docid", "document_content"])
                except Exception as e:
                    print(f"Warning: Failed to update tool registry document dataframe: {e}")

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

            print(
                f"Tool '{schema['name']}' successfully added and ready for use in both direct execution and retrieval"
            )
            self.configure()
            return schema

        except Exception as e:
            print(f"Error adding tool: {e}")
            import traceback

            traceback.print_exc()
            raise

    def get_custom_tool(self, name):
        """Get a custom tool by name.

        Args:
            name: The name of the custom tool

        Returns:
            The custom tool function if found, None otherwise

        """
        if hasattr(self, "_custom_functions") and name in self._custom_functions:
            return self._custom_functions[name]
        return None

    def list_custom_tools(self):
        """List all custom tools that have been added.

        Returns:
            A list of custom tool names

        """
        if hasattr(self, "_custom_functions"):
            return list(self._custom_functions.keys())
        return []

    def remove_custom_tool(self, name):
        """Remove a custom tool.

        Args:
            name: The name of the custom tool to remove

        Returns:
            True if the tool was removed, False if it wasn't found

        """
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
                # Rebuild the document dataframe
                try:
                    docs = []
                    for tool_id in range(len(self.tool_registry.tools)):
                        docs.append(
                            [
                                int(tool_id),
                                self.tool_registry.get_tool_by_id(int(tool_id)),
                            ]
                        )
                    self.tool_registry.document_df = pd.DataFrame(docs, columns=["docid", "document_content"])
                except Exception as e:
                    print(f"Warning: Failed to update tool registry document dataframe: {e}")

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
        """Add new data to the data lake.

        Args:
            data: Dictionary with file path as key and description as value
                  e.g., {'my_dataset.csv': 'A dataset containing gene expression data'}
                  or {'path/to/file.txt': 'Description of the file'}

        """
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
        """Get a custom data item by name.

        Args:
            name: The name of the custom data item

        Returns:
            The custom data item info if found, None otherwise

        """
        if hasattr(self, "_custom_data") and name in self._custom_data:
            return self._custom_data[name]
        return None

    def list_custom_data(self):
        """List all custom data items that have been added.

        Returns:
            A list of custom data item names and descriptions

        """
        if hasattr(self, "_custom_data"):
            return [(name, info["description"]) for name, info in self._custom_data.items()]
        return []

    def remove_custom_data(self, name):
        """Remove a custom data item.

        Args:
            name: The name of the custom data item to remove

        Returns:
            True if the data item was removed, False if it wasn't found

        """
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
        """Add new software to the software library.

        Args:
            software: Dictionary with software name as key and description as value
                     e.g., {'custom_tool': 'A custom analysis tool for processing data'}
                     or {'my_package': 'Description of the package functionality'}

        """
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
        """Get a custom software item by name.

        Args:
            name: The name of the custom software item

        Returns:
            The custom software item info if found, None otherwise

        """
        if hasattr(self, "_custom_software") and name in self._custom_software:
            return self._custom_software[name]
        return None

    def list_custom_software(self):
        """List all custom software items that have been added.

        Returns:
            A list of custom software item names and descriptions

        """
        if hasattr(self, "_custom_software"):
            return [(name, info["description"]) for name, info in self._custom_software.items()]
        return []

    def remove_custom_software(self, name):
        """Remove a custom software item.

        Args:
            name: The name of the custom software item to remove

        Returns:
            True if the software item was removed, False if it wasn't found

        """
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
        """Generate the system prompt based on the provided resources.

        Args:
            tool_desc: Dictionary of tool descriptions
            data_lake_content: List of data lake items
            library_content_list: List of libraries
            self_critic: Whether to include self-critic instructions
            is_retrieval: Whether this is for retrieval (True) or initial configuration (False)
            custom_tools: List of custom tools to highlight
            custom_data: List of custom data items to highlight
            custom_software: List of custom software items to highlight

        Returns:
            The generated system prompt

        """

        def format_item_with_description(name, description):
            """Format an item with its description in a readable way."""
            # Handle None or empty descriptions
            if not description:
                description = f"Data lake item: {name}"

            # Check if the item is already formatted (contains a colon)
            if isinstance(name, str) and ": " in name:
                return name

            # Wrap long descriptions to make them more readable
            max_line_length = 80
            if len(description) > max_line_length:
                # Simple wrapping for long descriptions
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

                # Join with newlines and proper indentation
                formatted_desc = f"{name}:\n  " + "\n  ".join(wrapped_desc)
                return formatted_desc
            else:
                return f"{name}: {description}"

        # Separate custom and default resources
        default_data_lake_content = []
        default_library_content_list = []

        # Filter out custom items from default lists
        custom_data_names = set()
        custom_software_names = set()

        if custom_data:
            custom_data_names = {item.get("name") if isinstance(item, dict) else item for item in custom_data}
        if custom_software:
            custom_software_names = {item.get("name") if isinstance(item, dict) else item for item in custom_software}

        # Separate default data lake items
        for item in data_lake_content:
            if isinstance(item, dict):
                name = item.get("name", "")
                if name not in custom_data_names:
                    default_data_lake_content.append(item)
            elif item not in custom_data_names:
                default_data_lake_content.append(item)

        # Separate default library items
        for lib in library_content_list:
            if isinstance(lib, dict):
                name = lib.get("name", "")
                if name not in custom_software_names:
                    default_library_content_list.append(lib)
            elif lib not in custom_software_names:
                default_library_content_list.append(lib)

        # Format the default data lake content
        if isinstance(default_data_lake_content, list) and all(
            isinstance(item, str) for item in default_data_lake_content
        ):
            # Simple list of strings - check if they already have descriptions
            data_lake_formatted = []
            for item in default_data_lake_content:
                # Check if the item already has a description (contains a colon)
                if ": " in item:
                    data_lake_formatted.append(item)
                else:
                    description = self.data_lake_dict.get(item, f"Data lake item: {item}")
                    data_lake_formatted.append(format_item_with_description(item, description))
        else:
            # List with descriptions
            data_lake_formatted = []
            for item in default_data_lake_content:
                if isinstance(item, dict):
                    name = item.get("name", "")
                    description = self.data_lake_dict.get(name, f"Data lake item: {name}")
                    data_lake_formatted.append(format_item_with_description(name, description))
                # Check if the item already has a description (contains a colon)
                elif isinstance(item, str) and ": " in item:
                    data_lake_formatted.append(item)
                else:
                    description = self.data_lake_dict.get(item, f"Data lake item: {item}")
                    data_lake_formatted.append(format_item_with_description(item, description))

        # Format the default library content
        if isinstance(default_library_content_list, list) and all(
            isinstance(item, str) for item in default_library_content_list
        ):
            if (
                len(default_library_content_list) > 0
                and isinstance(default_library_content_list[0], str)
                and "," not in default_library_content_list[0]
            ):
                # Simple list of strings
                libraries_formatted = []
                for lib in default_library_content_list:
                    description = self.library_content_dict.get(lib, f"Software library: {lib}")
                    libraries_formatted.append(format_item_with_description(lib, description))
            else:
                # Already formatted string
                libraries_formatted = default_library_content_list
        else:
            # List with descriptions
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

🚨 CRITICAL RULE: ONLY USE FUNCTIONS THAT ARE EXPLICITLY LISTED BELOW 🚨
- DO NOT invent, guess, or assume function names such as query_disgenet
- DO NOT use functions not in the provided dictionary
- IF a function you need is not listed, explain what you would need and ask for guidance
- ALWAYS check the exact function name and import path from the dictionary below

WRONG ❌:
from biomni.tool.database import query_disgenet  # This function doesn't exist!

RIGHT ✅:
# First check if the function exists in the dictionary below
# Use only the exact names and imports shown

Given a task, make a plan first. The plan should be a numbered list of steps.

At each turn:
1) FIRST: Check the function dictionary below for available tools
2) ONLY use functions that are explicitly listed with their exact import paths
3) If you need a function that's not available, explain the limitation

Your code should be enclosed using "<execute>" tag, for example: 
<execute>
# Only use functions from the dictionary below
from biomni.tool.synthetic_biology import design_knockout_sgrna  # This exists!
result = design_knockout_sgrna(...)
print(result)
</execute>

IMPORTANT: You must end the code block with </execute> tag.
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
            library_intro = (
                "Based on your query, I've identified the following most relevant libraries that you can use:"
            )
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
        """Configure the agent with the initial system prompt and workflow.

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
        print(f"Generated system prompt: {self.system_prompt}")

        # Define the nodes - make generate only responsible for LLM responses
        def generate(state: AgentState) -> AgentState:
            # Add system prompt if not present
            if not state["messages"] or not isinstance(state["messages"][0], SystemMessage):
                messages = [SystemMessage(content=self.system_prompt)] + state["messages"]
            else:
                messages = state["messages"]
            
            # Just get LLM response - NO routing logic here
            response = self.llm.invoke(messages)
            
            # Return only the new message - let validate_response handle routing
            return {"messages": [AIMessage(content=str(response.content))]}

        def execute(state: AgentState) -> AgentState:
            last_message = state["messages"][-1].content
            
            execute_match = re.search(r"<execute>(.*?)</execute>", last_message, re.DOTALL)
            if execute_match:
                code = execute_match.group(1)
                
                # VALIDATE: Check for hallucinated imports before execution
                validation_error = self._validate_imports(code)
                if validation_error:
                    # Append error to existing message to maintain role alternation
                    error_msg = f"\n<observation>IMPORT ERROR: {validation_error}\n\nAvailable functions:\n{self._list_available_imports()}</observation>"
                    updated_content = state["messages"][-1].content + error_msg
                    updated_messages = state["messages"][:-1] + [AIMessage(content=updated_content)]
                    return {"messages": updated_messages}
                
                # Continue with normal execution...
                timeout = self.timeout_seconds
                
                try:
                    if code.strip().startswith("#!R"):
                        r_code = re.sub(r"^#!R", "", code, 1).strip()
                        result = run_with_timeout(run_r_code, [r_code], timeout=timeout)
                    elif code.strip().startswith("#!BASH"):
                        bash_script = re.sub(r"^#!BASH", "", code, 1).strip()
                        result = run_with_timeout(run_bash_script, [bash_script], timeout=timeout)
                    else:
                        self._inject_custom_functions_to_repl()
                        result = run_with_timeout(run_python_repl, [code], timeout=timeout)
                    
                    if len(result) > 10000:
                        result = "The output is too long... " + result[:10000]
                        
                except Exception as e:
                    result = f"Execution Error: {str(e)}"
                
                # Append observation to existing message (maintains role alternation)
                observation = f"\n<observation>{result}</observation>"
                updated_content = state["messages"][-1].content + observation
                updated_messages = state["messages"][:-1] + [AIMessage(content=updated_content)]
                
                return {"messages": updated_messages}
            
            return {}

        def routing_function(state: AgentState) -> Literal["execute", "generate", "end"]:
            next_step = state.get("next_step")
            print(f"Routing with next_step: {next_step}")
            
            if next_step == "execute":
                return "execute"
            elif next_step == "end":
                return "end"
            else:
                return "generate"  # Default to generate for any other case

        def routing_function_self_critic(
            state: AgentState,
        ) -> Literal["generate", "end"]:
            next_step = state.get("next_step")
            if next_step == "generate":
                return "generate"
            elif next_step == "end":
                return "end"
            else:
                raise ValueError(f"Unexpected next_step: {next_step}")

        def self_critic(state: AgentState) -> AgentState:
            if self.critic_count < test_time_scale_round:
                # Generate feedback based on message history
                messages = state["messages"]
                feedback_prompt = f"""
                Here is a reminder of what is the user requested: {self.user_task}
                Examine the previous executions, reaosning, and solutions.
                Critic harshly on what could be improved?
                Be specific and constructive.
                Think hard what are missing to solve the task.
                No question asked, just feedbacks.
                """
                feedback = self.llm.invoke(messages + [HumanMessage(content=feedback_prompt)])

                # Add feedback as a new message
                state["messages"].append(
                    HumanMessage(
                        content=f"Wait... this is not enough to solve the task. Here are some feedbacks for improvement:\n{feedback.content}"
                    )
                )
                self.critic_count += 1
                state["next_step"] = "generate"
            else:
                state["next_step"] = "end"

            return state

        def validate_response(state: AgentState) -> AgentState:
            """Central routing hub - parses response and determines next step."""
            
            # Get the last AI message
            last_message = state["messages"][-1]
            msg = str(last_message.content)

            # Check for incomplete tags and fix them
            if "<execute>" in msg and "</execute>" not in msg:
                msg += "</execute>"
            if "<solution>" in msg and "</solution>" not in msg:
                msg += "</solution>"
            if "<think>" in msg and "</think>" not in msg:
                msg += "</think>"

            # Convert markdown code blocks to execute tags if no other tags exist
            if "<execute>" not in msg and "<solution>" not in msg:
                python_code_pattern = r'```(?:python)?\s*\n(.*?)\n```'
                code_match = re.search(python_code_pattern, msg, re.DOTALL)
                if code_match:
                    print("Converting markdown code block to execute tag")
                    code_content = code_match.group(1).strip()
                    msg = re.sub(python_code_pattern, f'<execute>\n{code_content}\n</execute>', msg, flags=re.DOTALL)

            # Parse tags
            think_match = re.search(r"<think>(.*?)</think>", msg, re.DOTALL)
            execute_match = re.search(r"<execute>(.*?)</execute>", msg, re.DOTALL)
            answer_match = re.search(r"<solution>(.*?)</solution>", msg, re.DOTALL)

            # Handle conflicting tags by asking for clarification
            if execute_match and answer_match:
                print("Conflicting tags detected - asking for clarification")
                clarification_msg = HumanMessage(
                    content="You included both <execute> and <solution> tags. Choose only ONE: either continue with <execute> OR end with <solution>. Please regenerate."
                )
                return {
                    "messages": [clarification_msg],
                    "next_step": "generate"
                }

            # Update the cleaned message content
            updated_messages = state["messages"][:-1] + [AIMessage(content=msg.strip())]

            # Determine next step - THIS IS THE ONLY PLACE THAT SETS next_step
            if answer_match:
                return {"messages": updated_messages, "next_step": "end"}
            elif execute_match:
                return {"messages": updated_messages, "next_step": "execute"}
            elif think_match:
                return {"messages": updated_messages, "next_step": "generate"}
            else:
                # Handle parsing errors
                print("parsing error...")
                error_count = sum(
                    1 for m in state["messages"] 
                    if isinstance(m, AIMessage) and "PARSING ERROR" in m.content
                )

                if error_count >= 2:
                    print("Too many parsing errors - ending")
                    error_msg = AIMessage(content=msg + "\n\n[PARSING ERROR] Multiple failures. Ending conversation.")
                    return {"messages": [error_msg], "next_step": "end"}
                else:
                    error_msg = HumanMessage(
                        content="Your response must include either <execute>code</execute> or <solution>answer</solution>. Please regenerate."
                    )
                    return {"messages": [error_msg], "next_step": "generate"}

        # Create the workflow
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("generate", generate)
        workflow.add_node("validate", validate_response)  # New central router
        workflow.add_node("execute", execute)

        # Define the flow - clean separation of concerns
        workflow.add_edge(START, "generate")
        workflow.add_edge("generate", "validate")  # Always validate after generate
        workflow.add_edge("execute", "generate")   # After execution, generate new response

        # Only validate_response controls routing
        workflow.add_conditional_edges(
            "validate",
            lambda state: state.get("next_step", "generate"),
            {
                "execute": "execute",
                "generate": "generate",
                "end": END,
            },
        )

        # Compile
        self.app = workflow.compile()
        self.checkpointer = MemorySaver()
        self.app.checkpointer = self.checkpointer
        
        graph_png = self.app.get_graph().draw_mermaid_png()
        with open('workflow_graph_a1_edited.png', 'wb') as f:
            f.write(graph_png)
        display(Image(graph_png))

    def go(self, prompt):
        """Execute the agent with the given prompt.

        Args:
            prompt: The user's query

        """
        self.critic_count = 0
        self.user_task = prompt

        if self.use_tool_retriever:
            # Gather all available resources
            # 1. Tools from the registry
            all_tools = self.tool_registry.tools if hasattr(self, "tool_registry") else []

            # 2. Data lake items with descriptions
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

            # 3. Libraries with descriptions - use library_content_dict directly
            library_descriptions = []
            for lib_name, lib_desc in self.library_content_dict.items():
                library_descriptions.append({"name": lib_name, "description": lib_desc})

            # Add custom software items to retrieval if they exist
            if hasattr(self, "_custom_software") and self._custom_software:
                for name, info in self._custom_software.items():
                    # Check if it's not already in the library descriptions to avoid duplicates
                    if not any(lib["name"] == name for lib in library_descriptions):
                        library_descriptions.append({"name": name, "description": info["description"]})

            # Use retrieval to get relevant resources
            resources = {
                "tools": all_tools,
                "data_lake": data_lake_descriptions,
                "libraries": library_descriptions,
            }

            print(f"We will use the following resources to answer the user's query: {len(resources['tools'])} tools, {len(resources['data_lake'])} data lake items, {len(resources['libraries'])} libraries")

            # Use prompt-based retrieval with the agent's LLM
            selected_resources = self.retriever.prompt_based_retrieval(prompt, resources, llm=self.llm)
            # print(f"Selected resources: {selected_resources} after the retrieval")
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
                    # If the item already has a description, extract just the name
                    name = item.split(": ")[0]
                    selected_resources_names["data_lake"].append(name)
                else:
                    selected_resources_names["data_lake"].append(item)
            print(f"Selected data lake items: {selected_resources_names['data_lake']}")
            # Update the system prompt with the selected resources
            print("Updating the system prompt with the selected resources")
            # #########################################################
            # ####################### MOST IMPORTANT STEP ##################################
            self.update_system_prompt_with_selected_resources(selected_resources_names)
            print("System prompt updated")
            # #########################################################

        inputs = {"messages": [HumanMessage(content=prompt)], "next_step": None}
        config = {"recursion_limit": 500, "configurable": {"thread_id": 42}}
        self.log = []

        
        # here we run the whole agent graph 
        for s in self.app.stream(inputs, stream_mode="values", config=config):
            message = s["messages"][-1]
            out = pretty_print(message)
            self.log.append(out)

        return self.log, message.content

    def update_system_prompt_with_selected_resources(self, selected_resources):
        """Update the system prompt with the selected resources."""
        # Extract tool descriptions for the selected tools
        tool_desc = {}
        for tool in selected_resources["tools"]:
            # Get the module name from the tool
            if isinstance(tool, dict):
                module_name = tool.get("module", None)

                # If module is not specified, try to find it in the module2api
                if not module_name and hasattr(self, "module2api"):
                    for mod, apis in self.module2api.items():
                        for api in apis:
                            if api.get("name") == tool.get("name"):
                                module_name = mod
                                # Update the tool with the module information
                                tool["module"] = module_name
                                break
                        if module_name:
                            break

                # If still not found, use a default
                if not module_name:
                    module_name = "biomni.tool.scRNA_tools"  # Default to scRNA_tools as a fallback
                    tool["module"] = module_name
            else:
                module_name = getattr(tool, "module_name", None)

                # If module is not specified, try to find it in the module2api
                if not module_name and hasattr(self, "module2api"):
                    tool_name = getattr(tool, "name", str(tool))
                    for mod, apis in self.module2api.items():
                        for api in apis:
                            if api.get("name") == tool_name:
                                module_name = mod
                                # Set the module_name attribute
                                tool.module_name = module_name
                                break
                        if module_name:
                            break

                # If still not found, use a default
                if not module_name:
                    module_name = "biomni.tool.scRNA_tools"  # Default to scRNA_tools as a fallback
                    tool.module_name = module_name

            if module_name not in tool_desc:
                tool_desc[module_name] = []

            # Add the tool to the appropriate module
            if isinstance(tool, dict):
                # Ensure the module is included in the tool description
                if "module" not in tool:
                    tool["module"] = module_name
                tool_desc[module_name].append(tool)
            else:
                # Convert tool object to dictionary
                tool_dict = {
                    "name": getattr(tool, "name", str(tool)),
                    "description": getattr(tool, "description", ""),
                    "parameters": getattr(tool, "parameters", {}),
                    "module": module_name,  # Explicitly include the module
                }
                tool_desc[module_name].append(tool_dict)

        # Prepare data lake items with descriptions
        data_lake_with_desc = []
        for item in selected_resources["data_lake"]:
            description = self.data_lake_dict.get(item, f"Data lake item: {item}")
            data_lake_with_desc.append({"name": item, "description": description})

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
            library_content_list=selected_resources["libraries"],
            self_critic=getattr(self, "self_critic", False),
            is_retrieval=True,
            custom_tools=custom_tools if custom_tools else None,
            custom_data=custom_data if custom_data else None,
            custom_software=custom_software if custom_software else None,
        )

        # Print the raw system prompt for debugging
        print("\n" + "="*20 + " RAW SYSTEM PROMPT FROM AGENT " + "="*20)
        print(self.system_prompt)
        print("="*70 + "\n")

    def result_formatting(self, output_class, task_intention):
        self.format_check_prompt = ChatPromptTemplate.from_messages(
            [
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
            ]
        )

        checker_llm = self.format_check_prompt | self.llm.with_structured_output(output_class)
        result = checker_llm.invoke({"messages": [("user", str(self.log))]}).dict()
        return result

    def _inject_custom_functions_to_repl(self):
        """Inject custom functions into the Python REPL execution environment.
        This makes custom tools available during code execution.
        """
        if hasattr(self, "_custom_functions") and self._custom_functions:
            # Access the persistent namespace used by run_python_repl
            from biomni.tool.support_tools import _persistent_namespace

            # Inject all custom functions into the execution namespace
            for name, func in self._custom_functions.items():
                _persistent_namespace[name] = func

            # Also make them available in builtins for broader access
            import builtins

            if not hasattr(builtins, "_biomni_custom_functions"):
                builtins._biomni_custom_functions = {}
            builtins._biomni_custom_functions.update(self._custom_functions)

    def clean_conflicting_tags(self, msg):
        """Intelligently remove conflicting tags based on content."""
        execute_match = re.search(r"<execute>(.*?)</execute>", msg, re.DOTALL)
        answer_match = re.search(r"<solution>(.*?)</solution>", msg, re.DOTALL)
        
        if execute_match and answer_match:
            execute_content = execute_match.group(1).strip()
            solution_content = answer_match.group(1).strip()
            
            # If solution is very short/empty, likely accidental
            if len(solution_content) < 10:
                print("Removing short/empty solution tag")
                msg = re.sub(r"<solution>.*?</solution>", "", msg, flags=re.DOTALL)
            # If execute block is empty, likely accidental
            elif len(execute_content.strip()) < 5:
                print("Removing empty execute tag")
                msg = re.sub(r"<execute>.*?</execute>", "", msg, flags=re.DOTALL)
            # If solution contains actual code, keep execute
            elif any(keyword in solution_content for keyword in ["import", "def", "print", "="]):
                print("Solution contains code - removing solution tag, keeping execute")
                msg = re.sub(r"<solution>.*?</solution>", "", msg, flags=re.DOTALL)
            
        return msg

    def _validate_imports(self, code):
        """Validate that all imports in the code are available."""
        import_pattern = r'from\s+([\w\.]+)\s+import\s+([\w\s,]+)'
        imports = re.findall(import_pattern, code)
        
        available_functions = {}
        for module_name, apis in self.module2api.items():
            for api in apis:
                if isinstance(api, dict) and api.get('name'):
                    if module_name not in available_functions:
                        available_functions[module_name] = []
                    available_functions[module_name].append(api['name'])
        
        for module, functions in imports:
            if module not in available_functions:
                return f"Module '{module}' is not available. Available modules: {list(available_functions.keys())}"
            
            func_list = [f.strip() for f in functions.split(',')]
            for func in func_list:
                if func not in available_functions[module]:
                    return f"Function '{func}' not found in module '{module}'. Available functions: {available_functions[module]}"
        
        return None

    def _list_available_imports(self):
        """List all available import statements."""
        imports = []
        for module_name, apis in self.module2api.items():
            for api in apis:
                if isinstance(api, dict) and api.get('name'):
                    imports.append(f"from {module_name} import {api['name']}")
        return "\n".join(imports[:10]) + f"\n... and {len(imports)-10} more"
