

from biomni.tool import _query_claude_for_api


result = _query_claude_for_api(prompt="what is the name of the tool that can be used to analyze circular dichroism spectra?", system_template="your are biologist", schema=None)


print(result)


