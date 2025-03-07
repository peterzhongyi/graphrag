import os
from typing import Optional

def getenv_or_exit(env_var: str) -> str:
    """Get environment variable or exit if not found."""
    value = os.getenv(env_var)
    if value is None:
        raise ValueError(f"Environment variable {env_var} not set")
    return value

# Define custom schema for knowledge graph
custom_schema = {
    "node_types": ["entity", "concept", "event"],
    "rel_types": ["related_to", "has_property", "occurs_in"],
    "attributes": {
        "entity": ["name", "type", "description"],
        "concept": ["name", "definition"],
        "event": ["name", "date", "description"]
    }
}
