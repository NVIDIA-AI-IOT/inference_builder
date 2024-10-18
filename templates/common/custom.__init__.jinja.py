import importlib
from pathlib import Path

custom_module_table = {
    {% for item in classes %}
    "{{ item.name }}" : {
        "module": "{{ item.module }}",
        "class": "{{ item.class_name }}"
    }{% if not loop.last %},{% endif %}
    {% endfor %}
}

def create_instance(name:str, config):
    if not name in custom_module_table:
        print(f"{name} not found")
        return None
    module_name = custom_module_table[name]["module"]
    class_name = custom_module_table[name]["class"]
    try:
        # Dynamically import the module
        module = importlib.import_module(f".{module_name}", __package__)

        # Get the class from the module
        cls = getattr(module, class_name)

        # Instantiate and return the class object
        return cls(config)
    except Exception as e:
        print(f"Error creating class '{class_name}' from module '{module_name}': {e}")
        return None