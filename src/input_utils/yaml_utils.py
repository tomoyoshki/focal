import yaml


def load_yaml(file_path):
    """Load the YAML config file

    Args:
        file_path (_type_): _description_
    """
    with open(file_path, "r", errors="ignore") as stream:
        yaml_data = yaml.safe_load(stream)

    return yaml_data
