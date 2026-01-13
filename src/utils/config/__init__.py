import yaml


def load_config(config_file):
    try:
        with open(config_file, "r") as config_file:
            return yaml.safe_load(config_file)
    except FileNotFoundError:
        print(f"Provided config file, {config_file} not found")
        quit()
    except yaml.YAMLError:
        print("Error processing the config file. Verify the YAML formatting")
        quit()
