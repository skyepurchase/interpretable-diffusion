import re
from os import PathLike

import jax
import yaml

Distribution = tuple[jax.Array, jax.Array]


def load_config(config_file: PathLike) -> dict:
    """
    Load and parse the provided config file.

    Parameters
    ----------
    config_file : PathLike
        The path to the config to load

    Returns
    -------
    config : ExperimentConfig
        The config for the given experiment
    """

    # Source - https://stackoverflow.com/a/48452275
    # Posted by Olivier
    # Retrieved 2026-08-07, License - CC BY-SA 3.0

    # this is to convert the string written as a tuple into a python tuple
    def yml_tuple_constructor(loader, node):
        def parse_tup_el(el):
            # try to convert into int or float else keep the string
            if el.isdigit():
                return int(el)
            try:
                return float(el)
            except ValueError:
                return el

        value = loader.construct_scalar(node)
        # remove the ( ) from the string
        tup_elements = value[1:-1].split(",")
        # remove the last element if the tuple was written as (x,b,)
        if tup_elements[-1] == "":
            tup_elements.pop(-1)
        tup = tuple(map(parse_tup_el, tup_elements))
        return tup

    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            """^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )

    loader.add_constructor("!tuple", yml_tuple_constructor)
    loader.add_implicit_resolver(
        "!tuple", re.compile(r"\(([^,\W]{,},){,}[^,\W]*\)"), list("(")
    )

    print("Attempting to load config...")
    try:
        with open(config_file, "r") as file:
            print(f"Opened config file {config_file}.")
            print("Attempting to parse...")
            return yaml.load(file, Loader=loader)
    except FileNotFoundError:
        print(f"Provided config file, {config_file} not found")
        quit()
    except yaml.YAMLError:
        print("Error processing the config file. Verify the YAML formatting")
        quit()
