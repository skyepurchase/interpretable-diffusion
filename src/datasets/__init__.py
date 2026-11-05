from datasets.burgers import get_burgers
from datasets.pivae import get_pivae

DATASETS = {
    "pivae": get_pivae,
    "burgers": get_burgers,
}
