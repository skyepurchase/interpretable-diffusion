from datasets.burgers import get_burgers

from .mnist import get_MNIST
from .pdebench import get_pdebench
from .pivae import get_pivae

DATASETS = {
    "mnist": get_MNIST,
    "pivae": get_pivae,
    "pdebench": get_pdebench,
    "burgers": get_burgers,
}
