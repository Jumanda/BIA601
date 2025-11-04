"""
Traditional statistical methods for feature selection
"""

from .correlation import CorrelationSelector
from .mutual_info import MutualInfoSelector
from .univariate import UnivariateSelector
from .recursive_elimination import RecursiveEliminationSelector
from .pca import PCASelector

__all__ = [
    'CorrelationSelector',
    'MutualInfoSelector',
    'UnivariateSelector',
    'RecursiveEliminationSelector',
    'PCASelector'
]

