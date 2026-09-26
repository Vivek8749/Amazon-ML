"""Business entity resolution: match Source 1 entities to Source 2/3 records.

Stages (one module each): preprocessing -> blocking -> features -> training
-> inference -> evaluation, orchestrated by ``pipeline``.
"""
import warnings

# Library warnings (sklearn/xgboost/pandas) are silenced for the whole package,
# including feature-worker processes, which import this package too.
warnings.filterwarnings("ignore")
