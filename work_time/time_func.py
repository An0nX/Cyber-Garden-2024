import gc
from asyncify import asyncify
import asyncio


@asyncify
def collect_garbage():
    """
    Forces a garbage collection.

    This function triggers Python's garbage collector to free up
    unused memory by reclaiming objects that are no longer reachable.
    """
    gc.collect()
