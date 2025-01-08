from typing import Callable, Any
import asyncio
import inspect


class AsyncMixin:
    """Makes sync methods async-compatible"""

    async def _run_async(self, func: Callable, *args, **kwargs) -> Any:
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return await asyncio.to_thread(func, *args, **kwargs)
