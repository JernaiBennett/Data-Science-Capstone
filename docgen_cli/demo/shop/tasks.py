"""Async helpers."""

import asyncio


async def retry(fn, attempts=3, delay=0.1):
    async def attempt(n):
        try:
            return True, await fn()
        except Exception as exc:
            if n == attempts - 1:
                raise
            await asyncio.sleep(delay * (2 ** n))
            return False, exc

    for n in range(attempts):
        ok, value = await attempt(n)
        if ok:
            return value


async def gather_limited(coros, limit=5):
    sem = asyncio.Semaphore(limit)

    async def run(c):
        async with sem:
            return await c

    return await asyncio.gather(*(run(c) for c in coros))
