import asyncio
from bot.search.factory import get_search_provider, close_search_client
from bot.search.types import SearchQueryParams, SafeSearch


async def main():
    provider = get_search_provider()
    params = SearchQueryParams(
        query="discord bot",
        max_results=5,
        safesearch=SafeSearch.MODERATE,
        locale=None,
        timeout_ms=6000,
    )
    results = await provider.search(params)
    print(f"got {len(results)} results")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r.title}\n   {r.url}\n   {r.snippet or ''}")
    await close_search_client()


if __name__ == "__main__":
    asyncio.run(main())
