import asyncio
import httpx

URL = "https://html.duckduckgo.com/html/?q=discord+bot"

async def main():
    async with httpx.AsyncClient() as client:
        r = await client.get(URL, headers={
            "User-Agent": "Mozilla/5.0 (compatible; LLMDiscordBot/1.0; +https://example.invalid)",
        }, timeout=8.0)
        print("status:", r.status_code)
        print("url:", str(r.url))
        print("len:", len(r.text))
        print(r.text[:1200])

if __name__ == "__main__":
    asyncio.run(main())
