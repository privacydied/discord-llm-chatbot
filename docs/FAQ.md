# FAQ

> **Fill me in**
> - [ ] Collect common user questions from issues.

### Why don't slash commands show up?
Discord can take up to an hour to propagate global commands. Verify the bot has `applications.commands` scope and correct intents.

### What permissions does the bot need?
At minimum: Send Messages, Attach Files, Embed Links, Read Message History.

### How do I change the command prefix?
Set `BOT_PREFIX` in the environment or `.env` file.

### Can I run the bot without OpenAI?
Yes. Set `TEXT_BACKEND=ollama` and provide a local Ollama server.

### The invite link fails. What can I do?
Ensure the permissions integer and scopes are correct, and that the bot is not already in the guild.
