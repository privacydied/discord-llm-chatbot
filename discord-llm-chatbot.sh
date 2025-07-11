#!/bin/sh
# Startup script for Discord LLM Chatbot

BOT_DIR="/volume1/py/discord-llm-chatbot"
LOGFILE="$BOT_DIR/bot.log"

case "$1" in
  start)
    cd "$BOT_DIR"
    . .venv/bin/activate
    nohup python main.py >> "$LOGFILE" 2>&1 &
    echo $! > "$BOT_DIR/bot.pid"
    ;;
  stop)
    if [ -f "$BOT_DIR/bot.pid" ]; then
      kill $(cat "$BOT_DIR/bot.pid") 2>/dev/null
      rm "$BOT_DIR/bot.pid"
    fi
    ;;
  *)
    echo "Usage: $0 {start|stop}"
    exit 1
    ;;
esac
exit 0

