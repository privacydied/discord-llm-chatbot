import runpy
import sys
import re

if __name__ == "__main__":
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(runpy.run_module("bot.main", run_name="__main__", alter_sys=True))
