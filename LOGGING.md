# Logging

Helpers emit single-line breadcrumbs using existing schema:

- `scope_resolved`: {case, scope}
- `chat_text`: {has_text, length}
- `media_intent`: boolean
- `harvest_complete`: counts of urls/attachments
- `route_selected`: final route
- `reply_target_ok`: chosen target id
- `local_context`: item and character counts after truncation
