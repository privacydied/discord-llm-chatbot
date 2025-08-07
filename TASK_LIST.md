# Multimodal Message Refactor Plan

## Notes
- Current on_message/modal processing only handles the first detected modality and returns early, skipping others.
- Goal: Sequentially process all modalities (attachments, URLs, embeds) in original message order, invoking specialized handlers for each.
- Introduce InputItem abstraction in modality.py to unify input handling.
- Implement robust error handling and timeout management for each handler.
- Enhance logging for stepwise visibility and error/timeout events.
- Add comprehensive tests for mixed-modality, timeout, error recovery, and no-input scenarios.
- Confirmed: Current implementation only processes the first detected modality and returns early, skipping others.
- All multimodal handler/text flow logic and tests are now fixed and passing.
- Change summaries added to all major modified files.
- [2025-08-07] Runtime error: process_url returns dict, not str, in _handle_general_url; causes AttributeError when .strip() is called. [REH]
- [2025-08-07] Fixed: Both _handle_general_url and PDF handler now extract text from result dicts, resolving .strip() AttributeError. [REH]

## Task List
- [x] Audit current message routing and modality handler flow.
- [x] Catalog all modality handler methods and their firing conditions.
- [x] Confirm early returns/branching that skip subsequent modalities.
- [x] Design InputItem dataclass and collection logic in modality.py.
- [x] Implement collect_input_items(message) to gather all candidate inputs in order.
- [x] Implement map_item_to_modality(item) for robust modality detection.
- [x] Refactor Router._process_multimodal_message_internal to sequentially process all items.
- [x] Refactor each _handle_* method to accept InputItem and return str.
- [x] Integrate per-item error and timeout handling with user feedback and logging.
- [x] Ensure _flow_process_text is called for each processed item.
- [x] Add and validate tests in tests/core/test_multimodal_sequence.py.
- [x] Fix multimodal handler/text flow logic and test failures.
- [x] Add Change Summary to top of each modified file.
- [x] Investigate and fix process_url/_handle_general_url return type bug.
- [x] Update logging_config.py for improved multimodal logs and close out.
