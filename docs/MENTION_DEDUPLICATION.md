# Mention Deduplication System

## Overview

The bot now includes a comprehensive mention deduplication system that prevents duplicate user mentions in replies, fixing the issue where users would see messages like `@RDR| pry @RDR| pry hello` instead of the correct `@RDR| pry hello`.

## Implementation

### Core Utilities (`bot/utils/mention_utils.py`)

The system provides several utility functions:

- **`format_mentions(users: List[str]) -> str`**: Creates a deduplicated mention string from user IDs
- **`extract_user_ids_from_mentions(text: str) -> List[str]`**: Extracts user IDs from Discord mention strings
- **`deduplicate_mentions_in_text(text: str) -> str`**: Removes duplicate mentions from text
- **`ensure_single_mention(content: str, target_user_id: str) -> str`**: Ensures a user is mentioned exactly once at the start

### Router Integration

The router (`bot/router.py`) now uses `ensure_single_mention()` instead of simple string concatenation:

```python
# Before (caused duplicates):
action.content = f"{message.author.mention} {action.content}"

# After (prevents duplicates):
action.content = ensure_single_mention(action.content, str(message.author.id))
```

## Examples

### Single User Mention
- Input: `"hello world"` + target user `123`
- Output: `"<@123> hello world"`

### Duplicate Mention Removal
- Input: `"<@123> hello <@123> world"` + target user `123`
- Output: `"<@123> hello world"`

### Multiple Users Preserved
- Input: `"<@456> hello <@123> world"` + target user `123`
- Output: `"<@123> <@456> hello world"`

## Testing

Comprehensive unit tests are provided in `tests/test_mention_utils.py` covering:

- Empty and single user cases
- Duplicate removal while preserving order
- Multiple different users
- Mixed mention formats (`<@123>` vs `<@!123>`)
- Spacing cleanup
- Edge cases (empty strings, whitespace-only content)

## Backward Compatibility

The system maintains full backward compatibility:
- Existing posting behavior is preserved
- No changes to when/how posts are emitted
- Only the mention formatting is improved
- All existing metadata and side effects remain intact

## Benefits

1. **Cleaner Messages**: No more duplicate mentions cluttering replies
2. **Better UX**: Users see clean, properly formatted mentions
3. **Idempotent**: Multiple applications don't create additional duplicates
4. **Flexible**: Handles various mention formats and edge cases
5. **Tested**: Comprehensive test coverage ensures reliability
