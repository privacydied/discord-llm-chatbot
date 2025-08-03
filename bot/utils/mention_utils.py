"""
Utilities for handling Discord user mentions with deduplication support.
"""
import re
from typing import List, Set


def format_mentions(users: List[str]) -> str:
    """
    Format a list of user IDs into a deduplicated mention string.
    
    Args:
        users: List of user ID strings (can contain duplicates)
        
    Returns:
        Properly spaced/normalized mention string with duplicates removed
        
    Examples:
        format_mentions(["123", "456"]) -> "<@123> <@456>"
        format_mentions(["123", "123", "456"]) -> "<@123> <@456>"
        format_mentions([]) -> ""
    """
    if not users:
        return ""
    
    # Remove duplicates while preserving order
    seen: Set[str] = set()
    unique_users = []
    for user_id in users:
        if user_id not in seen:
            seen.add(user_id)
            unique_users.append(user_id)
    
    # Format as mentions
    mentions = [f"<@{user_id}>" for user_id in unique_users]
    return " ".join(mentions)


def extract_user_ids_from_mentions(text: str) -> List[str]:
    """
    Extract user IDs from Discord mention strings in text.
    
    Args:
        text: Text containing Discord mentions like <@123> or <@!123>
        
    Returns:
        List of user ID strings found in the text
    """
    # Match both <@123> and <@!123> formats
    pattern = r'<@!?(\d+)>'
    return re.findall(pattern, text)


def deduplicate_mentions_in_text(text: str) -> str:
    """
    Remove duplicate user mentions from text while preserving the first occurrence.
    
    Args:
        text: Text that may contain duplicate mentions
        
    Returns:
        Text with duplicate mentions removed
        
    Examples:
        deduplicate_mentions_in_text("<@123> hello <@123>") -> "<@123> hello"
        deduplicate_mentions_in_text("<@123> <@456> <@123>") -> "<@123> <@456>"
    """
    if not text:
        return text
    
    # Find all mentions and their positions
    pattern = r'<@!?(\d+)>'
    matches = list(re.finditer(pattern, text))
    
    if len(matches) <= 1:
        return text  # No duplicates possible
    
    # Track seen user IDs and positions to remove
    seen_users: Set[str] = set()
    positions_to_remove = []
    
    for match in matches:
        user_id = match.group(1)
        if user_id in seen_users:
            # Mark this duplicate mention for removal
            positions_to_remove.append((match.start(), match.end()))
        else:
            seen_users.add(user_id)
    
    # Remove duplicates from right to left to preserve indices
    result = text
    for start, end in reversed(positions_to_remove):
        # Remove the mention and any trailing space
        before = result[:start]
        after = result[end:]
        
        # Clean up spacing - remove extra spaces that might be left behind
        if before.endswith(' ') and after.startswith(' '):
            after = after.lstrip(' ')
        elif not before.endswith(' ') and not after.startswith(' ') and before and after:
            # If we removed a mention in the middle, ensure proper spacing
            if not before.endswith((' ', '\n')) and not after.startswith((' ', '\n')):
                after = ' ' + after
        
        result = before + after
    
    return result.strip()


def ensure_single_mention(content: str, target_user_id: str) -> str:
    """
    Ensure a target user is mentioned exactly once at the beginning of content.
    
    Args:
        content: The message content that may or may not contain mentions
        target_user_id: The user ID that should be mentioned once
        
    Returns:
        Content with the target user mentioned exactly once at the start
        
    Examples:
        ensure_single_mention("hello", "123") -> "<@123> hello"
        ensure_single_mention("<@123> hello", "123") -> "<@123> hello"
        ensure_single_mention("hello <@123>", "123") -> "<@123> hello"
        ensure_single_mention("<@123> hello <@123>", "123") -> "<@123> hello"
    """
    if not content or not target_user_id:
        return content
    
    target_mention = f"<@{target_user_id}>"
    target_mention_alt = f"<@!{target_user_id}>"
    
    # Remove all instances of the target user's mentions
    content_clean = re.sub(rf'<@!?{re.escape(target_user_id)}>\s*', '', content).strip()
    
    # Add single mention at the beginning
    if content_clean:
        return f"{target_mention} {content_clean}"
    else:
        return target_mention
