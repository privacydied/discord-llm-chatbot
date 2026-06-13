import unittest

import pytest

from bot.router import BotAction, ResponseMessage


@pytest.mark.skip(reason="ResponseMessage is now a standalone class, not a BotAction alias")
class TestResponseMessageCompat(unittest.TestCase):
    def test_alias(self) -> None:
        """Test that ResponseMessage is an alias for BotAction."""
        assert ResponseMessage is BotAction

    def test_text_property(self) -> None:
        """Test backward compatibility for text property."""
        action = BotAction(content="Hello")
        assert action.text == "Hello"

        # Also test via alias
        response = ResponseMessage(content="World")
        assert response.text == "World"

    def test_embed_property(self) -> None:
        """Test backward compatibility for embed property."""
        from discord import Embed

        embed1 = Embed(title="Test")
        action = BotAction(embeds=[embed1])
        assert action.embed == embed1

        # Test when no embeds
        action = BotAction()
        assert action.embed is None

    def test_file_property(self) -> None:
        """Test backward compatibility for file property."""
        from io import BytesIO

        from discord import File

        file1 = File(BytesIO(b"hello"), filename="test.txt")
        action = BotAction(files=[file1])
        assert action.file == file1

        # Test when no files
        action = BotAction()
        assert action.file is None


if __name__ == "__main__":
    unittest.main()
