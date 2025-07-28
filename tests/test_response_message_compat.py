import unittest
from bot.router import BotAction, ResponseMessage

class TestResponseMessageCompat(unittest.TestCase):
    def test_alias(self):
        """Test that ResponseMessage is an alias for BotAction."""
        self.assertIs(ResponseMessage, BotAction)
        
    def test_text_property(self):
        """Test backward compatibility for text property."""
        action = BotAction(content="Hello")
        self.assertEqual(action.text, "Hello")
        
        # Also test via alias
        response = ResponseMessage(content="World")
        self.assertEqual(response.text, "World")
        
    def test_embed_property(self):
        """Test backward compatibility for embed property."""
        from discord import Embed
        embed1 = Embed(title="Test")
        action = BotAction(embeds=[embed1])
        self.assertEqual(action.embed, embed1)
        
        # Test when no embeds
        action = BotAction()
        self.assertIsNone(action.embed)
        
    def test_file_property(self):
        """Test backward compatibility for file property."""
        from discord import File
        file1 = File("test.txt")
        action = BotAction(files=[file1])
        self.assertEqual(action.file, file1)
        
        # Test when no files
        action = BotAction()
        self.assertIsNone(action.file)

if __name__ == '__main__':
    unittest.main()
