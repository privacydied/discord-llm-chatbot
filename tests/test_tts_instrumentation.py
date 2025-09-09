"""Tests for TTS instrumentation utilities."""

import unittest
from pathlib import Path
from unittest.mock import patch

from bot.tts_instrumentation import (
    log_tts_config,
    log_phonemiser_selection,
    log_voice_loading,
    log_tts_generation,
    log_tts_error,
    log_gibberish_detection,
    log_cache_event,
    timed_function,
    get_tts_metrics,
    reset_tts_metrics,
)


class TestTTSInstrumentation(unittest.TestCase):
    """Test cases for TTS instrumentation utilities."""

    def setUp(self):
        """Set up test fixtures."""
        # Reset metrics before each test
        reset_tts_metrics()

    def test_log_tts_config(self):
        """Test logging TTS configuration."""
        with patch('logging.Logger.info') as mock_info:
            # Test with regular config
            config = {
                'TTS_LANGUAGE': 'en',
                'TTS_VOICE': 'test_voice',
                'TTS_BACKEND': 'kokoro',
                'TTS_MODEL_PATH': '/path/to/model',
            }
            log_tts_config(config)
            
            # Check that info was logged
            self.assertTrue(mock_info.called)
            self.assertEqual(mock_info.call_count, 4)  # Main config + 3 specific items
            
            # Test with sensitive info
            config_with_sensitive = {
                'TTS_LANGUAGE': 'en',
                'API_KEY': 'secret_key',
                'PASSWORD': 'secret_password',
            }
            mock_info.reset_mock()
            log_tts_config(config_with_sensitive)
            
            # Check that sensitive info was filtered
            self.assertTrue(mock_info.called)
            for call_args in mock_info.call_args_list:
                args, kwargs = call_args
                if 'extra' in kwargs and 'config' in kwargs['extra']:
                    self.assertNotIn('API_KEY', kwargs['extra']['config'])
                    self.assertNotIn('PASSWORD', kwargs['extra']['config'])

    def test_log_phonemiser_selection(self):
        """Test logging phonemiser selection."""
        with patch('logging.Logger.info') as mock_info:
            available = {
                'espeak': True,
                'phonemizer': False,
                'g2p_en': True,
            }
            log_phonemiser_selection('en', 'espeak', available)
            
            # Check that info was logged
            self.assertTrue(mock_info.called)
            args, kwargs = mock_info.call_args
            self.assertIn('espeak', args[0])
            self.assertIn('en', args[0])
            self.assertEqual(kwargs['extra']['phonemiser'], 'espeak')
            self.assertEqual(kwargs['extra']['language'], 'en')
            self.assertEqual(kwargs['extra']['available'], available)

    def test_log_voice_loading(self):
        """Test logging voice loading."""
        with patch('logging.Logger.debug') as mock_debug:
            log_voice_loading('test_voice', (256,), 1.234)
            
            # Check that debug was logged
            self.assertTrue(mock_debug.called)
            args, kwargs = mock_debug.call_args
            self.assertIn('test_voice', args[0])
            self.assertIn('(256,)', args[0])
            self.assertEqual(kwargs['extra']['voice_id'], 'test_voice')
            self.assertEqual(kwargs['extra']['vector_shape'], (256,))
            self.assertAlmostEqual(kwargs['extra']['vector_norm'], 1.234)

    def test_log_tts_generation(self):
        """Test logging TTS generation."""
        with patch('logging.Logger.info') as mock_info:
            # Get initial metrics
            initial_metrics = get_tts_metrics()
            
            # Log a TTS generation
            log_tts_generation(
                'This is a test text',
                'test_voice',
                Path('/tmp/test_output.wav'),
                150.5
            )
            
            # Check that info was logged
            self.assertTrue(mock_info.called)
            args, kwargs = mock_info.call_args
            self.assertIn('test_voice', args[0])
            self.assertIn('150.5', args[0])
            self.assertEqual(kwargs['extra']['voice_id'], 'test_voice')
            self.assertEqual(kwargs['extra']['text_length'], 19)
            self.assertEqual(kwargs['extra']['output_path'], '/tmp/test_output.wav')
            self.assertAlmostEqual(kwargs['extra']['duration_ms'], 150.5)
            
            # Check that metrics were updated
            updated_metrics = get_tts_metrics()
            self.assertEqual(updated_metrics['tts_generation_count'], initial_metrics['tts_generation_count'] + 1)
            self.assertAlmostEqual(updated_metrics['tts_generation_time_total'], initial_metrics['tts_generation_time_total'] + 0.1505)

    def test_log_tts_error(self):
        """Test logging TTS error."""
        with patch('logging.Logger.error') as mock_error:
            # Get initial metrics
            initial_metrics = get_tts_metrics()
            
            # Log a TTS error
            log_tts_error('TestError', 'This is a test error', {'detail': 'test_detail'})
            
            # Check that error was logged
            self.assertTrue(mock_error.called)
            args, kwargs = mock_error.call_args
            self.assertIn('TestError', args[0])
            self.assertIn('This is a test error', args[0])
            self.assertEqual(kwargs['extra']['error_type'], 'TestError')
            self.assertEqual(kwargs['extra']['error_message'], 'This is a test error')
            self.assertEqual(kwargs['extra']['detail'], 'test_detail')
            
            # Check that metrics were updated
            updated_metrics = get_tts_metrics()
            self.assertEqual(updated_metrics['tts_generation_errors'], initial_metrics['tts_generation_errors'] + 1)

    def test_log_gibberish_detection(self):
        """Test logging gibberish detection."""
        with patch('logging.Logger.warning') as mock_warning:
            # Get initial metrics
            initial_metrics = get_tts_metrics()
            
            # Log a gibberish detection
            metrics = {
                'zero_crossing_rate': 0.5,
                'mean_amplitude': 0.001,
                'clipping_ratio': 0.0,
            }
            log_gibberish_detection(metrics)
            
            # Check that warning was logged
            self.assertTrue(mock_warning.called)
            args, kwargs = mock_warning.call_args
            self.assertIn('Gibberish audio detected', args[0])
            self.assertEqual(kwargs['extra']['zero_crossing_rate'], 0.5)
            self.assertEqual(kwargs['extra']['mean_amplitude'], 0.001)
            self.assertEqual(kwargs['extra']['clipping_ratio'], 0.0)
            
            # Check that metrics were updated
            updated_metrics = get_tts_metrics()
            self.assertEqual(updated_metrics['tts_gibberish_detected'], initial_metrics['tts_gibberish_detected'] + 1)

    def test_log_cache_event(self):
        """Test logging cache events."""
        with patch('logging.Logger.debug') as mock_debug:
            # Get initial metrics
            initial_metrics = get_tts_metrics()
            
            # Log a cache hit
            log_cache_event('test_hash_1', True)
            
            # Check that debug was logged
            self.assertTrue(mock_debug.called)
            args, kwargs = mock_debug.call_args
            self.assertIn('cache hit', args[0])
            self.assertIn('test_hash_1', args[0])
            self.assertEqual(kwargs['extra']['text_hash'], 'test_hash_1')
            self.assertEqual(kwargs['extra']['event'], 'cache.hit')
            
            # Check that metrics were updated
            updated_metrics = get_tts_metrics()
            self.assertEqual(updated_metrics['tts_cache_hits'], initial_metrics['tts_cache_hits'] + 1)
            
            # Reset mock
            mock_debug.reset_mock()
            
            # Log a cache miss
            log_cache_event('test_hash_2', False)
            
            # Check that debug was logged
            self.assertTrue(mock_debug.called)
            args, kwargs = mock_debug.call_args
            self.assertIn('cache miss', args[0])
            self.assertIn('test_hash_2', args[0])
            self.assertEqual(kwargs['extra']['text_hash'], 'test_hash_2')
            self.assertEqual(kwargs['extra']['event'], 'cache.miss')
            
            # Check that metrics were updated
            updated_metrics = get_tts_metrics()
            self.assertEqual(updated_metrics['tts_cache_misses'], initial_metrics['tts_cache_misses'] + 1)

    def test_timed_function(self):
        """Test timed function decorator."""
        with patch('logging.Logger.debug') as mock_debug:
            # Define a test function
            @timed_function
            def test_func(x, y):
                return x + y
            
            # Call the function
            result = test_func(1, 2)
            
            # Check that the function worked correctly
            self.assertEqual(result, 3)
            
            # Check that debug was logged
            self.assertTrue(mock_debug.called)
            args, kwargs = mock_debug.call_args
            self.assertIn('test_func', args[0])
            self.assertIn('executed', args[0])
            self.assertEqual(kwargs['extra']['function'], 'test_func')
            self.assertIn('duration_ms', kwargs['extra'])

    def test_timed_function_error(self):
        """Test timed function decorator with error."""
        with patch('logging.Logger.error') as mock_error:
            # Define a test function that raises an exception
            @timed_function
            def test_func_error():
                raise ValueError('Test error')
            
            # Call the function and expect an exception
            with self.assertRaises(ValueError):
                test_func_error()
            
            # Check that error was logged
            self.assertTrue(mock_error.called)
            args, kwargs = mock_error.call_args
            self.assertIn('test_func_error', args[0])
            self.assertIn('failed', args[0])
            self.assertIn('Test error', args[0])
            self.assertEqual(kwargs['extra']['function'], 'test_func_error')
            self.assertIn('duration_ms', kwargs['extra'])
            self.assertEqual(kwargs['extra']['error'], 'Test error')

    def test_get_tts_metrics(self):
        """Test getting TTS metrics."""
        # Initial metrics should all be zero
        metrics = get_tts_metrics()
        self.assertEqual(metrics['tts_generation_count'], 0)
        self.assertEqual(metrics['tts_generation_time_total'], 0.0)
        self.assertEqual(metrics['tts_generation_errors'], 0)
        self.assertEqual(metrics['tts_gibberish_detected'], 0)
        self.assertEqual(metrics['tts_cache_hits'], 0)
        self.assertEqual(metrics['tts_cache_misses'], 0)
        self.assertEqual(metrics['tts_generation_time_avg'], 0.0)
        self.assertEqual(metrics['tts_cache_hit_rate'], 0.0)
        self.assertEqual(metrics['tts_error_rate'], 0.0)
        
        # Simulate some activity
        log_tts_generation('Test 1', 'voice1', Path('/tmp/test1.wav'), 100.0)
        log_tts_generation('Test 2', 'voice1', Path('/tmp/test2.wav'), 200.0)
        log_tts_error('TestError', 'Test error')
        log_cache_event('hash1', True)
        log_cache_event('hash2', True)
        log_cache_event('hash3', False)
        
        # Check updated metrics
        metrics = get_tts_metrics()
        self.assertEqual(metrics['tts_generation_count'], 2)
        self.assertAlmostEqual(metrics['tts_generation_time_total'], 0.3)
        self.assertEqual(metrics['tts_generation_errors'], 1)
        self.assertEqual(metrics['tts_cache_hits'], 2)
        self.assertEqual(metrics['tts_cache_misses'], 1)
        self.assertAlmostEqual(metrics['tts_generation_time_avg'], 0.15)
        self.assertAlmostEqual(metrics['tts_cache_hit_rate'], 2/3)
        self.assertAlmostEqual(metrics['tts_error_rate'], 0.5)

    def test_reset_tts_metrics(self):
        """Test resetting TTS metrics."""
        # Simulate some activity
        log_tts_generation('Test', 'voice1', Path('/tmp/test.wav'), 100.0)
        log_tts_error('TestError', 'Test error')
        log_cache_event('hash', True)
        
        # Check that metrics were updated
        metrics_before = get_tts_metrics()
        self.assertGreater(metrics_before['tts_generation_count'], 0)
        
        # Reset metrics
        reset_tts_metrics()
        
        # Check that metrics were reset
        metrics_after = get_tts_metrics()
        self.assertEqual(metrics_after['tts_generation_count'], 0)
        self.assertEqual(metrics_after['tts_generation_time_total'], 0.0)
        self.assertEqual(metrics_after['tts_generation_errors'], 0)
        self.assertEqual(metrics_after['tts_gibberish_detected'], 0)
        self.assertEqual(metrics_after['tts_cache_hits'], 0)
        self.assertEqual(metrics_after['tts_cache_misses'], 0)


if __name__ == '__main__':
    unittest.main()
