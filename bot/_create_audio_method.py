    def _create_audio(self, phonemes: str, voice_embedding: np.ndarray, speed: float = 1.0) -> Tuple[np.ndarray, int]:
        """
        Create audio from phonemes and voice embedding.
        
        Args:
            phonemes: Phoneme string to synthesize
            voice_embedding: Voice embedding array
            speed: Speech speed factor (1.0 = normal)
            
        Returns:
            Tuple of (audio array, sample rate)
            
        Raises:
            ValueError: If tokenization fails or produces empty token sequence
        """
        start_t = time.time()
        
        # Normalize and sanitize input text
        phonemes = normalize_text(phonemes)
        if not phonemes:
            logger.warning("Empty phonemes input, using fallback text", 
                         extra={'subsys': 'tts', 'event': 'create_audio.empty_input'})
            phonemes = "Hello."  # Fallback to ensure we generate something
        
        # Tokenize phonemes using our robust tokenization method
        try:
            token_ids, method_used = self.tokenize_text(phonemes)
            logger.debug(f"Tokenized {len(phonemes)} chars to {len(token_ids)} tokens using {method_used.value}", 
                       extra={'subsys': 'tts', 'event': 'create_audio.tokenize', 'method': method_used.value})
            
            # Verify we have non-empty token sequence
            if not token_ids or len(token_ids) == 0:
                logger.error("Tokenization produced empty token sequence", 
                           extra={'subsys': 'tts', 'event': 'create_audio.error.empty_tokens'})
                raise ValueError("Tokenization produced empty token sequence")
                
        except Exception as e:
            logger.error(f"Failed to tokenize text: {e}", 
                       extra={'subsys': 'tts', 'event': 'create_audio.error.tokenize'}, 
                       exc_info=True)
            raise ValueError(f"Failed to tokenize text: {e}")
        
        # Prepare inputs for ONNX model
        try:
            # Get input names from model
            input_names = [input.name for input in self.sess.get_inputs()]
            logger.debug(f"Model input names: {input_names}", 
                       extra={'subsys': 'tts', 'event': 'create_audio.input_names'})
            
            # Create input dictionary with required inputs
            inputs = {}
            
            # Add token IDs with the appropriate name
            if 'input_ids' in input_names:
                # Ensure input_ids is 2D as required by ONNX model
                if isinstance(token_ids, np.ndarray) and len(token_ids.shape) == 1:
                    token_ids = token_ids.reshape(1, -1)
                elif isinstance(token_ids, list):
                    token_ids = np.array(token_ids).reshape(1, -1)
                
                inputs['input_ids'] = token_ids
                logger.debug(f"Added input_ids with shape {inputs['input_ids'].shape}", 
                           extra={'subsys': 'tts', 'event': 'create_audio.input_ids'})
            elif 'tokens' in input_names:
                # Ensure tokens is 2D as required by ONNX model
                if isinstance(token_ids, np.ndarray) and len(token_ids.shape) == 1:
                    token_ids = token_ids.reshape(1, -1)
                elif isinstance(token_ids, list):
                    token_ids = np.array(token_ids).reshape(1, -1)
                
                inputs['tokens'] = token_ids
                logger.debug(f"Added tokens with shape {inputs['tokens'].shape}", 
                           extra={'subsys': 'tts', 'event': 'create_audio.tokens'})
            else:
                logger.error("No compatible token input name found in model", 
                           extra={'subsys': 'tts', 'event': 'create_audio.error.no_token_input'})
                raise ValueError("No compatible token input name found in model")
            
            # Add speed parameter if model accepts it
            if 'speed' in input_names:
                inputs['speed'] = np.array([speed], dtype=np.float32)
                logger.debug(f"Added speed={speed}", 
                           extra={'subsys': 'tts', 'event': 'create_audio.speed'})
            
            # Add style parameter if model accepts it
            if 'style' in input_names:
                # Create style tensor with correct shape and dtype
                style = np.zeros((1, 256), dtype=np.float32)
                inputs['style'] = style
                logger.debug(f"Added style with shape {style.shape}", 
                           extra={'subsys': 'tts', 'event': 'create_audio.style'})
            
            # Process voice embedding to ensure it has the correct shape
            if voice_embedding is not None:
                # Check and reshape voice embedding as needed
                if len(voice_embedding.shape) == 1:
                    # Need to reshape to 3D for model
                    if voice_embedding.shape[0] == 256:
                        # Single 256-dim vector, reshape to [1, MAX_PHONEME_LENGTH, 256]
                        expanded = np.zeros((1, MAX_PHONEME_LENGTH, 256), dtype=np.float32)
                        for i in range(MAX_PHONEME_LENGTH):
                            expanded[0, i, :] = voice_embedding
                        voice_embedding = expanded
                        logger.debug(f"Expanded voice embedding to {voice_embedding.shape}", 
                                   extra={'subsys': 'tts', 'event': 'create_audio.expand_embedding'})
                    else:
                        # Need to add batch dimension for ONNX
                        voice_embedding = voice_embedding.reshape(1, MAX_PHONEME_LENGTH, 256)
                        logger.debug(f"Reshaped voice embedding to {voice_embedding.shape}", 
                                   extra={'subsys': 'tts', 'event': 'create_audio.reshape_embedding_2d'})
                elif len(voice_embedding.shape) == 2:
                    if voice_embedding.shape == (MAX_PHONEME_LENGTH, 256):
                        # Need to add batch dimension for ONNX
                        voice_embedding = voice_embedding.reshape(1, MAX_PHONEME_LENGTH, 256)
                        logger.debug(f"Reshaped voice embedding to {voice_embedding.shape}", 
                                   extra={'subsys': 'tts', 'event': 'create_audio.reshape_embedding_2d'})
                    elif voice_embedding.shape == (1, 256):
                        # Expand to full shape needed by model
                        expanded = np.zeros((1, MAX_PHONEME_LENGTH, 256), dtype=np.float32)
                        for i in range(MAX_PHONEME_LENGTH):
                            expanded[0, i, :] = voice_embedding[0, :]
                        voice_embedding = expanded
                        logger.debug(f"Expanded voice embedding to {voice_embedding.shape}", 
                                   extra={'subsys': 'tts', 'event': 'create_audio.expand_embedding'})
                elif len(voice_embedding.shape) == 3:
                    if voice_embedding.shape[0] != 1 or voice_embedding.shape[1] != MAX_PHONEME_LENGTH or voice_embedding.shape[2] != 256:
                        # Reshape to expected dimensions
                        logger.warning(f"Unexpected voice embedding shape: {voice_embedding.shape}, reshaping", 
                                     extra={'subsys': 'tts', 'event': 'create_audio.unexpected_shape'})
                        # Try to preserve data if possible
                        if voice_embedding.size >= MAX_PHONEME_LENGTH * 256:
                            # Reshape and take first MAX_PHONEME_LENGTH * 256 elements
                            flat = voice_embedding.flatten()[:MAX_PHONEME_LENGTH * 256]
                            voice_embedding = flat.reshape(1, MAX_PHONEME_LENGTH, 256)
                        else:
                            # Not enough data, create zeros and copy what we have
                            new_embedding = np.zeros((1, MAX_PHONEME_LENGTH, 256), dtype=np.float32)
                            flat = voice_embedding.flatten()
                            new_flat = new_embedding.flatten()
                            new_flat[:min(len(flat), len(new_flat))] = flat[:min(len(flat), len(new_flat))]
                            voice_embedding = new_embedding
                        logger.debug(f"Reshaped voice embedding to {voice_embedding.shape}", 
                                   extra={'subsys': 'tts', 'event': 'create_audio.reshape_embedding_3d'})
            
                # Add speaker embedding with the appropriate name
                if 'speaker_embedding' in input_names:
                    inputs['speaker_embedding'] = voice_embedding
                    logger.debug(f"Added speaker_embedding with shape {voice_embedding.shape}", 
                               extra={'subsys': 'tts', 'event': 'create_audio.speaker_embedding'})
                elif 'spk_emb' in input_names:
                    inputs['spk_emb'] = voice_embedding
                    logger.debug(f"Added spk_emb with shape {voice_embedding.shape}", 
                               extra={'subsys': 'tts', 'event': 'create_audio.spk_emb'})
                else:
                    # Try common embedding names if none matched
                    for name in ['speaker', 'voice_embedding', 'embedding']:
                        if name in input_names:
                            inputs[name] = voice_embedding
                            logger.debug(f"Added {name} with shape {voice_embedding.shape}", 
                                       extra={'subsys': 'tts', 'event': f'create_audio.{name}'})
                            break
                    else:
                        logger.warning("No matching embedding input name found in model", 
                                     extra={'subsys': 'tts', 'event': 'create_audio.no_embedding_input'})
            
            # Log final inputs for debugging
            input_shapes = {k: v.shape for k, v in inputs.items()}
            input_dtypes = {k: str(v.dtype) for k, v in inputs.items()}
            logger.debug(f"ONNX inputs: shapes={input_shapes}, dtypes={input_dtypes}", 
                       extra={'subsys': 'tts', 'event': 'create_audio.inputs'})
            
            # Run inference
            try:
                outputs = self.sess.run(None, inputs)
                
                # Extract audio from outputs
                audio = outputs[0][0]  # Shape: [batch_size=1, audio_length]
                
                # Log output shape and stats
                logger.debug(f"Output shape: {audio.shape}, min: {audio.min():.4f}, max: {audio.max():.4f}, mean: {audio.mean():.4f}", 
                           extra={'subsys': 'tts', 'event': 'create_audio.output_stats'})
                
                # Check if audio is silent (all zeros or very close to zero)
                if np.allclose(audio, 0, atol=1e-6):
                    logger.error("Generated audio contains all zeros", 
                               extra={'subsys': 'tts', 'event': 'create_audio.error.silent_audio'})
                    raise ValueError("Generated audio contains all zeros")
                
                # Check if audio is too quiet (max amplitude too low)
                if np.max(np.abs(audio)) < 0.01:
                    logger.warning("Generated audio is very quiet", 
                                 extra={'subsys': 'tts', 'event': 'create_audio.warning.quiet_audio'})
                
                # Calculate inference time
                inference_time = time.time() - start_t
                logger.debug(f"Inference time: {inference_time:.2f}s", 
                           extra={'subsys': 'tts', 'event': 'create_audio.inference_time'})
                
                # Calculate audio duration in seconds
                audio_duration = len(audio) / SAMPLE_RATE
                logger.debug(f"Audio duration: {audio_duration:.2f}s",
                           extra={'subsys': 'tts', 'event': 'create_audio.duration'})
                
                # Ensure audio is at least 1 second long for short inputs
                if audio_duration < 1.0 and len(audio) > 0:
                    # Pad with silence to reach 1 second
                    samples_needed = SAMPLE_RATE - len(audio)
                    if samples_needed > 0:
                        silence = np.zeros(samples_needed, dtype=audio.dtype)
                        audio = np.concatenate([audio, silence])
                        logger.debug(f"Padded audio to 1 second: new shape={audio.shape}",
                                   extra={'subsys': 'tts', 'event': 'create_audio.padding'})
                
                # Log performance metrics
                audio_duration = len(audio) / SAMPLE_RATE
                create_duration = time.time() - start_t
                speedup_factor = audio_duration / create_duration if create_duration > 0 else 0
                
                logger.debug(
                    f"Created {audio_duration:.2f}s audio for {len(phonemes)} phonemes in {create_duration:.2f}s ({speedup_factor:.2f}x real-time)",
                    extra={'subsys': 'tts', 'event': 'create_audio.complete'}
                )
                
                # Always return a tuple of (audio, sample_rate)
                return audio, SAMPLE_RATE
                
            except Exception as e:
                logger.error(f"Error during ONNX inference: {e}", 
                           extra={'subsys': 'tts', 'event': 'create_audio.error.inference'}, 
                           exc_info=True)
                # Return a short silent audio segment as fallback
                return np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE  # 1 second of silence
            
        except Exception as e:
            logger.error(f"Error in _create_audio: {e}", 
                       extra={'subsys': 'tts', 'event': 'create_audio.error'}, 
                       exc_info=True)
            # Return a short silent audio segment as fallback
            return np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE  # 1 second of silence
