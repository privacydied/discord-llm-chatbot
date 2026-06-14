    except httpx.HTTPStatusError as e:
        # Surface HTTP errors (e.g., 429 Too Many Requests from OpenRouter) as retriable APIError
        status = e.response.status_code if e.response is not None else "unknown"
        retry_after = None
        try:
            if getattr(e, "response", None) is not None:
                hdrs = e.response.headers
                ra = hdrs.get("retry-after") or hdrs.get("Retry-After")
                if ra is not None:
                    retry_after = float(ra)
                else:
                    # OpenRouter-style reset headers (epoch or delta seconds)
                    reset = hdrs.get("x-ratelimit-reset") or hdrs.get("X-RateLimit-Reset") or hdrs.get("rate-limit-reset")
                    if reset is not None:
                        try:
                            val = float(reset)
                            now = time.time()
                            retry_after = val - now if val > now + 1 else val
                            if retry_after < 0:
                                retry_after = 0.0
                        except (ValueError, TypeError):
                            retry_after = None
                except (AttributeError, TypeError, ValueError):
                    retry_after = None
        extra = f" (retry-after={retry_after}s)" if retry_after is not None else ""
        logger.warning(f"OpenAI HTTP error: {status} {e}{extra}")
        err = APIError(f"HTTP {status}: {e!s}{extra}")
        try:
            if retry_after is not None:
                err.retry_after_seconds = retry_after
        except (AttributeError, TypeError) as exc:
            logger.debug(f"Failed to set retry_after_seconds: {exc}")
        raise err
    except APIError as e:
        # Already normalized, don't double-wrap or spam error-level logs
        logger.warning(f"[OpenAI] Retriable APIError: {e}")
        raise

    except Exception as e:
        # Get detailed error information
        error_type = type(e).__name__
        error_msg = str(e) or "No error message"
        error_details = f"{error_type}: {error_msg}"

        logger.exception(f"Unexpected error in generate_openai_response: {error_details}")
        logger.debug(
            "Traceback for unexpected error in generate_openai_response",
            exc_info=True,
        )
        msg_0 = f"Failed to generate OpenAI response: {error_details}"
        raise APIError(msg_0)


async def get_base64_image(image_url: str) -> str:
    """Process image from URL or file path and convert to base64 data URI.
    CHANGE: Enhanced to handle both URLs and file paths.
    """
    logger.debug(f"📥 Processing image from: {image_url}")

    # Handle file paths (file:// protocol or direct file path)
    if image_url.startswith("file://") or os.path.exists(image_url):
        try:
            # Extract actual path from file:// URL if needed
            file_path = image_url.removeprefix("file://")

            # Verify file exists
            if not os.path.exists(file_path):
                error_msg = f"File not found: {file_path}"
                logger.error(error_msg)
                raise APIError(error_msg)

            # Read file and encode to base64
            with open(file_path, "rb") as f:
                data = f.read()

            # Determine content type from file extension
            ext = os.path.splitext(file_path)[1].lower()
            content_type = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png" if ext == ".png" else "image/webp" if ext == ".webp" else "image/gif" if ext == ".gif" else "image/png"  # Default to PNG

            base64_data = base64.b64encode(data).decode("utf-8")
            logger.debug(f"✅ Image loaded from file: size={len(data)} bytes, type={content_type}")
            return f"data:{content_type};base64,{base64_data}"

        except Exception as e:
            error_msg = f"Failed to process image file: {e}"
            logger.error(error_msg, exc_info=True)
            raise APIError(error_msg)

    # Handle HTTP/HTTPS URLs
    elif image_url.startswith(("http://", "https://")):
        try:
            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(total=10)
                async with session.get(image_url, timeout=timeout) as response:
                    if response.status == 200:
                        data = await response.read()
                        base64_data = base64.b64encode(data).decode("utf-8")
                        content_type = response.headers.get("Content-Type", "image/png")
                        logger.debug(f"✅ Image downloaded: size={len(data)} bytes, type={content_type}")