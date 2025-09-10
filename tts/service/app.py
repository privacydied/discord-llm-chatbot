from flask import Flask, request, send_file
from kokoro_onnx import KokoroONNXTTS
import hashlib
import os
from pathlib import Path
from pydub import AudioSegment
import onnxruntime as ort

app = Flask(__name__)

# Configuration
CACHE_DIR = Path("cache/tts")
CACHE_DIR.mkdir(exist_ok=True, parents=True)
TTS_MODEL = "kokoro-base"
MAX_TEXT_LENGTH = 500


# Initialize TTS engine
def get_optimal_providers():
    """Get optimal ONNX Runtime providers in priority order."""
    available_providers = ort.get_available_providers()

    # Priority order: CUDA > TensorRT > CPU
    preferred_providers = []

    if "CUDAExecutionProvider" in available_providers:
        cuda_options = {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB limit
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True,
        }
        preferred_providers.append(("CUDAExecutionProvider", cuda_options))

    if "TensorrtExecutionProvider" in available_providers:
        tensorrt_options = {
            "device_id": 0,
            "trt_max_workspace_size": 2147483648,  # 2GB
            "trt_fp16_enable": True,
            "trt_int8_enable": False,
        }
        preferred_providers.append(("TensorrtExecutionProvider", tensorrt_options))

    if "CPUExecutionProvider" in available_providers:
        cpu_options = {
            "intra_op_num_threads": 4,
            "inter_op_num_threads": 4,
            "omp_num_threads": 4,
        }
        preferred_providers.append(("CPUExecutionProvider", cpu_options))

    return preferred_providers


# Initialize TTS engine
tts = KokoroONNXTTS(
    model_name=TTS_MODEL,
    providers=get_optimal_providers(),
    temperature=1.8,
    top_p=0.90,
    top_k=45,
    max_length=1024,
    batch_size=1,
    sample_rate=22050,
    speed=1.0,
    alpha=1.0,
    length_penalty=1.0,
    repetition_penalty=1.0,
    num_beams=1,
    do_sample=True,
    early_stopping=False,
    pad_token_id=0,
    eos_token_id=2,
    decoder_start_token_id=1,
)


def clean_text(text: str) -> str:
    """Clean text for better TTS synthesis."""
    text = text.replace("*", "").replace("_", "").replace("`", "")
    import re

    text = re.sub(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        "",
        text,
    )
    return " ".join(text.split())[:MAX_TEXT_LENGTH]


@app.route("/synthesize", methods=["POST"])
def synthesize():
    text = request.json.get("text", "")
    if not text or len(text) > MAX_TEXT_LENGTH:
        return {"error": "Invalid text"}, 400

    clean_txt = clean_text(text)
    cache_key = hashlib.sha1(clean_txt.encode("utf-8")).hexdigest()
    ogg_path = CACHE_DIR / f"{cache_key}.ogg"

    if ogg_path.exists():
        return send_file(ogg_path, mimetype="audio/ogg")

    try:
        # Generate audio
        wav_bytes = tts.synthesize(clean_txt)

        # Write WAV to file
        wav_path = CACHE_DIR / f"{cache_key}.wav"
        with open(wav_path, "wb") as f:
            f.write(wav_bytes)

        # Convert to OGG
        audio = AudioSegment.from_wav(str(wav_path))
        audio.export(str(ogg_path), format="ogg", parameters=["-q:a", "3"])
        os.unlink(wav_path)  # Remove temp file

        return send_file(ogg_path, mimetype="audio/ogg")
    except Exception as e:
        return {"error": f"TTS generation failed: {str(e)}"}, 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
