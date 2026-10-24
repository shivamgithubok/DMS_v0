# models/

Nothing here is committed: model files are downloaded by

    python3 tools/download_models.py            # required (~115 MB, SHA-256 verified)
    python3 tools/download_models.py --optional # MediaPipe landmarker, YAMNet, Vosk
    python3 tools/download_models.py --vlm      # Qwen3.5-0.8B Q4_K_M + mmproj -> models/vlm/
    python3 tools/download_models.py --list     # what is present / missing

and TensorRT engines (`*.engine`) are built on the Jetson by `tools/build_engines.sh`.
