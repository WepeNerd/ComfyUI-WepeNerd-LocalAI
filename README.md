# ComfyUI-WepeNerd-LocalAI

Run local prompt enhancement and image, video, or folder captioning inside ComfyUI.
One model connection serves all task nodes through your installed llama.cpp server.

| Node | What it does |
|---|---|
| **Local AI Model** | Select a GGUF model and optional vision projector |
| **Prompt Enhancer** | Expand or refine prompts using H3, Krea2, or custom instructions |
| **H3 Prompt Enhancer** | Build structured H3 prompts with control over creative freedom and image roles |
| **Image Captioner** | Caption every image in a batch |
| **Folder Captioner** | Write matching caption files beside a folder of images |
| **Video Captioner** | Caption a video using supported native input or sampled frames |

Advanced nodes expose generation settings, server lifecycle, memory controls,
status, and manual model unloading.

## Installation

From your ComfyUI `custom_nodes` directory:

```sh
git clone https://github.com/WepeNerd/ComfyUI-WepeNerd-LocalAI.git
```

Using the Python environment that runs ComfyUI:

```sh
python -m pip install -r ComfyUI-WepeNerd-LocalAI/requirements.txt
```

Install [llama.cpp's llama-server](https://github.com/ggml-org/llama.cpp) separately.
Put it on PATH, set `LLAMA_SERVER_PATH`, or select its full executable path in
**Local AI Model (Advanced)**. Keep required runtime libraries alongside the
executable. Restart ComfyUI after installation.

Place a compatible GGUF in `ComfyUI/models/LLM`. Vision tasks also need the
matching mmproj projector for that exact model. The pack does not download
models or runtimes, install GPU wheels, or start a server just by being installed.

## Quick start

Open the [prompt enhancement example](examples/prompt-enhancement.json), select
your installed model in **Local AI Model**, enter a prompt, and click **Queue**.
Preview Any displays the generated text.

For image captioning, connect **Local AI Model → Image Captioner** and supply an
IMAGE. For a dataset, connect **Local AI Model → Folder Captioner**, select your
vision model and projector, enter the image folder, and click **Queue** once.
Each image gets a same-stem UTF-8 `.txt` caption; existing-caption handling is
controlled by the node's overwrite option.

See [usage and advanced settings](docs/usage.md) and the
[folder captioning guide](docs/folder-captioning.md).

## Memory and model lifecycle

The simple model node uses an 8192-token context, requests 24 GB of free VRAM,
and releases its managed server after generation. These settings do not guarantee
that a given model will fit. Use the advanced model node to reduce context or GPU
layers and control ComfyUI VRAM handoff.

Keeping a model resident speeds up consecutive tasks but retains its memory.
Use **Unload Local AI Model** before another memory-heavy workflow when needed.
For a separate GPU, choose the device and set `comfy_vram_handoff = never`.
These settings apply to the managed server rather than changing ComfyUI's environment.

## Compatibility and troubleshooting

Node integration and CPU tests cover ComfyUI 0.34.0, Python 3.12, and Windows.
Model and modality support depend on your llama.cpp build and GGUF/projector pair.
Native video is used only when the server advertises it; otherwise the node samples
chronological frames. Generated captions should be reviewed before training.

- **Server not found:** select the executable; a GGUF is a model file, not a program.
- **Out of memory:** reduce context or GPU layers, or release other model allocations.
- **Vision request rejected:** check the matching projector and model's vision support.
- **Startup failed:** include the relevant llama-server output from ComfyUI's console
  when [reporting the issue](https://github.com/WepeNerd/ComfyUI-WepeNerd-LocalAI/issues).

This package works independently of [core](https://github.com/WepeNerd/ComfyUI-WepeNerd)
and [Experimental](https://github.com/WepeNerd/ComfyUI-WepeNerd-Experimental).
If installing core too, use its current version; the older all-in-one core
includes duplicate Local AI nodes.

Originally part of ComfyUI-WepeNerd. Licensed under [MIT](LICENSE).
