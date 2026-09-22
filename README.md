# ComfyUI-WepeNerd-LocalAI

Run local prompt enhancement and image, video, or folder captioning inside ComfyUI.
One model connection serves all task nodes through your installed llama.cpp server.

| Node | What it does |
|---|---|
| **Local AI Model** | Select a GGUF model and optional vision projector |
| **Prompt Enhancer** | Expand or refine prompts for H3, Krea 2, Qwen Image 2.1, Flux, Wan, LTX Video, SDXL, or your own instructions |
| **H3 Prompt Enhancer** | Build structured H3 prompts with control over creative freedom and image roles |
| **Image Captioner** | Caption every image in a batch, with the same skills as Folder Captioner |
| **Folder Captioner** | Write matching caption files beside a folder of images |
| **Video Captioner** | Caption a video using supported native input or sampled frames |
| **Local AI Generate** | Send any prompt (and optionally an image) to the model and get the raw reply |

Prompt Enhancer and H3 Prompt Enhancer also output an `info` string that shows what
the node actually did (mode chosen, images sent, whether the output was checked).
To change generation settings on any task node, connect **Local AI Sampling** to its
`sampling` input.

Advanced nodes expose generation settings (Local AI Sampling), full server and
memory configuration, video frame sampling, status, and manual model unloading.

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

Open the [prompt enhancement example](example_workflows/prompt-enhancement.json)
(ComfyUI also lists it in its workflow templates under this pack's name), select
your installed model in **Local AI Model**, enter a prompt, and click **Queue**.
Preview Any displays the generated text.

For image captioning, connect **Local AI Model → Image Captioner** and supply an
IMAGE. For a dataset, connect **Local AI Model → Folder Captioner**, select your
vision model and projector, enter the image folder, and click **Queue** once.
Each image gets a same-stem UTF-8 `.txt` caption; existing-caption handling is
controlled by the node's overwrite option.

See [usage and advanced settings](docs/usage.md) and the
[folder captioning guide](docs/folder-captioning.md).

For Qwen generation, precise edits, and placing image 2's subject into image 1,
see the [Qwen Image 2.1 guide](docs/qwen-image-2.1.md). The enhancer accepts a
canvas through `image` and a separate source through `reference_images`.

## Memory and model lifecycle

The simple model node uses an 8192-token context. Its **memory** setting decides
how VRAM is shared with ComfyUI:

| memory | What happens |
|---|---|
| Unload ComfyUI models first (default) | Requests 24 GB of free VRAM, usually unloading your diffusion models, and releases the LLM after each run |
| Free only what the LLM needs | Frees an estimate based on the model and projector file sizes, so other models can stay loaded; releases the LLM after each run |
| Keep LLM loaded for 5 min | Faster repeated runs; the LLM unloads after 5 idle minutes |

These settings do not guarantee that a given model will fit. Use the advanced model
node to reduce context or GPU layers and control ComfyUI VRAM handoff.

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
