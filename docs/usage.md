# Local AI usage

## Folder captioning

Connect **Local AI Model â†’ Folder Captioner**, select a compatible vision model
and projector, enter an absolute image folder path, choose a **skill**, and click
**Queue** once. The node runs as an output node without another connection.
`photo.001.jpg` becomes `photo.001.txt` in the same directory, containing only its
UTF-8 caption. Images stay unchanged. The model is acquired once for the batch,
images are decoded one at a time, and each completed caption is saved immediately.
The connected model's release/keep-alive policy applies when the batch ends.

| Skill | Default captioning strategy |
|---|---|
| `Krea 2 - Character likeness` | Use the character trigger; describe pose, clothing, expression, surroundings, and other changeable details. Leave fixed likeness traits implicit. |
| `Krea 2 - Style` | Describe scene content while leaving the target visual treatment implicit. |
| `Krea 2 - Refiner` | Name the known concept and describe visible distinguishing structure and details. This prepares training captions; it does not refine images. |
| `General caption` / `Custom` | General visual description, or the complete skill supplied in `instruction`. |

`trigger_word` is optional and must be reproduced exactly when supplied. Use
`concept_context` for facts and the learning goal shared by the folder, such as
the target character, a specific car model, or a user-provided ethnicity label.
The skills instruct the LLM not to infer ethnicity, nationality, or identity from
appearance. `instruction` adds your direction to a bundled skill. For mixed
identities or concepts, process separate folders with their own context.

**Skip** preserves existing captions, including empty `.txt` files. Queue again to
resume or process newly added images. **Overwrite** replaces a caption only after
a complete response is ready. `include_subfolders` keeps captions beside each
image in its own subfolder; symlinks and junctions are not traversed. Conflicting
stems such as `photo.jpg` and `photo.png` in one folder are rejected before any
generation. Unreadable/multipage images or failed/incomplete responses stop the
batch with the current filename; already saved captions remain available.

Supported still formats: JPEG, PNG, WebP, BMP, TIFF. Images are EXIF-oriented and
resized proportionally to `image_max_edge` (1024 by default) for the LLM only.
Increase `max_tokens` (768 by default) for longer captions; the model context must
also accommodate the skill and image. There is no background folder watcher.

The three Krea skills are editable local Markdown templates in `skills/` and are
reloaded when their files change. Their choices are research-informed defaults,
not proven optimal Krea2 LoRA recipes. See the [folder captioning guide](folder-captioning.md)
for sources, trigger guidance, and the distinction between character/style
isolation and concept refinement. Restart ComfyUI after installing the new node.

## H3 prompt enhancement

`H3 Prompt Enhancer` preserves scene continuity: one `[Shot N]` block can contain several cuts or camera angles. A new tag marks a major scene/sequence boundary. Explicit action detail takes precedence over task defaults; `Strict` strengthens binding without overriding `Semantic`. Enhancement controls expression and organization; creative freedom controls permission to invent.

The optional inputs are:

| Input | Use |
|---|---|
| `duration_seconds` | Effective generated clip length. `0` means unspecified; new events use relative timing and first/last-frame alignment uses a semantic ending instead of an invented duration. |
| `reference_context` | Supplied asset aliases, roles, and constraints. For example: `<Picture 1>: replacement identity. <Video 1>: source motion and camera. Video audio is not enabled.` |
| `max_tokens` | Output budget, default 2048. Increase for long prompts if the model context has room. |
| `creative_freedom` | `Preserve` (default) clarifies existing ideas. `Fill in details` enriches an outline with setting, atmosphere, camera, sound, and natural action progression. `Develop scenario` can also add supporting beats, reactions, and transitions. |
| `image` | Optional IMAGE from Load Image or another image node. Leave `prompt` blank to create a video idea from the image alone, or add text direction. Requires a vision-capable local model and its matching projector. |
| `image_role` | `Visual inspiration` (default): use visible subjects, mood, or style to create a prompt; `First frame`: develop motion from the image's opening state; `Reference image`: retain referenced identity or visual characteristics in a new scenario. |

For a basic outline such as `A traveler finds an abandoned lighthouse`, use `Fill in details` with `Smart`. Choose `Develop scenario` when you also want the LLM to develop what happens. Explicit instructions, reference constraints, authored scene plans, and supplied dialogue/text take priority at every level. New dialogue, visible wording, lyrics, and music require a request. Expansion respects the clip duration and adds useful content rather than targeting a longer word count. Sampling stays the same across freedom levels; the permission is conveyed through the model instructions.

With an image connected, `Auto` selects T2V for Visual inspiration, I2V for First frame, or Ref2V for Reference image. An explicit mode takes priority. Without an image, `Auto` infers from text. The enhancer can inspect only the images connected to it and cannot inspect the downstream graph. Specific modes load only their relevant appendix; the generic H3 skill remains complete.

Connect `Load Image â†’ image` and `Local AI Model â†’ model`, choose an image role, and run with a blank prompt for an original short video scenario. For image-only input, the H3 node automatically uses `Develop scenario` when creative freedom is left on `Preserve`; text plus image follows your direction and chosen creative freedom. The image is sent to the local LLM only: connect it separately to H3 when using it as a first frame or generation reference. The same image inputs are available on `Prompt Enhancer` and `Prompt Enhancer (Advanced)`, including their H3 skill/style.

Image batches are sent together for one prompt. In First frame mode, only the first image anchors the opening; additional images supply visual guidance. Reference images default to `<Picture 1>`, `<Picture 2>`, etc. in attachment order when no Picture aliases are supplied. For a different workflow numbering, state the attachment mapping in `reference_context`, e.g. `Attached image 1 = <Picture 0>: character identity.` Each image uses the existing in-memory JPEG encoder with a maximum edge of 1024 pixels.

For text-only inference, the dedicated H3 node asks the local server to render and tokenize the chat, then checks that input plus output budget fits. Image requests use the backend's multimodal context handling; they skip the text-only token preflight. Increase the model context or reduce the output budget if images leave too little room. Returned prompts are checked for complete sections, explicit scene plans, identifiable dialogue/text literals, supplied subject/speaker labels, reference asset numbering, and timed-event bounds. Invalid output raises an actionable error. These checks cannot prove that every creative instruction was followed. Token-truncated output is rejected by the shared backend rather than returned as a usable prompt.

Both simple prompt enhancers recognize Qwen 3.8 27B from the model filename, including compatible Huihui derivatives. They explicitly set `enable_thinking=false` and use [Qwen's instruct sampling recommendation](https://huggingface.co/Qwen/Qwen3.8-27B): temperature 0.7, top-p 0.8, top-k 20, min-p 0, repetition penalty 1.0, and presence penalty 1.5. Frequency penalty stays at zero. Other models retain the existing conservative sampling. A renamed model file that omits its Qwen version/size will use those conservative defaults.


Video auto mode checks llama-server `/props`: it uses typed native `input_video` only when video support is explicit, otherwise it sends timestamped JPEG frames. Missing metadata is treated as unknown and falls back conservatively. File-backed clips use PyAV seek sampling, so memory scales with selected frames rather than total clip length. Audio and dialogue are not inferred.

## Qwen Image 2.1

`Prompt Enhancer` includes Generate, Edit, and Image 2 into Image 1 skills for
Qwen Image 2.1. Use `image` for the canvas and `reference_images` for additional
sources. Connect the same originals to your Qwen workflow in the same order.
See the [researched prompting and editing guide](qwen-image-2.1.md) for setup,
common tasks, source links, and preservation limits.

## Local AI / Advanced

**Category:** `WepeNerd/Local AI/Advanced`

Use the advanced nodes for raw generation, a custom context size, GPU layers, KV cache overrides, Flash Attention overrides, keep-alive, a custom server executable, secondary-GPU selection, native/sampled video controls, status, or manual unloading.

| Node | Purpose |
|---|---|
| `Local AI Model (Advanced)` | Full model, server, memory, and lifecycle configuration |
| `Local AI Generate` | General text generation with optional image input |
| `Prompt Enhancer (Advanced)` | Legacy styles and sampler controls, including H3 and Krea 2 |
| `Image Captioner (Advanced)` | Caption cleanup, encoding, and sampler controls |
| `Video Captioner (Advanced)` | Native/sampled modes and sampling controls |
| `Local AI Status` | Report the managed server and advertised modalities |
| `Unload Local AI Model` | Stop a resident keep-alive server |

`release_after_generate = false` is an advanced speed option for consecutive calls. `keep_alive_seconds` can release an idle server automatically; zero means manual indefinite keep-alive. While resident, external llama.cpp VRAM is invisible to ComfyUI, so run `Unload Local AI Model` before returning to a heavy diffusion or video branch and create an actual STRING dependency edge when sequencing matters.

Advanced config includes Flash Attention, F16/Q8 KV caches, vision-token bounds, and child-only `CUDA_VISIBLE_DEVICES`. For a dedicated secondary GPU, set `cuda_visible_devices` and choose `comfy_vram_handoff = never`; this does not modify ComfyUI's own environment. Q8 KV caches save memory but can change speed or quality slightly.

The backend expects a current llama.cpp build with `--jinja`, `/health`, streaming chat completions, `reasoning_effort`, and multimodal `image_url` support. `/props` enriches identity/capability checks but incomplete metadata is tolerated. Native video additionally requires typed `input_video`; auto mode uses it only when `/props` explicitly advertises video support.

Troubleshooting:

- **llama-server was not found:** set the executable path as described above. A `.gguf` file cannot run by itself.
- **Startup timeout or early exit:** inspect the ComfyUI console; the error includes the bounded tail of llama-server output.
- **CUDA out of memory:** enable `aggressive_vram_handoff`, reduce context size, or reduce GPU layers.
- **Image request rejected:** confirm the model is vision-capable, the projector belongs to the exact model, and the llama.cpp build is recent enough.
- **Native video unavailable:** use `sampled_frames`; native support depends on both the model/projector and the llama.cpp build.

---
