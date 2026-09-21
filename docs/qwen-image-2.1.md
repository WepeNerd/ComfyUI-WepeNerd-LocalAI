# Qwen Image 2.1 prompt enhancement

Research checked against the official Qwen and ComfyUI sources on 2026-09-21.
These presets use your LocalAI GGUF language model to write prompts **for** Qwen
Image 2.1. Image generation/editing still runs in your downstream Qwen workflow.

## Use in LocalAI

Restart ComfyUI and refresh the browser. Connect **Local AI Model → Prompt
Enhancer**, then choose a skill:

| Skill | Use |
|---|---|
| Qwen Image 2.1 - Generate | Describe a new still image from your brief |
| Qwen Image 2.1 - Edit | Make requested changes while preserving untargeted content |
| Qwen Image 2.1 - Image 2 into Image 1 | Insert a source subject into a destination canvas |

The advanced enhancer exposes the same choices as `qwen_image_2_1`,
`qwen_image_2_1_edit`, and `qwen_image_2_1_compose` in `prompt_style`.

For a single edit, connect the original to `image`. For composition, connect
the destination to `image` and the source to `reference_images`. Their dimensions
can differ. Both sockets accept batches: numbering runs through every image in
`image`, then every image in `reference_images`. To make the source image 2,
send exactly one canvas image to `image`.

Connect those same originals to the Qwen editing workflow in the same order,
and connect `enhanced_prompt` to its prompt input. The enhancer does not forward
images or change downstream resolution. Its `image_role` control belongs to the
other skills; Qwen uses the selected task and the roles you describe.

Image inspection requires a vision-capable GGUF and its matching mmproj in Local
AI Model. The enhancer sends aspect-preserving PNG previews, at most 1024 pixels
on the longest edge. Text-only editing instructions also work without connected
images, but the LLM then cannot inspect or disambiguate their contents.

`Edit` needs an instruction. The placement preset can run with a blank prompt,
which requests moving image 2's main subject into image 1. `Generate` with an
image and a blank prompt produces a still-image description.

## Prompting findings

Qwen's generation guidance favors a coherent description of the finished image:
medium, subject, spatial arrangement, materials and lighting. Keep authored
counts and relationships; quote exact visible wording in its original script.
Avoid vague quality slogans. The official rewriter expands aggressively;
LocalAI deliberately keeps expansion proportional to the request instead of
enforcing a long word count. [Official generation instructions](https://github.com/QwenLM/Qwen-Image-2.1/blob/main/prompt_rewrite/prompts/system_prompt_t2i.txt).

For editing, identify the change and preserve everything outside its scope.
Do not recaption unchanged details. For multiple images, use `<image1>`,
`<image2>`, etc., and identify the destination and what each reference supplies.
The official rewriter uses ordinary wording for a lone image. Its preservation
rules retain identity without weakening the requested edit. [Official editing instructions](https://github.com/QwenLM/Qwen-Image-2.1/blob/main/prompt_rewrite/prompts/system_prompt_edit.txt).

Qwen supports up to ten references and native RGBA output, including subject
extraction. Specify a transparent background and alpha channel when transparency
is intended. [Official model repository](https://github.com/QwenLM/Qwen-Image-2.1#transparent-image-generation-rgba).

In the official ComfyUI edit template, image 1 is the edit target. Keep the output
size close to the resized target to reduce shifts. Its standard path uses CFG 1,
where a negative prompt has no effect, so preservation belongs in the positive
edit instruction. [Official ComfyUI edit template](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_image_edit.json).

## Example requests

These are practical starter requests, not measured guarantees of model behavior.

| Task | Request to the enhancer |
|---|---|
| One attribute | Change only the jacket color to navy blue. Keep its cut and texture, the person, pose, background and framing unchanged. |
| Insert a subject | Place the dog from image 2 on the empty rug in image 1. Keep the room and existing people unchanged. |
| Replace an object | Replace the vase on the table with the vase from image 2. Keep the table and the rest of image 1 unchanged. |
| Clothing | Put the coat from image 2 on the person in image 1. Preserve their face, hair, pose and other clothing. |
| Remove | Remove only the bag on the floor and continue the surrounding floor texture through that area. |
| Background | Replace the background with a plain gray studio backdrop. Keep the person and their clothing unchanged. |
| Text | Replace the sign's wording with "OPEN LATE". Preserve its position, type style and all other text. |
| Style | Apply image 2's watercolor rendering to image 1 while preserving image 1's subjects and composition. |
| Outpaint | Extend the scene to the right. Keep the existing image interior unchanged. |
| Cutout | Extract only the bicycle as an RGBA image with a transparent background, preserving its geometry and colors. |

Preservation in a prompt cannot guarantee identical pixels outside the edit.
When exact preservation matters, composite the edited region over the original
with a mask in the downstream workflow. Supply region guides with an explicit
role; a mask reference is guidance, not a guarantee of exact preservation.

## Integration boundaries

The bundled instructions are a compact LocalAI adaptation. They return a single
STRING, not Qwen's official JSON size contract. If a model returns the official
`rewritten_prompt` envelope anyway, the node extracts the prompt; `wh_ratio` and
`ratio_follow` do not resize the downstream canvas. Set dimensions yourself.

Qwen also publishes separate [PE-T2I](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I)
and [PE-I2I](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I) fine-tuned Qwen3.5-VL
9B rewriters. This change does not install those weights or add their Transformers
backend. It retains LocalAI's existing llama.cpp model, sampling and lifecycle.
Compatibility with a converted PE checkpoint depends on the GGUF, matching
projector and llama.cpp build; those checkpoints were not tested here.

Validation: 16 automated regression tests plus live text-generation, single-edit,
and two-image composition rewrites using the installed Huihui Qwen3.8 27B model
and matching projector. The visual checks used simple geometric test images.
Qwen Image 2.1 diffusion output and the refreshed ComfyUI browser UI were not tested.
