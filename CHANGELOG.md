# Changelog

## Unreleased

- Folder Captioner revalidates captions after Sampling cleanup, preventing empty or missing-trigger captions from being written, and honors Sampling's JPEG quality.
- Local AI Sampling defaults reasoning to the selected preset (or the model default for custom); explicit reasoning choices override the preset and the enhancers' built-in Qwen thinking flag.
- New **Local AI Sampling** node: one generation-settings bundle for every simple task node's new `sampling` input. `Prompt Enhancer (Advanced)` and `Image Captioner (Advanced)` are deprecated (hidden from search, still load). `Video Captioner (Advanced)` stays for its video frame controls.
- Prompt Enhancer and H3 Prompt Enhancer gain a second output, `info`, reporting what the node decided (mode, images sent, sampling source, output checks). Output 0 is unchanged.
- Unload Local AI Model gains a `passthrough` input/output so it can sit inline and order later steps.
- Local AI Generate moved to the main Local AI category.
- One skill list per task, shared by simple and advanced nodes. Prompt Enhancer gains Flux, Wan, LTX Video, SDXL and Generic; Image Captioner gains the Krea 2 and General caption skills plus `trigger_word` / `concept_context`; Folder Captioner gains the Image Captioner styles. Advanced nodes show the same friendly names and still accept older internal names.
- `H3` in Prompt Enhancer now runs the H3 Prompt Enhancer node, including its output checks. H3 Prompt Enhancer gains `reference_images`.
- New `instruction` input on the enhancers adds direction to the chosen skill; with Custom it is the whole skill. Custom captioners give a clear error when the instruction is empty.
- Local AI Model: new `memory` setting (unload ComfyUI models first, free only what the LLM needs, or keep the LLM loaded for 5 minutes). The default keeps the previous behaviour.
- Local AI Model: projector `Auto / None` now picks the projector that matches the model by name, or the only generic projector in the model's own folder; added an explicit `None` choice. Saved workflows keep working.
- Prompt Enhancer and H3 Prompt Enhancer: new `seed` widget (default 0 = previous behaviour). Change it for a different version of the prompt.
- Every node now has a description, and every widget has a tooltip.
- Example workflow restored as `example_workflows/prompt-enhancement.json` so ComfyUI lists it among its templates; fixed garbled arrows in the usage guide.
- Add Qwen Image 2.1 generation, edit-preservation, and subject-placement prompt skills to both prompt enhancers, with a separate reference-image input.
- Standalone Local AI package with prompt enhancement and image, video, and folder captioning.
- Preserve current and advanced node IDs, shared server lifecycle, and explicit VRAM controls.
- Include H3 and Krea2 prompt and caption resources.
