# Folder captioning

Connect **Local AI Model → Folder Captioner**, select a vision-capable GGUF and
its matching mmproj, enter an absolute image folder path, and click **Queue**.
One queued run processes the folder; this is not a background folder watcher.

`photo.jpg` produces `photo.txt` in the same folder. Captions contain plain UTF-8
text with no JSON wrapper. Completed captions are saved as the batch progresses;
use the skip option to resume without replacing existing captions.

## Choosing a skill

| Skill | Intended use |
|---|---|
| Character likeness | Describe pose, clothing, setting, and other changeable details around a consistent character |
| Style | Describe image content while leaving the shared target style implicit |
| Refiner | Describe visible distinctions associated with a named concept |

Use `concept_context` to explain the intended concept and resolve ambiguities.
An optional `trigger_word` is written literally; configure any special trigger
substitution in your trainer. `instruction` can override the default emphasis.

Review a sample before training. Check the target, trigger, visible details,
unwanted repetition, and hallucinated attributes. The templates are starting
points, not a guarantee of better training results. Check your trainer's caption
and image-format requirements.

The model is acquired once per batch. Its release or keep-alive settings apply
when the batch ends. Reduce image/context settings or choose a smaller model if
memory is limited. A text-only GGUF cannot caption images without compatible
vision support and a matching projector.
