import base64
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
NAME = "wepenerd_localai_shared_skills_test"
SPEC = importlib.util.spec_from_file_location(NAME, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[NAME] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
nodes = sys.modules[NAME + ".wn_gguf_nodes"]


def config(**extra):
    return SimpleNamespace(model_path="some-model.gguf", mmproj_path="some-mmproj.gguf",
                           context_size=8192, validate=Mock(), **extra)


class Base(unittest.TestCase):
    def setUp(self):
        self.runner = patch.object(nodes, "_run_payloads", return_value=["A result."]).start()
        self.addCleanup(patch.stopall)
        self.image = np.zeros((1, 32, 32, 3), dtype=np.float32)

    def payload(self, index=0):
        return self.runner.call_args.args[1][index]

    def system(self):
        return self.payload()["messages"][0]["content"]


class SharedListTests(Base):
    def test_simple_and_advanced_offer_the_same_lists(self):
        pairs = [
            (nodes.WN_PromptEnhancer, "skill", nodes.WN_GGUFPromptEnhance, "prompt_style"),
            (nodes.WN_ImageCaptioner, "style", nodes.WN_GGUFCaptionImage, "caption_style"),
            (nodes.WN_VideoCaptioner, "style", nodes.WN_GGUFCaptionVideo, "caption_style"),
        ]
        for simple, s_name, advanced, a_name in pairs:
            with self.subTest(simple=simple.__name__):
                self.assertEqual(simple.INPUT_TYPES()["required"][s_name][0],
                                 advanced.INPUT_TYPES()["required"][a_name][0])
        self.assertEqual(nodes.WN_FolderCaptioner.INPUT_TYPES()["required"]["skill"][0],
                         list(nodes.IMAGE_CAPTION_SKILLS))

    def test_advanced_nodes_still_accept_old_internal_names(self):
        for value in ("generic", "minimax_h3", "flux", "qwen_image_2_1_edit", "custom", "Generic", "Flux"):
            self.assertIs(nodes.WN_GGUFPromptEnhance.VALIDATE_INPUTS(value), True)
        for value in ("dataset_natural", "booru_tags", "motion_camera", "custom", "Tags", "Krea 2 - Style"):
            self.assertIs(nodes.WN_GGUFCaptionImage.VALIDATE_INPUTS(value), True)
        for value in ("motion_camera", "Motion + Camera"):
            self.assertIs(nodes.WN_GGUFCaptionVideo.VALIDATE_INPUTS(value), True)
        self.assertIsInstance(nodes.WN_GGUFPromptEnhance.VALIDATE_INPUTS("nope"), str)

    def test_legacy_and_display_names_produce_the_same_request(self):
        nodes.WN_GGUFPromptEnhance().enhance(config(), "a car", 512, 0.7, 0, prompt_style="flux")
        legacy = self.system()
        nodes.WN_GGUFPromptEnhance().enhance(config(), "a car", 512, 0.7, 0, prompt_style="Flux")
        self.assertEqual(self.system(), legacy)
        self.assertEqual(legacy, nodes.PROMPT_STYLES["flux"])

    def test_simple_enhancer_now_offers_flux(self):
        nodes.WN_PromptEnhancer().enhance(config(), "a car", "Flux")
        self.assertEqual(self.system(), nodes.PROMPT_STYLES["flux"])


class InstructionTests(Base):
    def test_instruction_adds_to_the_skill(self):
        nodes.WN_PromptEnhancer().enhance(config(), "a car", "Flux", instruction="Keep it under 40 words.")
        self.assertTrue(self.system().startswith(nodes.PROMPT_STYLES["flux"]))
        self.assertTrue(self.system().endswith("Keep it under 40 words."))

    def test_custom_uses_instruction_or_override(self):
        nodes.WN_PromptEnhancer().enhance(config(), "a car", "Custom", instruction="Write haiku prompts.")
        self.assertEqual(self.system(), "Write haiku prompts.")
        nodes.WN_PromptEnhancer().enhance(config(), "a car", "Custom", system_prompt_override="Override.")
        self.assertEqual(self.system(), "Override.")
        with self.assertRaisesRegex(ValueError, "Custom needs your instructions"):
            nodes.WN_PromptEnhancer().enhance(config(), "a car", "Custom")

    def test_h3_node_instruction(self):
        with patch.object(nodes, "validate_h3_prompt", side_effect=lambda result, *a: result):
            nodes.WN_H3PromptEnhancer().enhance(config(), "a lighthouse", instruction="No camera cuts.")
        self.assertTrue(self.system().endswith("No camera cuts."))


class H3DelegationTests(Base):
    def test_prompt_enhancer_h3_matches_h3_node_and_passes_references(self):
        with patch.object(nodes, "validate_h3_prompt", side_effect=lambda result, *a: result) as check:
            nodes.WN_PromptEnhancer().enhance(config(), "use both", "H3", reference_images=self.image,
                                              image_role="Reference image", seed=5)
            via_enhancer = self.payload()
            nodes.WN_H3PromptEnhancer().enhance(config(), "use both", reference_images=self.image,
                                                image_role="Reference image", seed=5)
            via_node = self.payload()
        self.assertEqual(via_enhancer, via_node)
        self.assertEqual(check.call_count, 2)
        self.assertEqual(check.call_args.args[2], "Ref2V")
        self.assertEqual(check.call_args.args[5], 1)  # one reference image counted for alias checks

    def test_override_keeps_the_raw_path(self):
        with patch.object(nodes, "validate_h3_prompt") as check:
            nodes.WN_PromptEnhancer().enhance(config(), "a lighthouse", "H3", system_prompt_override="Mine.")
        check.assert_not_called()
        self.assertEqual(self.system(), "Mine.")


class CaptionTests(Base):
    def test_existing_image_styles_are_unchanged(self):
        result = nodes.WN_ImageCaptioner().caption(config(), self.image, "Dataset")
        self.assertEqual(self.system(), nodes.IMAGE_CAPTION_STYLES["dataset_natural"])
        self.assertEqual(self.payload()["messages"][-1]["content"][0]["text"], nodes.DEFAULT_IMAGE_INSTRUCTION)
        self.assertEqual(result, (["A result."],))

    def test_image_captioner_can_preview_a_krea_skill(self):
        self.runner.return_value = ["a woman in a red coat"]
        result = nodes.WN_ImageCaptioner().caption(config(), self.image, "Krea 2 - Character likeness",
                                                   trigger_word="ohwx", concept_context="Anna")
        system, user = nodes.caption_instructions("Krea 2 - Character likeness", "ohwx", "Anna", "")
        self.assertEqual(self.system(), system)
        self.assertEqual(self.payload()["messages"][-1]["content"][0]["text"], user)
        self.assertEqual(result, (["ohwx, a woman in a red coat"],))

    def test_folder_captioner_accepts_image_styles(self):
        folder = Path(tempfile.mkdtemp())
        from PIL import Image
        Image.new("RGB", (16, 16)).save(folder / "one.png")
        def fake_results(cfg, payloads, total, **kwargs):
            for _payload in payloads:
                yield "ohwx, red, coat"

        with patch.object(nodes, "_iter_payloads", side_effect=fake_results):
            report, written, skipped = nodes.WN_FolderCaptioner().caption(
                config(), str(folder), "Tags", trigger_word="ohwx")
        self.assertEqual((written, skipped), (1, 0))
        self.assertEqual((folder / "one.txt").read_text(encoding="utf-8"), "ohwx, red, coat\n")

    def test_old_folder_values_still_work(self):
        for value in ("General caption", "Krea 2 - Style", "Custom"):
            system, user, _ = nodes._image_caption_instructions(value, "Only nouns.", "", "")
            self.assertTrue(system)

    def test_custom_caption_needs_instruction(self):
        with self.assertRaisesRegex(ValueError, "Custom needs your instruction"):
            nodes.WN_ImageCaptioner().caption(config(), self.image, "Custom")
        with self.assertRaisesRegex(ValueError, "Custom needs your instruction"):
            nodes.WN_VideoCaptioner().caption(config(), object(), "Custom")


class MemoryModeTests(unittest.TestCase):
    def build(self, memory=None):
        model_file = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
        model_file.write(b"\0" * (4 * 1024 * 1024))
        model_file.close()
        with patch.object(nodes, "resolve_choice", return_value=model_file.name), \
             patch.object(nodes, "match_projector", return_value=None), \
             patch.object(nodes.WNGGUFConfig, "validate"):
            args = () if memory is None else (memory,)
            return nodes.WN_LocalAIModel().build("LLM/x.gguf", "Auto / None", *args)[0]

    def test_default_keeps_previous_behaviour(self):
        cfg = self.build()
        self.assertEqual((cfg.target_free_vram_mb, cfg.release_after_generate, cfg.keep_alive_seconds), (24576, True, 0))

    def test_free_only_needed(self):
        cfg = self.build("Free only what the LLM needs")
        self.assertEqual(cfg.target_free_vram_mb, 2048)  # tiny test model: floor
        self.assertTrue(cfg.release_after_generate)

    def test_keep_loaded(self):
        cfg = self.build("Keep LLM loaded for 5 min")
        self.assertEqual((cfg.release_after_generate, cfg.keep_alive_seconds), (False, 300))

    def test_estimate_scales_with_model_size(self):
        with patch.object(nodes.os.path, "getsize", side_effect=lambda p: 10 * 1024**3 if p == "m" else 1024**3):
            self.assertEqual(nodes._estimated_vram_mb("m", "p"), 14336)


class StructureTests(Base):
    def sampling(self, **overrides):
        values = dict(preset="custom", max_tokens=300, temperature=0.9, top_p=0.5, top_k=7, min_p=0.1,
                      repetition_penalty=1.2, presence_penalty=0.3, frequency_penalty=0.4, seed=99,
                      reasoning_effort="low", caption_prefix="PFX", banned_phrases="red")
        values.update(overrides)
        return nodes.WN_LocalAISampling().build(**values)[0]

    def test_sampling_overrides_prompt_enhancer(self):
        text, info = nodes.WN_PromptEnhancer().enhance(config(), "a car", "Flux", sampling=self.sampling())
        payload = self.payload()
        self.assertEqual((payload["max_tokens"], payload["temperature"], payload["top_p"], payload["top_k"],
                          payload["repeat_penalty"], payload["seed"], payload["reasoning_effort"]),
                         (300, 0.9, 0.5, 7, 1.2, 99, "low"))
        self.assertIn("sampling=Local AI Sampling", info)
        self.assertIn("output_checks=none", info)

    def test_qwen_preset(self):
        bundle = self.sampling(preset="qwen_thinking", reasoning_effort="default")
        self.assertEqual((bundle["temperature"], bundle["top_p"], bundle["top_k"], bundle["reasoning_effort"]),
                         (0.6, 0.95, 20, "high"))

    def test_sampling_defaults_follow_preset_and_respect_explicit_effort(self):
        defaults = {}
        for section in nodes.WN_LocalAISampling.INPUT_TYPES().values():
            for name, spec in section.items():
                defaults[name] = spec[1].get("default", spec[0][0] if isinstance(spec[0], list) else None)
        for preset, expected in (("qwen_thinking", "high"), ("qwen_non_thinking", "none"), ("custom", "default")):
            with self.subTest(preset=preset):
                values = {**defaults, "preset": preset}
                self.assertEqual(nodes.WN_LocalAISampling().build(**values)[0]["reasoning_effort"], expected)
                values.pop("reasoning_effort")
                self.assertEqual(nodes.WN_LocalAISampling().build(**values)[0]["reasoning_effort"], expected)
        self.assertEqual(self.sampling(preset="qwen_thinking", reasoning_effort="none")["reasoning_effort"], "none")
        self.assertEqual(self.sampling(preset="qwen_non_thinking", reasoning_effort="low")["reasoning_effort"], "low")

    def test_qwen_enhancers_honor_sampling_reasoning(self):
        cfg = config()
        cfg.model_path = "Huihui-Qwen3.8-27B-abliterated-Q4_K.gguf"
        with patch.object(nodes, "validate_h3_prompt", side_effect=lambda result, *a: result):
            for skill in ("Flux", "H3"):
                for effort in ("none", "low", "medium", "high", "default"):
                    with self.subTest(skill=skill, effort=effort):
                        nodes.WN_PromptEnhancer().enhance(cfg, "a car", skill,
                                                        sampling=self.sampling(reasoning_effort=effort))
                        payload = self.payload()
                        if effort == "default":
                            self.assertNotIn("reasoning_effort", payload)
                            self.assertNotIn("enable_thinking", payload.get("chat_template_kwargs", {}))
                        else:
                            self.assertEqual(payload["reasoning_effort"], effort)
                            self.assertEqual(payload["chat_template_kwargs"]["enable_thinking"], effort != "none")
                nodes.WN_PromptEnhancer().enhance(cfg, "a car", skill)
                self.assertEqual(self.payload()["reasoning_effort"], "none")
                self.assertEqual(self.payload()["chat_template_kwargs"], {"enable_thinking": False})

    def test_sampling_preserves_other_template_options(self):
        payload = {"chat_template_kwargs": {"enable_thinking": False, "custom_option": "keep"}}
        result = nodes._apply_sampling(payload, self.sampling(reasoning_effort="default"))
        self.assertEqual(result["chat_template_kwargs"], {"custom_option": "keep"})
        result = nodes._apply_sampling({}, self.sampling(reasoning_effort="high"))
        self.assertNotIn("chat_template_kwargs", result)

    def test_sampling_reaches_captioners_and_cleanup(self):
        self.runner.return_value = ["a red coat"]
        result = nodes.WN_ImageCaptioner().caption(config(), self.image, "Dataset", sampling=self.sampling())
        self.assertEqual(self.payload()["max_tokens"], 300)
        self.assertEqual(result, (["PFX a coat"],))

    def test_h3_info_reports_automatic_choices(self):
        with patch.object(nodes, "validate_h3_prompt", side_effect=lambda result, *a: result):
            text, info = nodes.WN_H3PromptEnhancer().enhance(config(), "", image=self.image, image_role="First frame")
        self.assertIn("mode=I2V", info)
        self.assertIn("Preserve -> Develop scenario", info)
        with patch.object(nodes, "validate_h3_prompt", side_effect=lambda result, *a: result):
            text, info = nodes.WN_PromptEnhancer().enhance(config(), "a lighthouse", "H3")
        self.assertTrue(info.startswith("skill=H3 (via H3 Prompt Enhancer)"))

    def test_unload_passes_value_through(self):
        with patch.object(nodes.SERVER_MANAGER, "stop", return_value=None):
            status, value = nodes.WN_GGUFLLMRelease().release(passthrough={"latent": 1})
        self.assertEqual(value, {"latent": 1})
        self.assertIn("No managed", status)

    def test_old_advanced_nodes_are_deprecated_not_removed(self):
        for node_id in ("WN_GGUFPromptEnhance", "WN_GGUFCaptionImage"):
            self.assertTrue(PACKAGE.NODE_CLASS_MAPPINGS[node_id].DEPRECATED)
        for node_id in ("WN_GGUFCaptionVideo", "WN_GGUFLLMConfig", "WN_GGUFLLMGenerate"):
            self.assertFalse(getattr(PACKAGE.NODE_CLASS_MAPPINGS[node_id], "DEPRECATED", False))
        self.assertEqual(nodes.WN_GGUFLLMGenerate.CATEGORY, "WepeNerd/Local AI")


class FolderSamplingTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.sampling = nodes.WN_LocalAISampling().build("custom", 768, 0.2, 0.8, 20, 0, 1, 0, 0, 0)[0]
        self.payloads = []

    def run_folder(self, replies, sampling=None, **kwargs):
        def results(cfg, payloads, total, **options):
            for payload, reply in zip(payloads, replies):
                self.payloads.append(payload)
                yield reply
        with patch.object(nodes, "_iter_payloads", side_effect=results):
            return nodes.WN_FolderCaptioner().caption(config(), str(self.folder), "Dataset",
                                                     sampling=sampling, **kwargs)

    def test_empty_cleanup_preserves_completed_files_and_can_resume(self):
        for name in ("a", "b"):
            Image.new("RGB", (16, 16)).save(self.folder / f"{name}.png")
        sampling = {**self.sampling, "banned_phrases": "a red coat"}
        with self.assertRaisesRegex(RuntimeError, "no visible completion content"):
            self.run_folder(["a blue coat", "a red coat"], sampling)
        self.assertEqual((self.folder / "a.txt").read_text(), "a blue coat\n")
        self.assertFalse((self.folder / "b.txt").exists())
        _, written, skipped = self.run_folder(["a green coat"], sampling)
        self.assertEqual((written, skipped), (1, 1))
        self.assertEqual((self.folder / "a.txt").read_text(), "a blue coat\n")
        self.assertEqual((self.folder / "b.txt").read_text(), "a green coat\n")

    def test_cleanup_cannot_remove_trigger_or_overwrite_existing_caption(self):
        Image.new("RGB", (16, 16)).save(self.folder / "one.png")
        caption = self.folder / "one.txt"
        caption.write_text("ohwx, original caption\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "omitted or changed trigger_word"):
            self.run_folder(["ohwx, a red coat"], {**self.sampling, "banned_phrases": "ohwx"},
                            trigger_word="ohwx", existing_captions="Overwrite")
        self.assertEqual(caption.read_text(), "ohwx, original caption\n")

    def test_valid_cleanup_and_prefix_are_saved(self):
        Image.new("RGB", (16, 16)).save(self.folder / "one.png")
        self.run_folder(["ohwx, a red coat"],
                        {**self.sampling, "banned_phrases": "red", "caption_prefix": "Photo:"},
                        trigger_word="ohwx")
        self.assertEqual((self.folder / "one.txt").read_text(), "Photo: ohwx, a coat\n")

    def test_folder_jpeg_quality_changes_encoded_images_and_defaults_to_90(self):
        pixels = np.random.default_rng(42).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(self.folder / "one.png")
        tables = []
        for sampling in (None, self.sampling, {**self.sampling, "jpeg_quality": 12}):
            self.run_folder(["a coat"], sampling, existing_captions="Overwrite")
            content = self.payloads[-1]["messages"][-1]["content"]
            url = next(part["image_url"]["url"] for part in content if part["type"] == "image_url")
            with Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as encoded:
                tables.append(encoded.quantization)
        self.assertEqual(tables[0], tables[1])
        self.assertNotEqual(tables[1], tables[2])


if __name__ == "__main__":
    unittest.main()
