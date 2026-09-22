import base64
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("wepenerd_localai_test", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
nodes = sys.modules[SPEC.name + ".wn_gguf_nodes"]
qwen = sys.modules[SPEC.name + ".qwen_image_prompt"]


class QwenImagePromptTests(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            model_path="Qwen3.8-27B.gguf", mmproj_path="matching-mmproj.gguf",
            context_size=8192, validate=Mock(),
        )
        self.canvas = np.zeros((1, 64, 96, 3), dtype=np.float32)
        self.source = np.ones((1, 80, 48, 3), dtype=np.float32)
        self.runner = patch.object(nodes, "_run_payloads", return_value=["Enhanced prompt."]).start()
        self.addCleanup(patch.stopall)

    def enhance(self, prompt="Change only the cup to blue.", skill="Qwen Image 2.1 - Edit", **kwargs):
        # The second output is the new info string; these tests check the prompt output.
        return nodes.WN_PromptEnhancer().enhance(self.config, prompt, skill=skill, **kwargs)[:1]

    def payload(self):
        return self.runner.call_args.args[1][0]

    def images(self):
        content = self.payload()["messages"][-1]["content"]
        return [part["image_url"]["url"] for part in content if part["type"] == "image_url"]

    def test_single_edit_uses_edit_guidance_and_shared_sampler(self):
        self.assertEqual(self.enhance(image=self.canvas), ("Enhanced prompt.",))
        payload = self.payload()
        self.assertIn("Change only the requested attributes", payload["messages"][0]["content"])
        self.assertNotIn("video's opening frame", str(payload))
        self.assertNotIn("proposed future events", str(payload))
        self.assertEqual(payload["messages"][-1]["content"][-1]["text"], "Change only the cup to blue.")
        self.assertEqual(payload["messages"][-1]["content"][0]["text"], "Input image:")
        self.assertTrue(self.runner.call_args.kwargs["require_image"])
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(payload["max_tokens"], 2048)

    def test_composition_keeps_different_sizes_and_input_order(self):
        canvas_before, source_before = self.canvas.copy(), self.source.copy()
        self.enhance("Put the object beside the lamp.", "Qwen Image 2.1 - Image 2 into Image 1",
                     image=self.canvas, reference_images=self.source)
        content = self.payload()["messages"][-1]["content"]
        self.assertEqual([part["text"] for part in content[:-1] if part["type"] == "text"], ["<image1>:", "<image2>:"])
        decoded = [Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) for url in self.images()]
        self.assertEqual([image.size for image in decoded], [(96, 64), (48, 80)])
        self.assertEqual([image.getpixel((0, 0)) for image in decoded], [(0, 0, 0), (255, 255, 255)])
        np.testing.assert_array_equal(self.canvas, canvas_before)
        np.testing.assert_array_equal(self.source, source_before)
        self.assertIn("<image1> is the destination canvas", self.payload()["messages"][0]["content"])

    def test_all_batch_frames_precede_reference_images_without_truncation(self):
        self.enhance(image=np.repeat(self.canvas, 10, axis=0), reference_images=self.source)
        self.assertEqual(len(self.images()), 11)
        self.assertIn("Attached image count: 11", self.payload()["messages"][0]["content"])

    def test_rgba_reference_is_encoded_as_png_with_alpha(self):
        rgba = np.zeros((1, 64, 64, 4), dtype=np.float32)
        rgba[0, 16:48, 16:48] = 1
        self.enhance("Extract the subject.", image=rgba)
        url = self.images()[0]
        self.assertTrue(url.startswith("data:image/png;base64,"))
        decoded = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
        self.assertEqual(decoded.mode, "RGBA")
        self.assertEqual(decoded.getpixel((0, 0))[3], 0)

    def test_text_only_edit_does_not_require_projector_or_claim_to_see_images(self):
        self.config.mmproj_path = None
        self.enhance("Move the object from image 2 into image 1.")
        self.assertFalse(self.runner.call_args.kwargs["require_image"])
        self.assertIsInstance(self.payload()["messages"][-1]["content"], str)
        self.assertIn("Attached image count: 0", self.payload()["messages"][0]["content"])
        self.assertIn("without inventing its appearance", self.payload()["messages"][0]["content"])

    def test_generation_has_no_edit_or_video_instructions(self):
        self.enhance('A paper sculpture with the word "日光".', "Qwen Image 2.1 - Generate")
        system = self.payload()["messages"][0]["content"]
        self.assertIn("finished still", system)
        self.assertNotIn("actionable edit instruction", system)
        self.assertEqual(self.payload()["messages"][-1]["content"], 'A paper sculpture with the word "日光".')

    def test_image_only_generation_stays_a_still_image(self):
        self.enhance("", "Qwen Image 2.1 - Generate", image=self.canvas)
        self.assertIn("still-image generation prompt", self.payload()["messages"][-1]["content"][-1]["text"])

    def test_blank_edit_does_not_invent_an_operation(self):
        with self.assertRaisesRegex(ValueError, "Prompt cannot be empty"):
            self.enhance("", image=self.canvas)
        self.runner.assert_not_called()

    def test_blank_composition_uses_the_selected_task(self):
        self.enhance("", "Qwen Image 2.1 - Image 2 into Image 1", image=self.canvas, reference_images=self.source)
        self.assertIn("main subject from image 2 into image 1", self.payload()["messages"][-1]["content"][-1]["text"])

    def test_missing_projector_fails_before_inference_including_reference_only(self):
        self.config.mmproj_path = None
        for kwargs in ({"image": self.canvas}, {"reference_images": self.source}):
            with self.subTest(kwargs=list(kwargs)), self.assertRaisesRegex(ValueError, "compatible mmproj"):
                self.enhance(**kwargs)
        self.runner.assert_not_called()

    def test_advanced_node_retains_explicit_sampling_and_unwraps_json(self):
        self.runner.return_value = [json.dumps({"rewritten_prompt": 'Keep the sign reading "OPEN".', "ratio_follow": "<image1>"})]
        result = nodes.WN_GGUFPromptEnhance().enhance(
            self.config, "Keep the sign.", 1000, 0.35, 123,
            prompt_style="qwen_image_2_1_edit", reasoning_effort="low",
            image=self.canvas, reference_images=self.source,
        )
        self.assertEqual(result, ('Keep the sign reading "OPEN".',))
        self.assertEqual(self.payload()["max_tokens"], 1000)
        self.assertEqual(self.payload()["temperature"], 0.35)
        self.assertEqual(self.payload()["seed"], 123)
        self.assertEqual(self.payload()["reasoning_effort"], "low")
        self.assertEqual(len(self.images()), 2)

    def test_system_override_replaces_bundled_guidance(self):
        self.enhance(system_prompt_override="Use restrained technical wording.")
        system = self.payload()["messages"][0]["content"]
        self.assertTrue(system.startswith("Use restrained technical wording."))
        self.assertNotIn("# Qwen Image", system)

    def test_h3_skill_uses_the_h3_node_path_with_image_role(self):
        self.runner.return_value = ["  Existing H3 result.  "]
        with patch.object(nodes, "validate_h3_prompt", side_effect=lambda result, *args: result) as check:
            self.assertEqual(self.enhance("", "H3", image=self.canvas, image_role="First frame"), ("  Existing H3 result.  ",))
        self.assertEqual(check.call_args.args[2], "I2V")  # Auto mode follows the image role, as on the H3 node
        content = self.payload()["messages"][-1]["content"]
        self.assertIn("short video scenario", content[0]["text"])
        self.assertIn("video's opening frame", content[1]["text"])
        self.assertTrue(self.images()[0].startswith("data:image/jpeg;base64,"))

    def test_existing_skills_consume_reference_input(self):
        self.enhance("Use both references.", "Custom", system_prompt_override="Return a prompt.",
                     image=self.canvas, reference_images=self.source, image_role="Reference image")
        self.assertEqual(len(self.images()), 2)

    def test_node_contract_and_registration_are_backward_compatible(self):
        simple = nodes.WN_PromptEnhancer.INPUT_TYPES()
        advanced = nodes.WN_GGUFPromptEnhance.INPUT_TYPES()
        self.assertEqual(list(simple["required"]), ["model", "prompt", "skill"])
        self.assertEqual(list(advanced["required"]), ["config", "prompt", "max_tokens", "temperature", "seed", "prompt_style", "reasoning_effort"])
        for schema in (simple, advanced):
            self.assertEqual(list(schema["optional"])[:3], ["system_prompt_override", "image", "image_role"])
            self.assertEqual(schema["optional"]["reference_images"][0], "IMAGE")
        # Saved workflows store the skill name, so every earlier name must still be offered.
        self.assertEqual(simple["required"]["skill"][0][0], "H3")
        self.assertLessEqual({"H3", "Krea 2", "Custom", *qwen.QWEN_IMAGE_SKILLS}, set(simple["required"]["skill"][0]))
        # Output 0 is unchanged, so saved links keep working; info was added after it.
        self.assertEqual(nodes.WN_PromptEnhancer.RETURN_TYPES, ("STRING", "STRING"))
        self.assertEqual(nodes.WN_PromptEnhancer.RETURN_NAMES[0], "enhanced_prompt")
        ownership = json.loads((ROOT / "node-ownership.json").read_text())
        self.assertEqual(set(PACKAGE.NODE_CLASS_MAPPINGS), set(ownership["nodes"]))

    def test_output_cleaning_preserves_literals_and_accepts_official_envelope(self):
        expected = 'Change the sign to "你好  world!". Keep everything else unchanged.'
        for value in (expected, json.dumps({"rewritten_prompt": expected, "wh_ratio": "16:9"}),
                      "```json\n" + json.dumps({"rewritten_prompt": expected}) + "\n```"):
            with self.subTest(value=value):
                self.assertEqual(qwen.clean_qwen_image_prompt(value), expected)
        self.assertEqual(qwen.clean_qwen_image_prompt('A symbol {x} on a sign.'), 'A symbol {x} on a sign.')


if __name__ == "__main__":
    unittest.main()
