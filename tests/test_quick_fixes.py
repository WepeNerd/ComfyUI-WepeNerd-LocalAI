import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
NAME = "wepenerd_localai_quickfix_test"
SPEC = importlib.util.spec_from_file_location(NAME, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
PACKAGE = importlib.util.module_from_spec(SPEC)
sys.modules[NAME] = PACKAGE
SPEC.loader.exec_module(PACKAGE)
nodes = sys.modules[NAME + ".wn_gguf_nodes"]
models = sys.modules[NAME + ".wn_gguf_models"]


class SeedTests(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(model_path="some-model.gguf", mmproj_path=None,
                                      context_size=8192, validate=Mock())
        self.runner = patch.object(nodes, "_run_payloads", return_value=["ok"]).start()
        self.addCleanup(patch.stopall)

    def test_seed_is_appended_after_existing_widgets_and_defaults_to_previous_behaviour(self):
        for cls in (nodes.WN_PromptEnhancer, nodes.WN_H3PromptEnhancer):
            optional = cls.INPUT_TYPES()["optional"]
            self.assertEqual(list(optional)[-3:], ["seed", "instruction", "sampling"])
            self.assertEqual(optional["seed"][1]["default"], 0)

    def test_prompt_enhancer_sends_seed(self):
        nodes.WN_PromptEnhancer().enhance(self.config, "a cat", "Krea 2", seed=42)
        self.assertEqual(self.runner.call_args.args[1][0]["seed"], 42)
        nodes.WN_PromptEnhancer().enhance(self.config, "a cat", "Krea 2")
        self.assertEqual(self.runner.call_args.args[1][0]["seed"], 0)

    def test_h3_sends_seed(self):
        with patch.object(nodes, "validate_h3_prompt", side_effect=lambda result, *a: result):
            nodes.WN_H3PromptEnhancer().enhance(self.config, "a traveler finds a lighthouse", seed=7)
        self.assertEqual(self.runner.call_args.args[1][0]["seed"], 7)


class ProjectorMatchTests(unittest.TestCase):
    def match(self, files, model):
        root = Path(tempfile.mkdtemp())
        found = {False: {}, True: {}}
        for name in files:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            found[models._is_projector(path.name)][name] = path.resolve()
        with patch.object(models, "_index", side_effect=lambda projectors: found[projectors]):
            result = models.match_projector(str(root / model))
        return result and Path(result).relative_to(root.resolve()).as_posix()

    def test_matches_by_name_across_quantizations(self):
        files = ["Qwen3-VL-8B-Instruct-Q4_K_M.gguf", "mmproj-Qwen3-VL-8B-Instruct-F16.gguf",
                 "gemma-3-12b-it-Q4_K_M.gguf", "mmproj-gemma-3-12b-it-f16.gguf"]
        self.assertEqual(self.match(files, files[0]), files[1])
        self.assertEqual(self.match(files, files[2]), files[3])

    def test_matches_finetune_to_base_projector(self):
        files = ["Huihui-Qwen3-VL-8B-Instruct-abliterated.Q4_K_M.gguf", "mmproj-Qwen3-VL-8B-Instruct-F16.gguf"]
        self.assertEqual(self.match(files, files[0]), files[1])

    def test_prefers_most_specific_name(self):
        files = ["Qwen3-VL-8B-Thinking-Q4.gguf", "mmproj-Qwen3-VL-8B-Instruct-F16.gguf",
                 "mmproj-Qwen3-VL-8B-Thinking-F16.gguf"]
        self.assertEqual(self.match(files, files[0]), files[2])

    def test_generic_projector_in_model_subfolder(self):
        files = ["qwen/model-q4.gguf", "qwen/mmproj-model-f16.gguf", "gemma/model.gguf", "gemma/mmproj-model-f16.gguf"]
        self.assertEqual(self.match(files, files[0]), files[1])

    def test_no_guessing(self):
        self.assertIsNone(self.match(["Llama-3.1-8B-Instruct-Q4_K_M.gguf", "mmproj-Qwen3-VL-8B-Instruct-F16.gguf"],
                                     "Llama-3.1-8B-Instruct-Q4_K_M.gguf"))
        self.assertIsNone(self.match(["a.gguf", "b.gguf", "mmproj-model-f16.gguf"], "a.gguf"))
        self.assertIsNone(self.match(["gemma-3-12b-it.gguf", "mmproj-gemma-3-1b-it.gguf", "mmproj-gemma-3-4b-it.gguf"],
                                     "gemma-3-12b-it.gguf"))

    def test_saved_workflow_values_stay_valid(self):
        choices = nodes.WN_LocalAIModel.INPUT_TYPES()["required"]["projector"][0]
        self.assertEqual(choices[:2], ["Auto / None", "None"])


class HelpTextTests(unittest.TestCase):
    def test_every_node_has_description_and_every_widget_a_tooltip(self):
        for node_id, cls in PACKAGE.NODE_CLASS_MAPPINGS.items():
            self.assertTrue(getattr(cls, "DESCRIPTION", ""), node_id)
            schema = cls.INPUT_TYPES()
            for section in ("required", "optional"):
                for name, spec in schema.get(section, {}).items():
                    with self.subTest(node=node_id, input=name):
                        self.assertGreater(len(spec), 1)
                        self.assertTrue(spec[1].get("tooltip"))

    def test_inline_tooltips_win(self):
        optional = nodes.WN_FolderCaptioner.INPUT_TYPES()["optional"]
        self.assertIn("for Custom, this is the complete skill", optional["instruction"][1]["tooltip"])


if __name__ == "__main__":
    unittest.main()
