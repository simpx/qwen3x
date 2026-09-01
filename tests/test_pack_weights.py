import copy
import struct
import unittest

from scripts import pack_weights


def config_for(model):
    config = {
        "model_type": "qwen3_5_text",
        "max_position_embeddings": pack_weights.MAX_CONTEXT,
        "tie_word_embeddings": True,
        "dtype": "bfloat16",
        "rope_parameters": {
            "partial_rotary_factor": model["rotary_dim"] / model["head_dim"]
        },
    }
    for key in (
        "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
        "full_attention_interval", "num_attention_heads", "num_key_value_heads",
        "head_dim", "linear_num_key_heads", "linear_num_value_heads",
        "linear_key_head_dim", "linear_value_head_dim", "linear_conv_kernel_dim",
    ):
        config[key] = model[key]
    return config


class PackWeightsTest(unittest.TestCase):
    def test_selects_both_supported_models_and_rejects_near_match(self):
        for model in pack_weights.SUPPORTED_MODELS:
            self.assertIs(pack_weights.select_model(config_for(model)), model)
        wrong = copy.deepcopy(config_for(pack_weights.SUPPORTED_MODELS[1]))
        wrong["linear_num_value_heads"] = 16
        with self.assertRaisesRegex(ValueError, "only official"):
            pack_weights.select_model(wrong)

    def test_v2_header_matches_cpp_field_order(self):
        model = pack_weights.SUPPORTED_MODELS[1]
        packed = pack_weights.HEADER.pack(
            pack_weights.MAGIC, pack_weights.FORMAT_VERSION, 0,
            *pack_weights.header_values(model)
        )
        self.assertEqual(len(packed), 80)
        self.assertEqual(struct.unpack_from("<I", packed, 16)[0], 4000)
        self.assertEqual(struct.unpack_from("<I", packed, 24)[0], 2560)
        self.assertEqual(struct.unpack_from("<I", packed, 60)[0], 32)

    def test_4b_schema_has_expected_layer_and_tensor_counts(self):
        model = pack_weights.SUPPORTED_MODELS[1]
        tensors = list(pack_weights.expected_tensors(model))
        self.assertEqual(len(tensors), 426)
        self.assertEqual(
            tensors[0],
            ("model.language_model.embed_tokens.weight", "BF16", (248320, 2560)),
        )
        names = {name for name, _, _ in tensors}
        self.assertIn("model.language_model.layers.31.self_attn.o_proj.weight", names)


if __name__ == "__main__":
    unittest.main()
