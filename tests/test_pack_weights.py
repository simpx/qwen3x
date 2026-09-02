import copy
import io
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import pack_weights


def config_for(model):
    config = {
        "model_type": "qwen3_5_text",
        "max_position_embeddings": pack_weights.MAX_CONTEXT,
        "tie_word_embeddings": model["tie_word_embeddings"],
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


def q8_reference(data):
    values = np.frombuffer(data, dtype="<u2").astype(np.uint32)
    values = (values << 16).view(np.float32)
    output = bytearray()
    for block in values.reshape(-1, pack_weights.Q8_BLOCK_SIZE):
        maximum = np.float32(max(abs(float(value)) for value in block))
        scale = np.float32(maximum / np.float32(127.0))
        inverse = np.float32(1.0 / scale) if scale else np.float32(0.0)
        output.extend(struct.pack("<e", float(scale)))
        for value in block:
            normalized = float(np.float32(value * inverse))
            rounded = math.copysign(math.floor(abs(normalized) + 0.5), normalized)
            output.extend(struct.pack("b", max(-127, min(127, int(rounded)))))
    return bytes(output)


def bf16_bytes(values):
    values = np.asarray(values, dtype=np.float32)
    return (values.view(np.uint32) >> 16).astype("<u2").tobytes()


class PackWeightsTest(unittest.TestCase):
    def test_selects_supported_models_and_rejects_near_match(self):
        for model in pack_weights.SUPPORTED_MODELS:
            self.assertIs(pack_weights.select_model(config_for(model)), model)
        qwen9 = config_for(pack_weights.SUPPORTED_MODELS[2])
        del qwen9["tie_word_embeddings"]
        self.assertIs(
            pack_weights.select_model(qwen9, tie_word_embeddings=False),
            pack_weights.SUPPORTED_MODELS[2],
        )
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
            ("model.language_model.embed_tokens.weight", "BF16", (248320, 2560), True),
        )
        names = {name for name, _, _, _ in tensors}
        self.assertIn("model.language_model.layers.31.self_attn.o_proj.weight", names)

    def test_9b_schema_has_independent_lm_head_and_expected_matrix_count(self):
        model = pack_weights.SUPPORTED_MODELS[2]
        tensors = list(pack_weights.expected_tensors(model))
        self.assertEqual(len(tensors), 427)
        self.assertEqual(tensors[0][0], "model.language_model.embed_tokens.weight")
        self.assertEqual(tensors[1][0], "lm_head.weight")
        matrix_parameters = sum(
            math.prod(shape) for _, _, shape, matrix in tensors if matrix
        )
        self.assertEqual(matrix_parameters, 8_952_741_888)

    def test_q8_zero_and_rounding_boundaries_match_scalar_reference(self):
        values = np.zeros(64, dtype=np.float32)
        values[32:] = [127.0, -127.0, 126.5, -126.5, 0.5, -0.5] + [0.0] * 26
        source = bf16_bytes(values)
        packed = pack_weights.quantize_q8_0(source)
        self.assertEqual(packed, q8_reference(source))
        self.assertEqual(packed[:34], bytes(34))
        self.assertEqual(struct.unpack_from("<6b", packed, 36), (127, -127, 127, -127, 1, -1))

    def test_q8_random_bf16_blocks_match_scalar_reference(self):
        random = np.random.default_rng(20260903)
        values = random.normal(size=32 * 17).astype(np.float32)
        source = bf16_bytes(values)
        self.assertEqual(pack_weights.quantize_q8_0(source), q8_reference(source))

    def test_q8_rejects_partial_block_and_non_finite_input(self):
        with self.assertRaisesRegex(ValueError, "complete 32-value blocks"):
            pack_weights.quantize_q8_0(bytes(62))
        values = np.zeros(32, dtype=np.float32)
        values[4] = np.inf
        with self.assertRaisesRegex(ValueError, "non-finite"):
            pack_weights.quantize_q8_0(bf16_bytes(values))

    def test_shard_rejects_wrong_schema_and_truncated_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "part.safetensors")
            header = json.dumps({
                "tensor": {
                    "dtype": "BF16", "shape": [1, 32],
                    "data_offsets": [0, 64],
                }
            }).encode()
            path.write_bytes(struct.pack("<Q", len(header)) + header + bytes(63))
            with pack_weights.SafetensorsShard(path) as shard:
                with self.assertRaisesRegex(ValueError, "expected F32"):
                    shard.copy_tensor("tensor", "F32", (1, 32), False, io.BytesIO())
                with self.assertRaisesRegex(ValueError, "expected BF16"):
                    shard.copy_tensor("tensor", "BF16", (2, 16), False, io.BytesIO())
                with self.assertRaisesRegex(ValueError, "truncated"):
                    shard.copy_tensor("tensor", "BF16", (1, 32), True, io.BytesIO())

    def test_pack_rejects_missing_tensor(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            Path(checkpoint, "config.json").write_text(
                '{"text_config": {}}', encoding="utf-8")
            Path(checkpoint, "model.safetensors.index.json").write_text(
                '{"weight_map": {}}', encoding="utf-8")
            schema = [("required.weight", "BF16", (1, 32), True)]
            with mock.patch.object(pack_weights, "select_model",
                                   return_value=pack_weights.SUPPORTED_MODELS[2]), \
                 mock.patch.object(pack_weights, "expected_tensors",
                                   return_value=iter(schema)):
                with self.assertRaisesRegex(ValueError, "misses text tensors"):
                    pack_weights.pack(checkpoint, Path(directory, "out.bin"))


if __name__ == "__main__":
    unittest.main()
