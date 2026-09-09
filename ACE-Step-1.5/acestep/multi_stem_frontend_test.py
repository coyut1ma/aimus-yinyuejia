import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from acestep.models.common.configuration_acestep_v15 import AceStepConfig
from acestep.models.turbo.modeling_acestep_v15_turbo import (
    AceStepConditionGenerationModel,
    AceStepDiTModel,
)


def _tiny_config() -> AceStepConfig:
    config = AceStepConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        use_sliding_window=False,
        layer_types=["full_attention", "full_attention"],
        audio_acoustic_hidden_dim=6,
        in_channels=18,
        patch_size=2,
        attention_dropout=0.0,
        num_lyric_encoder_hidden_layers=1,
        num_timbre_encoder_hidden_layers=1,
        num_attention_pooler_hidden_layers=1,
        num_audio_decoder_hidden_layers=1,
        fsq_dim=32,
        fsq_input_levels=[4, 4],
        text_hidden_dim=16,
        timbre_hidden_dim=6,
    )
    config._attn_implementation = "eager"
    return config


def _decoder_inputs():
    batch_size, seq_len, audio_dim = 2, 8, 6
    context_dim = 12
    hidden_states = torch.randn(batch_size, seq_len, audio_dim)
    context_latents = torch.randn(batch_size, seq_len, context_dim)
    encoder_hidden_states = torch.randn(batch_size, 5, 32)
    encoder_attention_mask = torch.ones(batch_size, 5)
    attention_mask = torch.ones(batch_size, seq_len)
    timestep = torch.full((batch_size,), 0.7)
    multi_context = torch.randn(batch_size, 4, seq_len, context_dim)
    multi_hidden = torch.randn(batch_size, 4, seq_len, audio_dim)
    target_stem_id = 2
    multi_context[:, target_stem_id] = context_latents
    multi_hidden[:, target_stem_id] = hidden_states
    return {
        "hidden_states": hidden_states,
        "timestep": timestep,
        "timestep_r": timestep,
        "attention_mask": attention_mask,
        "encoder_hidden_states": encoder_hidden_states,
        "encoder_attention_mask": encoder_attention_mask,
        "context_latents": context_latents,
        "target_stem_id": target_stem_id,
        "multi_stem_context_latents": multi_context,
        "multi_stem_hidden_states": multi_hidden,
    }


class MultiStemFrontendTests(unittest.TestCase):
    def test_zero_initialized_joint_frontend_matches_single_path(self):
        torch.manual_seed(0)
        model = AceStepDiTModel(_tiny_config()).eval()
        inputs = _decoder_inputs()

        with torch.no_grad():
            single = model(
                inputs["hidden_states"],
                inputs["timestep"],
                inputs["timestep_r"],
                inputs["attention_mask"],
                inputs["encoder_hidden_states"],
                inputs["encoder_attention_mask"],
                inputs["context_latents"],
                use_cache=False,
            )[0]
            joint = model(
                inputs["hidden_states"],
                inputs["timestep"],
                inputs["timestep_r"],
                inputs["attention_mask"],
                inputs["encoder_hidden_states"],
                inputs["encoder_attention_mask"],
                inputs["context_latents"],
                use_cache=False,
                joint_frontend=True,
                target_stem_id=inputs["target_stem_id"],
                multi_stem_hidden_states=inputs["multi_stem_hidden_states"],
                multi_stem_context_latents=inputs["multi_stem_context_latents"],
            )[0]

        self.assertEqual(single.shape, joint.shape)
        self.assertTrue(torch.allclose(single, joint, atol=1e-5, rtol=1e-5))

    def test_joint_frontend_accepts_sampler_cache(self):
        torch.manual_seed(0)
        model = AceStepDiTModel(_tiny_config()).eval()
        inputs = _decoder_inputs()

        with torch.no_grad():
            first = model(
                inputs["hidden_states"],
                inputs["timestep"],
                inputs["timestep_r"],
                inputs["attention_mask"],
                inputs["encoder_hidden_states"],
                inputs["encoder_attention_mask"],
                inputs["context_latents"],
                use_cache=True,
                joint_frontend=True,
                target_stem_id=inputs["target_stem_id"],
                multi_stem_hidden_states=inputs["multi_stem_hidden_states"],
                multi_stem_context_latents=inputs["multi_stem_context_latents"],
            )
            second = model(
                inputs["hidden_states"],
                inputs["timestep"],
                inputs["timestep_r"],
                inputs["attention_mask"],
                inputs["encoder_hidden_states"],
                inputs["encoder_attention_mask"],
                inputs["context_latents"],
                use_cache=True,
                past_key_values=first[1],
                joint_frontend=True,
                target_stem_id=inputs["target_stem_id"],
                multi_stem_hidden_states=inputs["multi_stem_hidden_states"],
                multi_stem_context_latents=inputs["multi_stem_context_latents"],
            )

        self.assertEqual(first[0].shape, inputs["hidden_states"].shape)
        self.assertEqual(second[0].shape, inputs["hidden_states"].shape)

    def test_multi_stem_adapter_training_freezes_base_model(self):
        model = AceStepConditionGenerationModel(_tiny_config())

        trainable_names = model.enable_multi_stem_adapter_training()
        self.assertIn("decoder.multi_stem_frontend.residual_gate", trainable_names)
        self.assertIn("decoder.multi_stem_frontend.stem_embedding", trainable_names)
        self.assertTrue(all(name.startswith("decoder.multi_stem_frontend.") for name in trainable_names))

        trainable_from_model = [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ]
        self.assertEqual(sorted(trainable_names), sorted(trainable_from_model))

    def test_loading_legacy_checkpoint_zero_initializes_residual_gate(self):
        torch.manual_seed(0)
        config = _tiny_config()
        model = AceStepConditionGenerationModel(config)
        legacy_state = {
            key: value for key, value in model.state_dict().items()
            if not key.startswith("decoder.multi_stem_frontend.")
        }

        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir)
            config.save_pretrained(path)
            torch.save(legacy_state, path / "pytorch_model.bin")
            loaded = AceStepConditionGenerationModel.from_pretrained(path)

        frontend = loaded.decoder.multi_stem_frontend
        self.assertEqual(float(frontend.residual_gate), 0.0)
        self.assertEqual(float(frontend.stem_embedding.abs().sum()), 0.0)
        self.assertTrue(torch.allclose(frontend.cross_stem_norm.weight, torch.ones_like(frontend.cross_stem_norm.weight)))


if __name__ == "__main__":
    unittest.main()
