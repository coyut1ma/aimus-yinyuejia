import random
import tempfile
import unittest
from pathlib import Path

import torch

from acestep.models.common.configuration_acestep_v15 import AceStepConfig
from acestep.models.turbo.modeling_acestep_v15_turbo import AceStepConditionGenerationModel
from acestep.training.multistem_adapter import (
    STEM_ORDER,
    build_repaint_masks,
    build_training_context,
    gather_target_latents,
    load_adapter_checkpoint,
    save_adapter_checkpoint,
)


class _Handler:
    def __init__(self):
        self.device = "cpu"
        self.dtype = torch.float32
        self.silence_latent = torch.zeros(1, 16, 3)
        self.model = AceStepConditionGenerationModel(self._tiny_config())

    @staticmethod
    def _tiny_config():
        config = AceStepConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            use_sliding_window=False,
            layer_types=["full_attention", "full_attention"],
            audio_acoustic_hidden_dim=3,
            in_channels=9,
            patch_size=1,
            num_lyric_encoder_hidden_layers=1,
            num_timbre_encoder_hidden_layers=1,
            num_attention_pooler_hidden_layers=1,
            num_audio_decoder_hidden_layers=1,
            fsq_dim=16,
            fsq_input_levels=[4, 4],
            text_hidden_dim=8,
            timbre_hidden_dim=3,
        )
        config._attn_implementation = "eager"
        return config


class MultiStemAdapterTrainingTests(unittest.TestCase):
    def test_gather_target_latents_selects_per_batch_stems(self):
        latents = torch.arange(2 * 4 * 5 * 3, dtype=torch.float32).reshape(2, 4, 5, 3)
        targets = torch.tensor([0, 2])

        gathered = gather_target_latents(latents, targets)

        self.assertTrue(torch.equal(gathered[0], latents[0, 0]))
        self.assertTrue(torch.equal(gathered[1], latents[1, 2]))

    def test_build_repaint_masks_marks_requested_fraction(self):
        masks = build_repaint_masks(3, 20, random.Random(0), min_fraction=0.25, max_fraction=0.25)

        self.assertEqual(tuple(masks.shape), (3, 20))
        self.assertTrue(torch.equal(masks.sum(dim=1), torch.full((3,), 5)))

    def test_build_training_context_silences_target_repaint_region(self):
        handler = _Handler()
        latents = torch.ones(1, 4, 8, 3)
        latents[:, 1] = 2
        targets = torch.tensor([1])
        mask = torch.zeros(1, 8, dtype=torch.bool)
        mask[:, 2:5] = True

        target_latents, context_latents, multi_context = build_training_context(handler, latents, targets, mask)

        self.assertTrue(torch.equal(target_latents, latents[:, 1]))
        self.assertTrue(torch.equal(context_latents[:, 2:5, :3], torch.zeros(1, 3, 3)))
        self.assertTrue(torch.equal(context_latents[:, :2, :3], torch.full((1, 2, 3), 2.0)))
        self.assertEqual(tuple(multi_context.shape), (1, 4, 8, 6))

    def test_adapter_checkpoint_roundtrip(self):
        handler = _Handler()
        model = handler.model
        with torch.no_grad():
            model.decoder.multi_stem_frontend.residual_gate.fill_(0.25)

        with tempfile.TemporaryDirectory() as tmp:
            path = save_adapter_checkpoint(model, Path(tmp), 7, {"stem_order": list(STEM_ORDER)})
            with torch.no_grad():
                model.decoder.multi_stem_frontend.residual_gate.zero_()
            metadata = load_adapter_checkpoint(model, path)

        self.assertEqual(metadata["step"], 7)
        self.assertAlmostEqual(float(model.decoder.multi_stem_frontend.residual_gate), 0.25)


if __name__ == "__main__":
    unittest.main()
