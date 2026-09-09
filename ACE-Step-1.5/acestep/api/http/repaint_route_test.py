import asyncio
import tempfile
import unittest
from pathlib import Path

import torch
from fastapi import FastAPI

from acestep.api.http.repaint_route import RepaintRequest, register_repaint_route


class _Decoder:
    multi_stem_frontend = object()


class _Model:
    decoder = _Decoder()


class _Handler:
    def __init__(self):
        self.model = _Model()
        self.generate_calls = []

    def process_src_audio(self, _path):
        return torch.zeros(2, 32)

    def generate_music(self, **kwargs):
        self.generate_calls.append(kwargs)
        return {
            "success": True,
            "audios": [
                {
                    "tensor": torch.zeros(2, 32),
                    "sample_rate": 48000,
                }
            ],
        }


class RepaintRouteTests(unittest.TestCase):
    def test_repaint_route_forwards_joint_frontend_controls(self):
        app = FastAPI()
        handler = _Handler()
        app.state.handler = handler
        register_repaint_route(app=app, verify_api_key=lambda: None)
        endpoint = next(route.endpoint for route in app.routes if getattr(route, "path", None) == "/v1/repaint")

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "out"
            payload = RepaintRequest(**{
                "stems": {
                    "vocals": "/tmp/vocals.wav",
                    "drums": "/tmp/drums.wav",
                    "bass": "/tmp/bass.wav",
                    "other": "/tmp/other.wav",
                },
                "region": {"start_sec": 1.0, "end_sec": 2.0},
                "target_stems": ["bass"],
                "prompt": "make bass tighter",
                "seeds": [123],
                "output_dir": str(output_dir),
                "joint_frontend": True,
            })

            body = asyncio.run(endpoint(payload, None))

        self.assertEqual(body["candidates"][0]["seed"], 123)
        self.assertIn("bass", body["candidates"][0]["generated_targets"])
        self.assertEqual(len(handler.generate_calls), 1)
        call = handler.generate_calls[0]
        self.assertTrue(call["joint_frontend"])
        self.assertEqual(call["target_stem_id"], 2)
        self.assertEqual(tuple(call["multi_stem_target_wavs"].shape), (1, 4, 2, 32))
        self.assertEqual(call["task_type"], "repaint")


if __name__ == "__main__":
    unittest.main()
