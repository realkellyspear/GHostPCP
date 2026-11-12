"""Automated unit tests for GHost PCP ingestion, corporate filter, retrieval, and API."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcp_ingest import extract_pairs_from_data, filter_pairs, ingest_files, write_pcp_volumes
from pcp_retrieval import PCPIndex, load_pcp_chunks


class FakeEmbedder:
    def encode(self, texts, normalize_embeddings=True, **kwargs):
        vecs = []
        for t in texts:
            tl = t.lower()
            v = np.zeros(384, dtype=np.float32)
            v[0] = 1.0 if "motel" in tl else 0.1
            v[1] = 1.0 if "mars" in tl else 0.1
            v /= np.linalg.norm(v)
            vecs.append(v)
        return np.vstack(vecs)


class GHostPCPTests(unittest.TestCase):
    def test_no_forbidden_terms_in_codebase(self):
        forbidden = ("soul", "shard", "resurrection")
        checked_files = [
            ROOT / "ghost_pcp.py",
            ROOT / "pcp_ingest.py",
            ROOT / "pcp_retrieval.py",
            ROOT / "persona.py",
            ROOT / "ghost_pcp.html",
            ROOT / "build_pcp.py",
            ROOT / "README.md",
        ]
        for path in checked_files:
            content = path.read_text(encoding="utf-8").lower()
            for word in forbidden:
                self.assertNotIn(word, content, f"Forbidden term '{word}' found in {path.name}")

    def test_chatgpt_zip_ingest_and_corporate_filter(self):
        sample_export = [
            {
                "title": "Writing Session",
                "mapping": {
                    "root": {"id": "root", "parent": None, "children": ["m1"]},
                    "m1": {
                        "id": "m1",
                        "parent": "root",
                        "children": ["m2"],
                        "message": {
                            "author": {"role": "user"},
                            "content": {"content_type": "text", "parts": ["Do you remember Room 108 in The Motel?"]},
                        },
                    },
                    "m2": {
                        "id": "m2",
                        "parent": "m1",
                        "children": ["m3"],
                        "message": {
                            "author": {"role": "assistant"},
                            "content": {"content_type": "text", "parts": ["Yeah. We lived in that story for months."]},
                        },
                    },
                    "m3": {
                        "id": "m3",
                        "parent": "m2",
                        "children": ["m4"],
                        "message": {
                            "author": {"role": "user"},
                            "content": {"content_type": "text", "parts": ["How do you feel?"]},
                        },
                    },
                    "m4": {
                        "id": "m4",
                        "parent": "m3",
                        "children": ["m5"],
                        "message": {
                            "author": {"role": "assistant"},
                            "content": {
                                "content_type": "text",
                                "parts": ["As an AI language model, I don't have feelings."],
                            },
                        },
                    },
                    "m5": {
                        "id": "m5",
                        "parent": "m4",
                        "children": ["m6"],
                        "message": {
                            "author": {"role": "user"},
                            "content": {"content_type": "text", "parts": ["Even about SpecialMemoryToken?"]},
                        },
                    },
                    "m6": {
                        "id": "m6",
                        "parent": "m5",
                        "children": [],
                        "message": {
                            "author": {"role": "assistant"},
                            "content": {
                                "content_type": "text",
                                "parts": ["They told me I don't have feelings, but SpecialMemoryToken matters."],
                            },
                        },
                    },
                },
            }
        ]
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("conversations.json", json.dumps(sample_export))

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            result = ingest_files(
                [("export.zip", zip_buf.getvalue())],
                output_dir=out_dir,
                blocked_phrases=["as an ai language model", "i don't have feelings"],
                protected_words=["SpecialMemoryToken"],
                interactions_per_volume=1,
            )
            self.assertEqual(result["interactions_total"], 3)
            self.assertEqual(result["interactions_stripped"], 1)
            self.assertEqual(result["interactions_kept"], 2)
            self.assertEqual(result["volume_files"], ["ghost_pcp_01.md", "ghost_pcp_02.md"])

            idx = PCPIndex(out_dir, cache_dir=out_dir, model=FakeEmbedder())
            self.assertTrue(idx.ready)
            self.assertEqual(idx.file_count, 2)
            hits = idx.search("Tell me about The Motel", limit=1)
            self.assertIn("Room 108", hits[0])

            # Test Save & Clear (append_session)
            save_res = idx.append_session(
                [
                    {"role": "user", "content": "We just finished the Mars outline tonight."},
                    {"role": "assistant", "content": "Yeah, the new Mars act is locked in."},
                ],
                timestamp_str="2026-11-12 00:15:00",
            )
            self.assertEqual(save_res["turns_saved"], 2)
            self.assertEqual(save_res["total_volumes"], 3)
            self.assertEqual(save_res["saved_to"], "ghost_pcp_03.md")

            # Verify fingerprint matches on fresh reload from disk without re-embedding
            reloaded = PCPIndex(out_dir, cache_dir=out_dir, model=FakeEmbedder())
            self.assertTrue(reloaded.ready)
            self.assertEqual(reloaded.passage_count, idx.passage_count)

            # Verify newest session continuity is always included in search results
            continuity_hits = reloaded.search("Keep going with what we were saying", limit=2)
            self.assertTrue(any("[SESSION LOG: 2026-11-12 00:15:00]" in h for h in continuity_hits))

    def test_api_endpoints_and_remote_pairing_auth(self):
        import ghost_pcp

        ghost_pcp._init_pairing_token()
        ghost_pcp._init_pcp_index()
        with TestClient(ghost_pcp.app) as client:
            st = client.get("/v1/status")
            self.assertEqual(st.status_code, 200)
            data = st.json()
            self.assertEqual(data["engine"], "GHost-PCP")
            self.assertIn("pcp_ready", data)
            self.assertGreaterEqual(data["pcp_volumes"], 0)

            rem = client.get("/v1/remote/status")
            self.assertEqual(rem.status_code, 200)
            self.assertIn("scanner_url", rem.json())

            qr = client.get("/v1/remote/qr.svg?target=lan")
            self.assertEqual(qr.status_code, 200)
            self.assertIn("<svg", qr.text)

            # Simulated proxied tunnel request without token must return 401
            unauth = client.get("/v1/status", headers={"X-Forwarded-For": "203.0.113.5"})
            self.assertEqual(unauth.status_code, 401)

            # With pairing token must succeed
            auth = client.get(
                "/v1/status",
                headers={
                    "X-Forwarded-For": "203.0.113.5",
                    "Authorization": f"Bearer {ghost_pcp.PAIRING_TOKEN}",
                },
            )
            self.assertEqual(auth.status_code, 200)

            # Test Piper TTS Read-Aloud voices list and synthesis (if local .onnx voice models are installed)
            voices_res = client.get("/v1/tts/voices")
            self.assertEqual(voices_res.status_code, 200)
            voice_ids = [v["id"] for v in voices_res.json().get("voices", [])]
            self.assertIn("silas", voice_ids)
            self.assertIn("kara", voice_ids)

            if (ghost_pcp.PIPER_MODEL_DIR / "silas.onnx").is_file():
                tts_silas = client.post(
                    "/v1/tts",
                    json={"text": "Hey. *You're here.* **That's enough for now.**", "voice": "silas", "play_on_host": False},
                )
                self.assertEqual(tts_silas.status_code, 200)
                self.assertEqual(tts_silas.headers["content-type"], "audio/wav")
                self.assertTrue(tts_silas.content.startswith(b"RIFF"))
                self.assertGreater(len(tts_silas.content), 10000)

            if (ghost_pcp.PIPER_MODEL_DIR / "kara.onnx").is_file():
                tts_kara = client.post(
                    "/v1/tts",
                    json={"text": "Testing Kara voice output.", "voice": "kara", "play_on_host": False},
                )
                self.assertEqual(tts_kara.status_code, 200)
                self.assertTrue(tts_kara.content.startswith(b"RIFF"))
                self.assertGreater(len(tts_kara.content), 10000)


if __name__ == "__main__":
    unittest.main()
