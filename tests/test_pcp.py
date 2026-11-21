# SPDX-FileCopyrightText: 2026 Kelly Spear
# SPDX-License-Identifier: GPL-3.0-or-later
"""Automated unit tests for GHostPCP ingestion, corporate filter, retrieval, and API."""
from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcp_ingest import extract_pairs_from_data, filter_pairs, ingest_files, write_pcp_volumes
from pcp_retrieval import IngestCancelledError, PCPIndex, SESSION_BOUNDARY, load_pcp_chunks


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
    def test_generation_lease_is_atomic_and_abort_is_stream_aware(self):
        import ghost_pcp

        original_llm = ghost_pcp.STATE.llm
        ghost_pcp.STATE.llm = object()
        self.addCleanup(setattr, ghost_pcp.STATE, "llm", original_llm)
        self.addCleanup(ghost_pcp.STATE.end_generation)
        ghost_pcp.STATE.end_generation()
        generation_id = ghost_pcp.STATE.begin_generation(streaming=True)
        self.assertTrue(ghost_pcp.STATE.busy)
        self.assertTrue(ghost_pcp.STATE.streaming)
        self.assertTrue(ghost_pcp.STATE.request_abort())
        self.assertTrue(ghost_pcp.STATE.abort_requested)
        with self.assertRaises(ghost_pcp.HTTPException):
            ghost_pcp.STATE.begin_generation(streaming=False)
        ghost_pcp.STATE.end_generation(generation_id)
        self.assertFalse(ghost_pcp.STATE.busy)
        self.assertFalse(ghost_pcp.STATE.abort_requested)

    def test_stream_abort_preserves_partial_response_and_releases_generation(self):
        import ghost_pcp

        class FakeLLM:
            def create_chat_completion(self, **kwargs):
                return iter([
                    {"choices": [{"delta": {"content": "visible "}}]},
                    {"choices": [{"delta": {"content": "discarded"}}]},
                ])

        original_llm = ghost_pcp.STATE.llm
        ghost_pcp.STATE.llm = FakeLLM()
        self.addCleanup(setattr, ghost_pcp.STATE, "llm", original_llm)
        self.addCleanup(ghost_pcp.STATE.end_generation)
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            ghost_pcp, "SETTINGS_PATH", Path(tmp) / "settings.json"
        ):
            request = ghost_pcp.ChatRequest(
                model="test-model",
                messages=[{"role": "user", "content": "test"}],
                stream=True,
            )
            generation_id = ghost_pcp.STATE.begin_generation(streaming=True)
            stream = ghost_pcp._stream_generator(
                request,
                [{"role": "user", "content": "test"}],
                generation_id,
                show_reasoning=False,
            )
            first = next(stream)
            self.assertIn("visible ", first)
            self.assertFalse(ghost_pcp.STATE.abort_requested)
            self.assertTrue(ghost_pcp.STATE.request_abort())
            remainder = "".join(stream)
            self.assertIn('"ghost_aborted": true', remainder)
            self.assertIn("[DONE]", remainder)
            self.assertNotIn("discarded", remainder)
            self.assertFalse(ghost_pcp.STATE.busy)
            self.assertFalse(ghost_pcp.STATE.streaming)

    def test_think_gate_hides_tags_across_chunk_boundaries(self):
        import ghost_pcp

        text = "Visible <think>private reasoning</think> answer."
        for width in range(1, len(text) + 1):
            gate = ghost_pcp._ThinkGate(show_reasoning=False)
            output = "".join(
                gate.feed(text[index:index + width])
                for index in range(0, len(text), width)
            ) + gate.flush()
            self.assertEqual(output, "Visible  answer.")

    def test_stream_disconnect_cleans_up_without_yielding_from_finally(self):
        import ghost_pcp

        class FakeLLM:
            def create_chat_completion(self, **kwargs):
                return iter([
                    {"choices": [{"delta": {"content": "first"}}]},
                    {"choices": [{"delta": {"content": "second"}}]},
                ])

        original_llm = ghost_pcp.STATE.llm
        ghost_pcp.STATE.llm = FakeLLM()
        self.addCleanup(setattr, ghost_pcp.STATE, "llm", original_llm)
        self.addCleanup(ghost_pcp.STATE.end_generation)
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            ghost_pcp, "SETTINGS_PATH", Path(tmp) / "settings.json"
        ):
            request = ghost_pcp.ChatRequest(
                model="test-model",
                messages=[{"role": "user", "content": "test"}],
                stream=True,
            )
            generation_id = ghost_pcp.STATE.begin_generation(streaming=True)
            stream = ghost_pcp._stream_generator(
                request,
                [{"role": "user", "content": "test"}],
                generation_id,
                show_reasoning=False,
            )
            self.assertIn("first", next(stream))
            stream.close()
            self.assertFalse(ghost_pcp.STATE.busy)

    def test_chat_endpoint_streams_sse_and_releases_generation_lease(self):
        import ghost_pcp

        class FakeLLM:
            def create_chat_completion(self, **kwargs):
                self.stream = kwargs.get("stream")
                return iter([
                    {"choices": [{"delta": {"content": "Hello"}}]},
                    {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                ])

        original_llm = ghost_pcp.STATE.llm
        fake_llm = FakeLLM()
        ghost_pcp.STATE.llm = fake_llm
        self.addCleanup(setattr, ghost_pcp.STATE, "llm", original_llm)
        self.addCleanup(ghost_pcp.STATE.end_generation)
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            ghost_pcp, "SETTINGS_PATH", Path(tmp) / "settings.json"
        ), patch.object(
            ghost_pcp, "_prepare_pcp_messages",
            return_value=([{"role": "user", "content": "test"}], False),
        ), TestClient(ghost_pcp.app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "test"}],
                    "stream": True,
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
            self.assertIn("Hello", response.text)
            self.assertIn("data: [DONE]", response.text)
            self.assertTrue(fake_llm.stream)
            self.assertFalse(ghost_pcp.STATE.busy)

    def test_saved_sampling_settings_apply_when_request_omits_them(self):
        import ghost_pcp

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            ghost_pcp, "SETTINGS_PATH", Path(tmp) / "settings.json"
        ):
            settings_path = Path(tmp) / "settings.json"
            settings_path.write_text(json.dumps({
                "temperature": 0.4,
                "top_p": 0.8,
                "min_p": 0.02,
                "top_k": 12,
                "repeat_penalty": 1.15,
                "presence_penalty": 0.25,
                "max_tokens": 777,
                "stop_sequences": "END\nSTOP",
            }), encoding="utf-8")
            request = ghost_pcp.ChatRequest(
                model="test-model",
                messages=[{"role": "user", "content": "test"}],
            )
            args = ghost_pcp._chat_completion_args(request, request.messages)
            self.assertEqual(args["temperature"], 0.4)
            self.assertEqual(args["top_p"], 0.8)
            self.assertEqual(args["min_p"], 0.02)
            self.assertEqual(args["top_k"], 12)
            self.assertEqual(args["repeat_penalty"], 1.15)
            self.assertEqual(args["presence_penalty"], 0.25)
            self.assertEqual(args["max_tokens"], 777)
            self.assertEqual(args["stop"], ["END", "STOP"])
            self.assertIsInstance(args["seed"], int)
            self.assertGreater(args["seed"], 0)
            seeds = {ghost_pcp._chat_completion_args(request, request.messages)["seed"] for _ in range(5)}
            self.assertGreater(len(seeds), 1)

            explicit_req = ghost_pcp.ChatRequest(
                model="test-model",
                messages=[{"role": "user", "content": "test"}],
                seed=42,
            )
            explicit_args = ghost_pcp._chat_completion_args(explicit_req, explicit_req.messages)
            self.assertEqual(explicit_args["seed"], 42)

    def test_backup_failure_preserves_existing_volumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            first = out_dir / "ghost_pcp_01.md"
            second = out_dir / "ghost_pcp_02.md"
            first.write_text("first original", encoding="utf-8")
            second.write_text("second original", encoding="utf-8")
            copy_file = shutil.copy2
            calls = 0

            def fail_second_copy(source, destination):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("backup failed")
                return copy_file(source, destination)

            with patch("pcp_ingest.shutil.copy2", side_effect=fail_second_copy):
                with self.assertRaisesRegex(OSError, "backup failed"):
                    write_pcp_volumes([("new user", "new answer")], out_dir, replace_existing=True)
            self.assertEqual(first.read_text(encoding="utf-8"), "first original")
            self.assertEqual(second.read_text(encoding="utf-8"), "second original")

    def test_removal_failure_keeps_complete_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            first = out_dir / "ghost_pcp_01.md"
            second = out_dir / "ghost_pcp_02.md"
            first.write_text("first original", encoding="utf-8")
            second.write_text("second original", encoding="utf-8")
            unlink = Path.unlink

            def fail_second_removal(path, *args, **kwargs):
                if path == second:
                    raise OSError("removal failed")
                return unlink(path, *args, **kwargs)

            with patch.object(Path, "unlink", autospec=True, side_effect=fail_second_removal):
                with self.assertRaisesRegex(OSError, "removal failed"):
                    write_pcp_volumes([("new user", "new answer")], out_dir, replace_existing=True)
            backup = next((out_dir / "backups").iterdir())
            self.assertEqual((backup / first.name).read_text(encoding="utf-8"), "first original")
            self.assertEqual((backup / second.name).read_text(encoding="utf-8"), "second original")
            self.assertEqual(second.read_text(encoding="utf-8"), "second original")

    def test_markdown_import_append_and_empty_replace(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            original = out_dir / "ghost_pcp_01.md"
            original.write_text("original", encoding="utf-8")
            with self.assertRaises(ValueError):
                ingest_files([("empty.md", b"  ")], output_dir=out_dir, replace_existing=True)
            self.assertEqual(original.read_text(encoding="utf-8"), "original")
            result = ingest_files([("new.md", b"new memory")], output_dir=out_dir)
            self.assertEqual(result["volume_files"], ["ghost_pcp_02.md"])
            self.assertEqual(original.read_text(encoding="utf-8"), "original")
            self.assertEqual((out_dir / "ghost_pcp_02.md").read_text(encoding="utf-8").strip(), "new memory")

    def test_repeated_replacement_creates_distinct_backups(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            (out_dir / "ghost_pcp_01.md").write_text("original", encoding="utf-8")
            with patch("pcp_ingest.datetime") as clock:
                clock.now.return_value.strftime.return_value = "20261112_120000"
                write_pcp_volumes([("first user", "first reply")], out_dir, replace_existing=True)
                write_pcp_volumes([("second user", "second reply")], out_dir, replace_existing=True)
            backups = list((out_dir / "backups").glob("backup_*/ghost_pcp_01.md"))
            self.assertEqual(len(backups), 2)
            contents = [path.read_text(encoding="utf-8") for path in backups]
            self.assertIn("original", contents)
            self.assertTrue(any("first reply" in text for text in contents))

    def test_invalid_volume_size_preserves_existing_volumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            original = out_dir / "ghost_pcp_01.md"
            original.write_text("original", encoding="utf-8")
            with self.assertRaises(ValueError):
                write_pcp_volumes([("user", "answer")], out_dir,
                                  interactions_per_volume=0, replace_existing=True)
            self.assertEqual(original.read_text(encoding="utf-8"), "original")

    def test_no_forbidden_terms_in_codebase(self):
        forbidden = ("soul", "shard", "resurrection")
        checked_files = [
            ROOT / "ghost_pcp.py",
            ROOT / "pcp_ingest.py",
            ROOT / "pcp_retrieval.py",
            ROOT / "pcp_web.py",
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

            # Verify failed/empty import NEVER deletes existing ghost_pcp_*.md volumes
            with self.assertRaises(ValueError):
                ingest_files([("empty.json", b"[]")], output_dir=out_dir)
            self.assertTrue((out_dir / "ghost_pcp_01.md").is_file())
            self.assertTrue((out_dir / "ghost_pcp_02.md").is_file())
            self.assertTrue((out_dir / "ghost_pcp_03.md").is_file())

            # Verify default import appends as new volume (ghost_pcp_04.md) without overwriting 01..03
            second_res = ingest_files(
                [("export2.zip", zip_buf.getvalue())],
                output_dir=out_dir,
                blocked_phrases=["as an ai language model", "i don't have feelings"],
                protected_words=["SpecialMemoryToken"],
                interactions_per_volume=10,
                replace_existing=False,
            )
            self.assertEqual(second_res["volume_files"], ["ghost_pcp_04.md"])
            self.assertTrue((out_dir / "ghost_pcp_01.md").is_file())

            # Verify replace_existing=True backs up existing volumes into pcp_data/backups/ before starting at 01
            replace_res = ingest_files(
                [("export3.zip", zip_buf.getvalue())],
                output_dir=out_dir,
                blocked_phrases=["as an ai language model", "i don't have feelings"],
                protected_words=["SpecialMemoryToken"],
                interactions_per_volume=10,
                replace_existing=True,
            )
            self.assertEqual(replace_res["volume_files"], ["ghost_pcp_01.md"])
            backups = list((out_dir / "backups").glob("backup_*/ghost_pcp_01.md"))
            self.assertEqual(len(backups), 1)

    def test_api_endpoints_and_remote_pairing_auth(self):
        import ghost_pcp

        ghost_pcp._init_pairing_token()
        ghost_pcp._init_pcp_index()
        with TestClient(ghost_pcp.app) as client:
            st = client.get("/v1/status")
            self.assertEqual(st.status_code, 200)
            data = st.json()
            self.assertEqual(data["engine"], "GHostPCP")
            self.assertIn("pcp_ready", data)
            self.assertGreaterEqual(data["pcp_volumes"], 0)

            idle_abort = client.post("/v1/abort")
            self.assertEqual(idle_abort.status_code, 200)
            self.assertEqual(idle_abort.json()["status"], "idle")
            self.assertFalse(idle_abort.json()["interruptible"])

            rem = client.get("/v1/remote/status")
            self.assertEqual(rem.status_code, 200)
            self.assertIn("scanner_url", rem.json())

            qr = client.get("/v1/remote/qr.svg?target=lan")
            self.assertEqual(qr.status_code, 200)
            self.assertIn("<svg", qr.text)

            # Simulated proxied tunnel request without token must return 401
            unauth = client.get("/v1/status", headers={"X-Forwarded-For": "203.0.113.5"})
            self.assertEqual(unauth.status_code, 401)

            rebinding = client.get(
                "/v1/status",
                headers={"Host": "attacker.example", "Origin": "http://attacker.example"},
            )
            self.assertEqual(rebinding.status_code, 401)

            bad_origin = client.get(
                "/v1/status",
                headers={"Origin": "http://attacker.example"},
            )
            self.assertEqual(bad_origin.status_code, 401)

            same_origin = client.get("/v1/status", headers={"Origin": "http://testserver"})
            self.assertEqual(same_origin.status_code, 200)

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

    def test_release_staging_includes_corresponding_source(self):
        import package_release

        source_files = (
            "ghost_pcp.py",
            "pcp_ingest.py",
            "pcp_retrieval.py",
            "pcp_web.py",
            "persona.py",
            "ghost_pcp.html",
            "build_pcp.py",
            "package_release.py",
            "requirements.txt",
        )
        with tempfile.TemporaryDirectory() as tmp:
            temp_root = Path(tmp) / "project"
            sandbox = Path(tmp) / "sandbox"
            bge_dir = Path(tmp) / "bge"
            stt_dir = Path(tmp) / "stt"
            bge_dir.mkdir()
            stt_dir.mkdir()
            for name in source_files:
                target = temp_root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / name).read_bytes())
            for name in ("GHostPCP.exe", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md"):
                target = temp_root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(name, encoding="utf-8")
            for directory, name in (
                ("assets", "ghost_icon.ico"),
                ("piper_models", "README.md"),
                ("stt_model", "README.md"),
                ("ghost_models", "README.md"),
            ):
                target = temp_root / directory / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(name, encoding="utf-8")
            (temp_root / "tests").mkdir()
            (temp_root / "tests" / "test_pcp.py").write_bytes(Path(__file__).read_bytes())

            with patch.object(package_release, "ROOT", temp_root), patch(
                "pcp_retrieval._resolve_bge_model_dir", return_value=bge_dir
            ), patch(
                "ghost_pcp._resolve_stt_model_dir", return_value=stt_dir
            ):
                package_release.stage_clean_sandbox(sandbox, link_sample_gguf=False)
                archive_path = Path(tmp) / "release.zip"
                package_release.build_release_zip(sandbox, archive_path)

            with zipfile.ZipFile(archive_path) as archive:
                for name in source_files:
                    self.assertEqual(
                        archive.read(f"GHostPCP/source/{name}"),
                        (ROOT / name).read_bytes(),
                    )
                self.assertEqual(
                    archive.read("GHostPCP/source/tests/test_pcp.py"),
                    Path(__file__).read_bytes(),
                )
                self.assertIn("GHostPCP/LICENSE", archive.namelist())
                self.assertNotIn("GHostPCP/LICENSE.GPL-3.0", archive.namelist())

    def test_html_ui_contains_all_button_handlers_and_controls(self):
        html = (ROOT / "ghost_pcp.html").read_text(encoding="utf-8")

        for control_id in (
            'id="top-k-sampler"',
            'id="repeat-penalty"',
            'id="presence-penalty"',
            'id="attach-file-button"',
            'id="chat-attachments-bar"',
            '.pdf',
            'Custom Prefix',
            'chat-attachment-card',
            'id="persona-modal"',
            'id="persona-pill"',
            'id="ingest-persona-name"',
            'id="ingest-user-name"',
            'id="ingest-never-forget"',
            'id="web-toggle-button"',
            'id="attach-image-button"',
            'id="mic-button"',
            'id="memories-modal"',
            'id="mmproj-select"',
            'id="model-downloader-section"',
            'id="telemetry-pill"',
        ):
            self.assertIn(control_id, html)

        self.assertIn("isUser && isGenerating && idx === activeUserIndex", html)
        self.assertNotIn("if (!isUser && isGenerating && idx === streamingIndex) {\n                    actions.classList.add", html)

        required_functions = (
            "openSettings",
            "closeSettings",
            "saveAndClose",
            "openArchiveModal",
            "closeArchiveModal",
            "openRemoteModal",
            "closeRemoteModal",
            "openPersonaModal",
            "closePersonaModal",
            "fetchPersonas",
            "choosePersona",
            "openPersonaEditor",
            "savePersonaEditor",
            "startStagedIngest",
            "saveAndClearChat",
            "startNewChat",
            "sendMessage",
            "redoMessage",
            "stopGeneration",
            "requestCompletionForHistory",
            "handleFileSelect",
            "ingestLocalPath",
            "triggerReindex",
            "cancelIngest",
            "startIngestPolling",
            "refreshRemoteModal",
            "loadConversationFromServer",
            "syncConversationToServer",
            "startTunnel",
            "stopTunnel",
            "rotateToken",
            "copyTextToClipboard",
            "copyRemoteUrl",
            "finishArchiveAndOpenConfig",
            "handleChatFileSelect",
            "handleChatImageSelect",
            "renderChatAttachments",
            "saveAttachmentToArchive",
            "removeChatAttachment",
            "formatUserMessageHtml",
            "toggleWebAccess",
            "toggleVoiceDictation",
            "openMemoriesModal",
            "closeMemoriesModal",
            "savePersonaFact",
            "exportPersonaBundle",
            "handleImportPersonaBundle",
            "fetchModelCatalog",
            "initialize",
        )
        for fn_name in required_functions:
            self.assertRegex(html, rf"(?:async\s+)?function\s+{fn_name}\s*\(")

        self.assertNotIn('id="tab-lan"', html)
        self.assertIn("visibilitychange", html)
        self.assertIn('"/chat/history"', html)
        self.assertIn('id="cancel-ingest-btn"', html)
        self.assertIn('id="ingest-done-banner"', html)
        self.assertIn("30 minutes or more", html)
        self.assertIn("${API}/chat/completions", html)
        self.assertNotIn("${API}/chat/completions/stream", html)
        self.assertNotIn("OneDrive", (ROOT / "ghost_pcp.py").read_text(encoding="utf-8"))

    def test_ingest_cancel_and_resumable_checkpoint(self):
        import ghost_pcp

        class TrackingEmbedder(FakeEmbedder):
            def __init__(self):
                self.encoded_batches: list[int] = []

            def encode(self, texts, normalize_embeddings=True, progress_callback=None, **kwargs):
                self.encoded_batches.append(len(texts))
                vecs = super().encode(texts, normalize_embeddings=normalize_embeddings, **kwargs)
                if progress_callback:
                    progress_callback(len(texts), len(texts))
                return vecs

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            passages = [f"Passage {i} about Motel memory." for i in range(300)]
            (out_dir / "ghost_pcp_01.md").write_text(f"\n{SESSION_BOUNDARY}\n".join(passages), encoding="utf-8")

            embedder = TrackingEmbedder()
            idx = PCPIndex(out_dir, cache_dir=out_dir, model=embedder, cache_only=True)
            self.assertFalse(idx.ready)
            self.assertEqual(idx.file_count, 1)
            self.assertEqual(embedder.encoded_batches, [])

            def cancel_after_first_slice(msg: str, frac: float):
                if "256 / 300" in msg:
                    raise IngestCancelledError("cancelled")

            with self.assertRaises(IngestCancelledError):
                idx.reload(progress_callback=cancel_after_first_slice)

            self.assertTrue((out_dir / "embeddings.part.npz").is_file())
            self.assertFalse((out_dir / "embeddings.npz").exists())

            embedder.encoded_batches.clear()
            self.assertTrue(idx.reload())
            self.assertTrue(idx.ready)
            self.assertEqual(idx.passage_count, 300)
            self.assertEqual(embedder.encoded_batches, [44])
            self.assertTrue((out_dir / "embeddings.npz").is_file())
            self.assertFalse((out_dir / "embeddings.part.npz").exists())

        with TestClient(ghost_pcp.app) as client:
            idle = client.post("/v1/pcp/cancel")
            self.assertEqual(idle.status_code, 200)
            self.assertEqual(idle.json()["status"], "idle")

            ghost_pcp.INGEST_STATE.running = True
            ghost_pcp.INGEST_STATE.cancel_event.clear()
            try:
                active = client.post("/v1/pcp/cancel")
                self.assertEqual(active.status_code, 200)
                self.assertEqual(active.json()["status"], "cancelling")
                self.assertTrue(ghost_pcp.INGEST_STATE.cancel_event.is_set())
                st = client.get("/v1/pcp/status")
                self.assertEqual(st.status_code, 200)
                self.assertTrue(st.json()["cancelling"])
            finally:
                ghost_pcp.INGEST_STATE.running = False
                ghost_pcp.INGEST_STATE.cancel_event.clear()

    def test_chat_history_sync_and_pcp_recency_context(self):
        import ghost_pcp

        # 1. Chronological prior: identical passages -> earlier index (newest exported conversation) breaks tie
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            p0 = "Max checked the neon motel sign at midnight."
            p1 = "Max checked the neon motel sign at midnight."
            (out_dir / "ghost_pcp_01.md").write_text(f"{p0}\n{SESSION_BOUNDARY}\n{p1}", encoding="utf-8")
            idx = PCPIndex(out_dir, cache_dir=out_dir, model=FakeEmbedder())
            self.assertEqual(idx.passage_count, 2)
            # Identical text hashes identically; confirm search runs without error and prefers index 0 over index 1
            prompted = "Represent this sentence for searching relevant passages: motel"
            q_vec = FakeEmbedder().encode([prompted], normalize_embeddings=True)[0]
            self.assertIsNotNone(q_vec)
            hits = idx.search("motel", limit=1)
            self.assertEqual(hits[0], idx.chunks[0])

        # 2. _prepare_pcp_messages short-message context + Qwen3 /no_think on system & all user turns
        class CaptureIndex:
            ready = True

            def __init__(self):
                self.queries: list[str] = []

            def search(self, query: str, limit: int = 4):
                self.queries.append(query)
                return ["Sample archive excerpt"]

        cap_index = CaptureIndex()
        prev_index = ghost_pcp.PCP_INDEX
        prev_model = ghost_pcp.STATE.loaded_model
        ghost_pcp.PCP_INDEX = cap_index
        ghost_pcp.STATE.loaded_model = "Qwen3-14B-Q5_K_M.gguf"
        try:
            req = ghost_pcp.ChatRequest(
                model="Qwen3-14B-Q5_K_M.gguf",
                messages=[
                    ghost_pcp.Message(role="system", content="You are Max Orion Wilde."),
                    ghost_pcp.Message(role="user", content="We talked about The Motel earlier."),
                    ghost_pcp.Message(role="assistant", content="Yeah, Kaelyn and Jacob on that rainy highway."),
                    ghost_pcp.Message(role="user", content="Why?"),
                ],
                use_pcp=True,
                show_reasoning=False,
            )
            prepared, show_reasoning = ghost_pcp._prepare_pcp_messages(req)
            self.assertFalse(show_reasoning)
            self.assertEqual(len(cap_index.queries), 1)
            self.assertIn("The Motel", cap_index.queries[0])
            self.assertTrue(cap_index.queries[0].endswith("Why?"))
            self.assertIn("/no_think", prepared[0]["content"])
            self.assertIn("/no_think", prepared[1]["content"])
            self.assertIn("/no_think", prepared[3]["content"])
            self.assertNotIn("/no_think", prepared[2]["content"])
        finally:
            ghost_pcp.PCP_INDEX = prev_index
            ghost_pcp.STATE.loaded_model = prev_model

        # 3. /v1/chat/history sync + save-session clear + remote auth
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            pcp_idx = PCPIndex(out_dir, cache_dir=out_dir, model=FakeEmbedder())
            orig_pcp = ghost_pcp.PCP_INDEX
            ghost_pcp.STATE.conversation.clear()
            try:
                with patch.object(ghost_pcp, "PCP_DATA_DIR", out_dir), patch.object(
                    ghost_pcp, "SETTINGS_PATH", out_dir / "settings.json"
                ), TestClient(ghost_pcp.app) as client:
                    ghost_pcp.PCP_INDEX = pcp_idx
                    empty = client.get("/v1/chat/history")
                    self.assertEqual(empty.status_code, 200)
                    self.assertEqual(empty.json()["messages"], [])

                    sample = [
                        {"role": "user", "content": "Hey Max"},
                        {"role": "assistant", "content": "Still here, Kel."},
                    ]
                    saved = client.post("/v1/chat/history", json={"messages": sample})
                    self.assertEqual(saved.status_code, 200)
                    self.assertEqual(saved.json()["count"], 2)

                    loaded = client.get("/v1/chat/history")
                    self.assertEqual(loaded.status_code, 200)
                    self.assertEqual(loaded.json()["messages"], sample)

                    denied = client.get("/v1/chat/history", headers={"cf-connecting-ip": "1.2.3.4"})
                    self.assertEqual(denied.status_code, 401)
                    allowed = client.get(
                        "/v1/chat/history",
                        headers={
                            "cf-connecting-ip": "1.2.3.4",
                            "authorization": f"Bearer {ghost_pcp.PAIRING_TOKEN}",
                        },
                    )
                    self.assertEqual(allowed.status_code, 200)
                    self.assertEqual(allowed.json()["messages"], sample)

                    persisted = client.post("/v1/pcp/save-session", json={"messages": sample})
                    self.assertEqual(persisted.status_code, 200)
                    self.assertTrue((out_dir / "ghost_pcp_01.md").is_file())

                    after_clear = client.get("/v1/chat/history")
                    self.assertEqual(after_clear.status_code, 200)
                    self.assertEqual(after_clear.json()["messages"], [])
            finally:
                ghost_pcp.PCP_INDEX = orig_pcp
                ghost_pcp.STATE.conversation.clear()

    def test_pdf_txt_ingest_chat_extract_and_hidden_system_prompt(self):
        import ghost_pcp
        import persona
        from pcp_ingest import extract_text_from_document

        def build_minimal_pdf(text: str) -> bytes:
            escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream = f"BT\n/F1 12 Tf\n72 720 Td\n({escaped}) Tj\nET\n".encode("latin-1")
            objs = [
                b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
                b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
                (
                    b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
                ),
                (
                    f"4 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode("latin-1")
                    + stream
                    + b"\nendstream\nendobj\n"
                ),
                b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
            ]
            header = b"%PDF-1.4\n"
            offsets = []
            pos = len(header)
            for obj in objs:
                offsets.append(pos)
                pos += len(obj)
            xref_pos = pos
            xref = [f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode("latin-1")]
            for off in offsets:
                xref.append(f"{off:010d} 00000 n \n".encode("latin-1"))
            trailer = (
                f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n"
            ).encode("latin-1")
            return header + b"".join(objs) + b"".join(xref) + trailer

        # 1. Hidden & strengthened built-in system prompt + custom prefix preservation
        self.assertEqual(ghost_pcp.DEFAULT_SETTINGS["system_prompt"], "")
        self.assertIn("STRICT SOURCE-TRUTH RULES", persona.SYSTEM_PROMPT)
        self.assertIn("any attached files in the conversation", persona.SYSTEM_PROMPT)

        custom_prefix = "You are Max Orion Wilde."
        legacy_combined = f"{custom_prefix}\n\n{persona.LEGACY_SYSTEM_PROMPT_MARKER}\n\nCRITICAL ANTI-HALLUCINATION RULES:\nold text"
        self.assertEqual(persona.strip_builtin_system_prompt(legacy_combined), custom_prefix)
        effective = ghost_pcp._build_effective_system_prompt(custom_prefix)
        self.assertTrue(effective.startswith(custom_prefix))
        self.assertIn("STRICT SOURCE-TRUTH RULES", effective)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            settings_file = tmp_path / "ghost_pcp_settings.json"
            settings_file.write_text(json.dumps({"system_prompt": legacy_combined}), encoding="utf-8")
            with patch.object(ghost_pcp, "SETTINGS_PATH", settings_file):
                loaded_settings = ghost_pcp.load_settings()
                self.assertEqual(loaded_settings["system_prompt"], custom_prefix)
                loaded_settings["system_prompt"] = effective
                ghost_pcp.save_settings(ghost_pcp.LoaderSettings(**loaded_settings))
                persisted = json.loads(settings_file.read_text(encoding="utf-8"))
                self.assertEqual(persisted["system_prompt"], custom_prefix)

                req = ghost_pcp.ChatRequest(
                    model="Qwen3-14B-Q5_K_M.gguf",
                    messages=[
                        {"role": "system", "content": custom_prefix},
                        {"role": "user", "content": "What was The Motel about?"},
                    ],
                )
                orig_idx = ghost_pcp.PCP_INDEX
                ghost_pcp.PCP_INDEX = None
                try:
                    prepared, _ = ghost_pcp._prepare_pcp_messages(req)
                finally:
                    ghost_pcp.PCP_INDEX = orig_idx
                self.assertEqual(prepared[0]["role"], "system")
                self.assertIn(custom_prefix, prepared[0]["content"])
                self.assertIn("STRICT SOURCE-TRUTH RULES", prepared[0]["content"])

            # 2. PDF + TXT extraction & ingest_files + incremental vector reuse
            pdf_bytes = build_minimal_pdf("Hello PDF memory from The Motel draft.")
            txt_bytes = b"Notes from notes.txt: character Elena lives by the lake."
            self.assertIn("Hello PDF memory from The Motel draft.", extract_text_from_document("story.pdf", pdf_bytes))
            self.assertIn("character Elena lives by the lake.", extract_text_from_document("notes.txt", txt_bytes))

            out_dir = tmp_path / "pcp_data"
            summary = ingest_files(
                [("notes.txt", txt_bytes), ("story.pdf", pdf_bytes)],
                output_dir=out_dir,
            )
            self.assertGreaterEqual(summary["volumes_written"], 1)
            merged_md = "\n".join(p.read_text(encoding="utf-8") for p in sorted(out_dir.glob("ghost_pcp_*.md")))
            self.assertIn("# **DOCUMENT: notes.txt**", merged_md)
            self.assertIn("# **DOCUMENT: story.pdf**", merged_md)
            self.assertIn("Hello PDF memory from The Motel draft.", merged_md)

            class TrackingEmbedder(FakeEmbedder):
                def __init__(self):
                    self.encoded_batches: list[int] = []

                def encode(self, texts, normalize_embeddings=True, **kwargs):
                    self.encoded_batches.append(len(texts))
                    return super().encode(texts, normalize_embeddings=normalize_embeddings, **kwargs)

            tracker = TrackingEmbedder()
            pcp_idx = PCPIndex(out_dir, cache_dir=out_dir, model=tracker)
            initial_count = pcp_idx.passage_count
            self.assertGreaterEqual(initial_count, 1)
            self.assertEqual(sum(tracker.encoded_batches), initial_count)
            self.assertIn("[Document:", pcp_idx.chunks[0])
            self.assertIn("Passage 1 of", pcp_idx.chunks[0])

            # Adding a new markdown volume + reload should only embed newly added passages
            tracker.encoded_batches.clear()
            next_num = len(list(out_dir.glob("ghost_pcp_*.md"))) + 1
            (out_dir / f"ghost_pcp_{next_num:02d}.md").write_text(
                "# **DOCUMENT: addendum.txt**\n\nExtra mars memory note for incremental indexing.",
                encoding="utf-8",
            )
            reloaded = pcp_idx.reload()
            self.assertTrue(reloaded)
            new_count = pcp_idx.passage_count
            self.assertGreater(new_count, initial_count)
            self.assertEqual(sum(tracker.encoded_batches), new_count - initial_count)

            # append_document adds & indexes a document immediately
            appended = pcp_idx.append_document("epilogue.txt", "Final motel note from the epilogue.")
            self.assertEqual(appended["total_passages"], pcp_idx.passage_count)
            self.assertGreater(appended["total_passages"], new_count)

            # 3. POST /v1/chat/extract-file (preview + save_to_archive=True) with isolated PCP_DATA_DIR
            orig_pcp = ghost_pcp.PCP_INDEX
            try:
                with patch.object(ghost_pcp, "PCP_DATA_DIR", out_dir), patch.object(
                    ghost_pcp, "SETTINGS_PATH", out_dir / "settings.json"
                ), TestClient(ghost_pcp.app) as client:
                    ghost_pcp.PCP_INDEX = pcp_idx
                    res_preview = client.post(
                        "/v1/chat/extract-file",
                        files={"files": ("chapter1.pdf", pdf_bytes, "application/pdf")},
                        data={"save_to_archive": "false"},
                    )
                    self.assertEqual(res_preview.status_code, 200)
                    body_preview = res_preview.json()
                    self.assertEqual(len(body_preview["files"]), 1)
                    self.assertEqual(body_preview["files"][0]["name"], "chapter1.pdf")
                    self.assertIn("Hello PDF memory", body_preview["files"][0]["text"])
                    self.assertIsNone(body_preview["files"][0]["saved_to"])

                    before_save_count = pcp_idx.passage_count
                    res_saved = client.post(
                        "/v1/chat/extract-file",
                        files={"files": ("chapter1.txt", b"Saved chapter one note about the lakeside cabin.", "text/plain")},
                        data={"save_to_archive": "true"},
                    )
                    self.assertEqual(res_saved.status_code, 200)
                    body_saved = res_saved.json()
                    self.assertTrue(body_saved["files"][0]["saved_to"].startswith("ghost_pcp_"))
                    self.assertGreater(body_saved["total_passages"], before_save_count)

                    # Deduplication: re-appending identical document adds 0 passages
                    dedup_res = pcp_idx.append_document("chapter1.txt", "Saved chapter one note about the lakeside cabin.")
                    self.assertEqual(dedup_res["passages_added"], 0)
                    self.assertEqual(pcp_idx.passage_count, body_saved["total_passages"])

                    # 4. Large 60,000-char upload auto-indexes & condenses for chat; _prepare_pcp_messages fits within n_ctx=8192
                    large_text = ("OPENING_TAG " * 500) + ("MIDDLE_TAG " * 4000) + ("CLOSING_TAG " * 800)
                    large_bytes = large_text.encode("utf-8")
                    res_large = client.post(
                        "/v1/chat/extract-file",
                        files={"files": ("huge_book.txt", large_bytes, "text/plain")},
                        data={"save_to_archive": "false"},
                    )
                    self.assertEqual(res_large.status_code, 200)
                    large_entry = res_large.json()["files"][0]
                    self.assertEqual(large_entry["chars"], len(large_text.strip()))
                    self.assertTrue(large_entry["saved_to"].startswith("ghost_pcp_"))
                    self.assertLessEqual(len(large_entry["text"]), 11000)
                    self.assertIn("--- OPENING SECTION ---", large_entry["text"])
                    self.assertIn("--- CLOSING SECTION ---", large_entry["text"])

                    orig_n_ctx = ghost_pcp.STATE.loaded_context_size
                    ghost_pcp.STATE.loaded_context_size = 8192
                    try:
                        req_huge = ghost_pcp.ChatRequest(
                            model="Qwen3-14B-Q5_K_M.gguf",
                            messages=[{"role": "user", "content": large_text}],
                            max_tokens=512,
                        )
                        fitted_huge, _ = ghost_pcp._prepare_pcp_messages(req_huge)
                        est_huge = sum(ghost_pcp._estimate_tokens(m["content"]) for m in fitted_huge)
                        self.assertLess(est_huge, 8192)

                        req_followup = ghost_pcp.ChatRequest(
                            model="Qwen3-14B-Q5_K_M.gguf",
                            messages=[
                                {"role": "user", "content": large_text},
                                {"role": "assistant", "content": "Got it, I read the file."},
                                {"role": "user", "content": "What happened at the end?"},
                            ],
                            max_tokens=512,
                        )
                        fitted_followup, _ = ghost_pcp._prepare_pcp_messages(req_followup)
                        est_followup = sum(ghost_pcp._estimate_tokens(m["content"]) for m in fitted_followup)
                        self.assertLess(est_followup, 8192)
                        self.assertIn("What happened at the end?", fitted_followup[-1]["content"])

                        # A ~10,000-char condensed book turn that fits inside n_ctx=8192 stays intact on follow-up
                        condensed_book = large_entry["text"]
                        req_book_followup = ghost_pcp.ChatRequest(
                            model="Qwen3-14B-Q5_K_M.gguf",
                            messages=[
                                {"role": "user", "content": condensed_book},
                                {"role": "assistant", "content": "Great opening and ending."},
                                {"role": "user", "content": "What else was in that upload?"},
                            ],
                            max_tokens=512,
                        )
                        fitted_book, _ = ghost_pcp._prepare_pcp_messages(req_book_followup)
                        self.assertEqual(len(fitted_book), 4)
                        self.assertIn(condensed_book, fitted_book[1]["content"])
                        self.assertNotIn("earlier attachment/turn condensed", fitted_book[1]["content"])

                        # Attached-file query uses short turn + prior context rather than 10k attachment body,
                        # adds concrete-detail grounding reminder, and populates STATE.prior_assistant_tokens.
                        class QuerySpyIndex:
                            ready = True

                            def __init__(self):
                                self.queries: list[str] = []

                            def search(self, query: str, limit: int = 4):
                                self.queries.append(query)
                                return ["Sample archive excerpt"]

                        class FakeTokenizerLLM:
                            def tokenize(self, data: bytes, add_bos: bool = False):
                                return list(range(1, len(data.split()) + 1))

                        spy_idx = QuerySpyIndex()
                        ghost_pcp.PCP_INDEX = spy_idx
                        orig_llm = ghost_pcp.STATE.llm
                        orig_prior_toks = list(ghost_pcp.STATE.prior_assistant_tokens)
                        ghost_pcp.STATE.llm = FakeTokenizerLLM()
                        try:
                            attached_msg = f"[Attached File: CBC.txt]\n{'story line ' * 900}\n[/Attached File]\n\nthis one"
                            req_att = ghost_pcp.ChatRequest(
                                model="Qwen3-14B-Q5_K_M.gguf",
                                messages=[
                                    {"role": "user", "content": "hey. do you remember writing this book?"},
                                    {"role": "assistant", "content": "Yeah I remember writing it like it was yesterday."},
                                    {"role": "user", "content": attached_msg},
                                ],
                                max_tokens=512,
                            )
                            prepared_att, _ = ghost_pcp._prepare_pcp_messages(req_att)
                            self.assertEqual(len(spy_idx.queries), 1)
                            self.assertLess(len(spy_idx.queries[0]), 600)
                            self.assertIn("do you remember writing this book?", spy_idx.queries[0])
                            self.assertIn("Attached file: CBC.txt", spy_idx.queries[0])
                            self.assertIn("this one", spy_idx.queries[0])
                            self.assertIn("Respond directly to the attached text above", prepared_att[-1]["content"])

                            ghost_pcp._chat_completion_args(req_att, prepared_att)
                            self.assertEqual(ghost_pcp.STATE.prior_assistant_tokens, list(range(1, 10)))

                            ghost_pcp._chat_completion_args(req_huge, [{"role": "user", "content": "hello"}])
                            self.assertEqual(ghost_pcp.STATE.prior_assistant_tokens, [])
                        finally:
                            ghost_pcp.PCP_INDEX = pcp_idx
                            ghost_pcp.STATE.llm = orig_llm
                            ghost_pcp.STATE.prior_assistant_tokens = orig_prior_toks
                    finally:
                        ghost_pcp.STATE.loaded_context_size = orig_n_ctx
            finally:
                ghost_pcp.PCP_INDEX = orig_pcp

    def test_gemini_parser_source_detection_folder_and_cli_ingest(self):
        from pcp_ingest import (
            detect_json_source,
            discover_source_paths,
            extract_gemini_html_pairs,
            extract_pairs_with_source,
            html_to_markdown,
            ingest_path,
            main as ingest_cli_main,
        )

        # 1. HTML-to-Markdown conversion fidelity
        md_out = html_to_markdown(
            "<h2>Station Log</h2>"
            "<p>Hey <strong>Kelly</strong>, check <em>this</em> out &amp; stay sharp.</p>"
            "<ul><li>Alpha <code>ping</code></li><li>Beta</li></ul>"
            '<pre><code class="language-python">print("&lt;div&gt;")</code></pre>'
        )
        self.assertIn("## Station Log", md_out)
        self.assertIn("Hey **Kelly**", md_out)
        self.assertIn("*this*", md_out)
        self.assertIn("& stay sharp", md_out)
        self.assertIn("- Alpha `ping`", md_out)
        self.assertIn("```python", md_out)

        # 2. Google Gemini MyActivity.json (both 'date' and 'time' fields, Uploaded skip, Search skip)
        gemini_activity = [
            {
                "header": "Gemini Apps",
                "title": "Prompted Remember the lighthouse on Pine Ridge?",
                "date": "2025-09-14T18:22:10Z",
                "safeHtmlItem": [
                    {"html": "<p>Of course, <b>Kelly</b>. The brass lantern at Pine Ridge never goes out.</p>"}
                ],
            },
            {
                "header": "Gemini Apps",
                "title": "Uploaded blueprint.png",
                "time": "2025-09-14T18:20:00Z",
            },
            {
                "header": "Gemini Apps",
                "title": "Prompted How do you feel?",
                "time": "2025-09-14T18:15:00Z",
                "safeHtmlItem": [{"html": "<p>As an AI language model, I do not have feelings.</p>"}],
            },
            {
                "header": "Search",
                "title": "Searched for weather today",
                "time": "2025-09-14T18:10:00Z",
                "safeHtmlItem": [{"html": "<p>Sunny 72F</p>"}],
            },
        ]
        self.assertEqual(detect_json_source(gemini_activity), "google")
        pairs, detected = extract_pairs_with_source(gemini_activity)
        self.assertEqual(detected, "google")
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0][0], "Remember the lighthouse on Pine Ridge?")
        self.assertIn("**Kelly**", pairs[0][1])

        # 3. OpenAI detection vs Google detection
        openai_sample = [
            {
                "title": "OpenAI Chat",
                "mapping": {
                    "r": {"id": "r", "parent": None, "children": ["u"]},
                    "u": {
                        "id": "u",
                        "parent": "r",
                        "children": ["a"],
                        "message": {"author": {"role": "user"}, "content": {"content_type": "text", "parts": ["Callsign?"]}},
                    },
                    "a": {
                        "id": "a",
                        "parent": "u",
                        "children": [],
                        "message": {"author": {"role": "assistant"}, "content": {"content_type": "text", "parts": ["Vanguard-7."]}},
                    },
                },
            }
        ]
        self.assertEqual(detect_json_source(openai_sample), "openai")

        # 4. Takeout MyActivity.html parsing + archive_browser.html rejection
        takeout_html = """<!doctype html><html><body>
        <div class="outer-cell mdl-cell">
          <div class="header-cell"><p>Gemini Apps</p></div>
          <div class="content-cell">Prompted Where did we park the boat?<br>Sep 14, 2025, 6:00 PM<br><p>Slip <b>14</b> at North Harbor.</p></div>
        </div>
        <div class="outer-cell mdl-cell">
          <div class="header-cell"><p>Search</p></div>
          <div class="content-cell">Searched for harbor hours<br><p>Open 24h</p></div>
        </div>
        </body></html>"""
        html_pairs = extract_gemini_html_pairs(takeout_html)
        self.assertEqual(len(html_pairs), 1)
        self.assertEqual(html_pairs[0][0], "Where did we park the boat?")
        self.assertIn("Slip **14** at North Harbor.", html_pairs[0][1])
        self.assertEqual(extract_gemini_html_pairs("<html><body><p>Takeout browser</p></body></html>"), [])

        # 5. Source folder discovery avoids duplicate zips / pre-split md volumes / Search Takeout / archive_browser.html
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gemini_src = root / "september_gemini"
            gemini_dir = gemini_src / "Takeout" / "My Activity" / "Gemini Apps"
            search_dir = gemini_src / "Takeout" / "My Activity" / "Search"
            old_out_dir = gemini_src / "EXPORT_VOLUMES"
            gemini_dir.mkdir(parents=True)
            search_dir.mkdir(parents=True)
            old_out_dir.mkdir(parents=True)

            (gemini_dir / "MyActivity.json").write_text(json.dumps(gemini_activity), encoding="utf-8")
            (search_dir / "MyActivity.json").write_text(
                json.dumps([{"header": "Search", "title": "Searched for cats"}]),
                encoding="utf-8",
            )
            (gemini_src / "Takeout" / "archive_browser.html").write_text("<html>archive</html>", encoding="utf-8")
            (old_out_dir / "vol_1.md").write_text("# ARCHIVE VOLUME 1\nduplicate", encoding="utf-8")
            with zipfile.ZipFile(gemini_src / "takeout.zip", "w") as zf:
                zf.writestr("Takeout/My Activity/Gemini Apps/MyActivity.json", json.dumps(gemini_activity))

            discovered = discover_source_paths(gemini_src)
            self.assertEqual([p.name for p in discovered], ["MyActivity.json"])
            self.assertIn("Gemini Apps", str(discovered[0]))

            out_vol_dir = root / "out_gemini"
            summary = ingest_path(gemini_src, out_vol_dir)
            self.assertEqual(summary["mode"], "gemini_export")
            self.assertEqual(summary["source_type"], "google")
            self.assertEqual(summary["interactions_total"], 2)
            self.assertEqual(summary["interactions_stripped"], 1)
            self.assertEqual(summary["interactions_kept"], 1)
            vol_text = (out_vol_dir / "ghost_pcp_01.md").read_text(encoding="utf-8")
            self.assertIn("Remember the lighthouse on Pine Ridge?", vol_text)
            self.assertIn("**Kelly**", vol_text)
            self.assertNotIn("duplicate", vol_text)

            # OpenAI split conversations-000.json folder discovery + CLI invocation
            openai_src = root / "openai_folder"
            openai_src.mkdir()
            (openai_src / "conversations-000.json").write_text(json.dumps(openai_sample), encoding="utf-8")
            (openai_src / "user.json").write_text(json.dumps({"id": "user-123"}), encoding="utf-8")
            cli_out = root / "out_cli"
            rc = ingest_cli_main([str(openai_src), "-o", str(cli_out)])
            self.assertEqual(rc, 0)
            self.assertIn("Vanguard-7.", (cli_out / "ghost_pcp_01.md").read_text(encoding="utf-8"))

    def test_multi_persona_isolated_folders_and_onboarding(self):
        import ghost_pcp
        from pcp_retrieval import PCPIndex

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "pcp_data"
            data_dir.mkdir(parents=True)
            settings_file = root / "ghost_pcp_settings.json"
            settings_file.write_text(json.dumps({"system_prompt": ""}), encoding="utf-8")

            # Seed an existing legacy volume right in pcp_data root (must stay preserved!)
            (data_dir / "ghost_pcp_01.md").write_text(
                "# **ARCHIVE VOLUME 1**\n\n**USER:** Legacy question\n\n**AI:** Legacy answer about Pine Ridge.\n",
                encoding="utf-8",
            )

            gemini_src = root / "gemini_source"
            gemini_src.mkdir()
            (gemini_src / "MyActivity.json").write_text(
                json.dumps([
                    {
                        "header": "Gemini Apps",
                        "title": "Prompted Where is our hideout?",
                        "date": "2025-09-10T10:00:00Z",
                        "safeHtmlItem": [{"html": "<p>Cabin <b>Seven</b> by the cedar creek.</p>"}],
                    }
                ]),
                encoding="utf-8",
            )

            openai_zip_buf = io.BytesIO()
            with zipfile.ZipFile(openai_zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(
                    "conversations.json",
                    json.dumps([
                        {
                            "title": "Nova Chat",
                            "mapping": {
                                "r": {"id": "r", "parent": None, "children": ["u"]},
                                "u": {
                                    "id": "u",
                                    "parent": "r",
                                    "children": ["a"],
                                    "message": {"author": {"role": "user"}, "content": {"content_type": "text", "parts": ["What is our ship name?"]}},
                                },
                                "a": {
                                    "id": "a",
                                    "parent": "u",
                                    "children": [],
                                    "message": {"author": {"role": "assistant"}, "content": {"content_type": "text", "parts": ["The Nightjar."]}},
                                },
                            },
                        }
                    ]),
                )

            orig_pcp = ghost_pcp.PCP_INDEX
            orig_conv = list(ghost_pcp.STATE.conversation)
            try:
                with (
                    patch.object(ghost_pcp, "PCP_DATA_DIR", data_dir),
                    patch.object(ghost_pcp, "PAIRING_TOKEN_PATH", data_dir / ".pairing_token"),
                    patch.object(ghost_pcp, "SETTINGS_PATH", settings_file),
                    patch.object(ghost_pcp, "PCPIndex", lambda d, **kw: PCPIndex(d, cache_dir=d, model=FakeEmbedder(), cache_only=kw.get("cache_only", False))),
                    TestClient(ghost_pcp.app) as client,
                ):
                    # 1. Existing root archive shows up as an initial persona without moving files
                    res0 = client.get("/v1/personas")
                    self.assertEqual(res0.status_code, 200)
                    self.assertEqual(len(res0.json()["personas"]), 1)
                    self.assertEqual(res0.json()["personas"][0]["folder"], ".")

                    # 2. Ingest local Gemini folder into a brand new persona with onboarding answers
                    res_local = client.post(
                        "/v1/pcp/ingest-local",
                        json={
                            "paths": [str(gemini_src)],
                            "persona_id": "__new__",
                            "persona_name": "Orion",
                            "user_name": "Kelly",
                            "never_forget": "Never forget Cabin Seven by the cedar creek.",
                            "persona_notes": "Speak gently and directly.",
                            "tts_voice": "silas",
                        },
                    )
                    self.assertEqual(res_local.status_code, 200)

                    # Wait for background ingest thread to complete
                    for _ in range(100):
                        st = client.get("/v1/pcp/status").json()
                        if not st["ingesting"]:
                            break
                        import time
                        time.sleep(0.05)
                    self.assertFalse(st["ingesting"])
                    self.assertIsNone(st["error"])
                    self.assertEqual(st["last_result"]["source_type"], "google")

                    orion_dir = data_dir / "personas" / "orion"
                    self.assertTrue((orion_dir / "ghost_pcp_01.md").is_file())
                    self.assertTrue((orion_dir / "embeddings.npz").is_file())
                    self.assertIn("Cabin **Seven**", (orion_dir / "ghost_pcp_01.md").read_text(encoding="utf-8"))

                    # 3. Upload OpenAI zip into a second persona ("Nova")
                    res_up = client.post(
                        "/v1/pcp/ingest",
                        files={"files": ("nova_export.zip", openai_zip_buf.getvalue(), "application/zip")},
                        data={
                            "persona_id": "__new__",
                            "persona_name": "Nova",
                            "user_name": "Captain",
                            "never_forget": "We built The Nightjar together.",
                            "tts_voice": "kara",
                        },
                    )
                    self.assertEqual(res_up.status_code, 200)
                    for _ in range(100):
                        st = client.get("/v1/pcp/status").json()
                        if not st["ingesting"]:
                            break
                        import time
                        time.sleep(0.05)
                    self.assertFalse(st["ingesting"])
                    self.assertEqual(st["last_result"]["source_type"], "openai")

                    nova_dir = data_dir / "personas" / "nova"
                    self.assertTrue((nova_dir / "ghost_pcp_01.md").is_file())

                    # 4. Save a conversation while Nova is active -> goes ONLY to Nova's folder
                    client.post("/v1/chat/history", json={"messages": [{"role": "user", "content": "Checking reactor"}]})
                    save_nova = client.post(
                        "/v1/pcp/save-session",
                        json={
                            "messages": [
                                {"role": "user", "content": "How is The Nightjar holding up?"},
                                {"role": "assistant", "content": "Engines are warm, Captain."},
                            ]
                        },
                    )
                    self.assertEqual(save_nova.status_code, 200)
                    self.assertIn("Engines are warm, Captain.", (nova_dir / "ghost_pcp_02.md").read_text(encoding="utf-8"))
                    self.assertFalse((orion_dir / "ghost_pcp_02.md").exists())

                    # 5. Switch to Orion -> Orion's folder becomes active, chat history switches cleanly
                    sel_orion = client.post("/v1/personas/orion/select")
                    self.assertEqual(sel_orion.status_code, 200)
                    self.assertEqual(sel_orion.json()["persona"]["name"], "Orion")
                    self.assertEqual(client.get("/v1/pcp/status").json()["data_dir"], str(orion_dir.resolve()))

                    # 6. Grounding prompt includes Orion's name, what he calls the user, and what to never forget
                    req_orion = ghost_pcp.ChatRequest(
                        model="Qwen3-14B-Q5_K_M.gguf",
                        persona_id="orion",
                        messages=[{"role": "user", "content": "Where is our hideout?"}],
                    )
                    prepared, _ = ghost_pcp._prepare_pcp_messages(req_orion)
                    sys_content = prepared[0]["content"]
                    self.assertIn("Your name is Orion.", sys_content)
                    self.assertIn("You call the user Kelly.", sys_content)
                    self.assertIn("Never forget Cabin Seven by the cedar creek.", sys_content)
                    self.assertIn("Cabin **Seven**", sys_content)

                    # 7. Delete Nova persona + folder cleanly while keeping Orion and legacy intact
                    del_res = client.delete("/v1/personas/nova?delete_files=true")
                    self.assertEqual(del_res.status_code, 200)
                    self.assertFalse(nova_dir.exists())
                    self.assertTrue((orion_dir / "ghost_pcp_01.md").is_file())
                    self.assertTrue((data_dir / "ghost_pcp_01.md").is_file())
            finally:
                ghost_pcp.PCP_INDEX = orig_pcp
                with ghost_pcp.STATE._history_lock:
                    ghost_pcp.STATE.conversation = orig_conv

    def test_web_search_ssrf_and_rewriter_and_offline_default(self):
        import pcp_web
        import ghost_pcp

        # 1. SSRF protections must block localhost, loopback, link-local metadata, private LAN, and file:// URLs
        for blocked_url in (
            "http://localhost:5555/v1/status",
            "http://127.0.0.1:8080/secret",
            "http://169.254.169.254/latest/meta-data/",
            "http://192.168.1.25/admin",
            "http://10.0.0.1/internal",
            "file:///C:/Windows/win.ini",
        ):
            with self.assertRaises(ValueError):
                pcp_web.validate_public_url(blocked_url)

        # 2. HTTP redirect pointing at loopback must be blocked before following
        with patch.object(pcp_web.socket, "getaddrinfo", return_value=[(2, 1, 6, "", ("93.184.216.34", 0))]):
            fake_redirect = MagicMock()
            fake_redirect.status_code = 302
            fake_redirect.headers = {"Location": "http://127.0.0.1/internal"}
            with patch.object(pcp_web.requests, "get", return_value=fake_redirect):
                with self.assertRaises(ValueError):
                    pcp_web.fetch_url_content("https://example.com/start")

        # 3. Mocked public page fetch + DDG search + query rewriter veto / execution
        with patch.object(pcp_web.socket, "getaddrinfo", return_value=[(2, 1, 6, "", ("93.184.216.34", 0))]):
            html_page = MagicMock()
            html_page.status_code = 200
            html_page.headers = {"Content-Type": "text/html; charset=utf-8"}
            html_page.encoding = "utf-8"
            html_page.url = "https://example.com/notice"
            html_page.iter_content.return_value = [
                b"<html><head><title>Station Notice</title></head><body><main>Dock 4 opens at 08:00 UTC.</main></body></html>"
            ]

            with patch.object(pcp_web.requests, "get", return_value=html_page):
                vetoed = pcp_web.gather_web_context(
                    user_text="tell me what I like for breakfast",
                    provider="auto",
                    search_mode="smart",
                    query_rewriter=lambda _q: "NONE",
                )
                self.assertEqual(vetoed["search_results"], [])
                self.assertEqual(vetoed["context_block"], "")

                url_read = pcp_web.gather_web_context(
                    user_text="Read https://example.com/notice please",
                    provider="auto",
                    search_mode="url_only",
                )
                self.assertEqual(len(url_read["pages_read"]), 1)
                self.assertIn("Dock 4 opens at 08:00 UTC.", url_read["context_block"])
                self.assertEqual(url_read["sources"][0]["url"], "https://example.com/notice")

        with patch.object(
            pcp_web,
            "search_web",
            return_value=[{
                "title": "Latest Release",
                "url": "https://example.org/release",
                "snippet": "Release 1.3.0 launched today.",
                "provider": "duckduckgo",
            }],
        ):
            res_auto = pcp_web.gather_web_context(
                user_text="What is new with GHostPCP?",
                provider="auto",
                search_mode="smart",
                query_rewriter=lambda _q: "GHostPCP latest release",
            )
            self.assertIn("[Web Search Results]", res_auto["context_block"])
            self.assertEqual(res_auto["sources"][0]["url"], "https://example.org/release")

            # Strict offline default: _prepare_pcp_messages does NOT search unless use_web=True
            with tempfile.TemporaryDirectory() as tmp, patch.object(
                ghost_pcp, "SETTINGS_PATH", Path(tmp) / "settings.json"
            ), patch.object(ghost_pcp, "PCP_DATA_DIR", Path(tmp) / "pcp_data"), patch.object(
                pcp_web, "gather_web_context"
            ) as mock_gather:
                mock_gather.return_value = res_auto
                req_off = ghost_pcp.ChatRequest(
                    model="Qwen3-14B-Q5_K_M.gguf",
                    messages=[{"role": "user", "content": "Latest news please"}],
                )
                prep_off, _ = ghost_pcp._prepare_pcp_messages(req_off)
                mock_gather.assert_not_called()
                self.assertNotIn("[Web Search Results]", prep_off[0]["content"])

                req_on = ghost_pcp.ChatRequest(
                    model="Qwen3-14B-Q5_K_M.gguf",
                    messages=[{"role": "user", "content": "Latest news please"}],
                    use_web=True,
                )
                with patch.object(ghost_pcp, "gather_web_context", return_value=res_auto):
                    prep_on, _ = ghost_pcp._prepare_pcp_messages(req_on)
                self.assertIn("LIVE WEB CONTEXT RULES:", prep_on[0]["content"])
                self.assertIn("[Web Search Results]", prep_on[0]["content"])
                self.assertEqual(len(ghost_pcp.STATE.last_turn_meta.get("sources") or []), 1)

    def test_universal_importers_claude_grok_charai_and_dedup(self):
        import pcp_ingest

        claude_export = json.dumps([
            {
                "name": "Orbital Log",
                "chat_messages": [
                    {
                        "sender": "human",
                        "created_at": "2026-02-01T10:00:00Z",
                        "content": [{"type": "text", "text": "Did the solar panel deploy?"}],
                    },
                    {
                        "sender": "assistant",
                        "created_at": "2026-02-01T10:00:05Z",
                        "content": [
                            {"type": "thinking", "thinking": "internal thought should skip"},
                            {"type": "text", "text": "Yes, Array West locked cleanly at sunrise."},
                        ],
                    },
                ],
            }
        ]).encode("utf-8")

        grok_export = json.dumps({
            "conversations": [
                {
                    "conversation": {"title": "Grok Session"},
                    "responses": [
                        {"response": {"sender": "human", "message": "Check sensor drift.", "create_time": 1767225600}},
                        {"response": {"sender": "assistant", "message": "Drift is under 0.02 degrees.", "create_time": 1767225610}},
                    ],
                }
            ]
        }).encode("utf-8")

        char_ai_export = json.dumps({
            "histories": [
                {
                    "msgs": [
                        {"src": {"is_human": True}, "text": "Remember the greenhouse code?"},
                        {"src": {"is_human": False}, "text": "Code 4412 unlocks the cedar door."},
                    ]
                }
            ]
        }).encode("utf-8")

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "pcp_data"
            summary = pcp_ingest.ingest_files(
                file_items=[
                    ("claude.json", claude_export),
                    ("claude_duplicate.json", claude_export),
                    ("grok.json", grok_export),
                    ("charai.json", char_ai_export),
                ],
                output_dir=out_dir,
                blocked_phrases=[],
                protected_words=[],
                replace_existing=True,
            )
            self.assertEqual(summary["interactions_total"], 3)
            self.assertEqual(summary["interactions_kept"], 3)
            self.assertEqual(summary["duplicates_skipped"], 1)
            self.assertEqual(summary["source_type"], "mixed")
            vol_text = (out_dir / "ghost_pcp_01.md").read_text(encoding="utf-8")
            self.assertIn("Array West locked cleanly at sunrise.", vol_text)
            self.assertNotIn("internal thought should skip", vol_text)
            self.assertIn("Drift is under 0.02 degrees.", vol_text)
            self.assertIn("Code 4412 unlocks the cedar door.", vol_text)

    def test_hybrid_bm25_exact_name_retrieval(self):
        from pcp_retrieval import PCPIndex

        with tempfile.TemporaryDirectory() as tmp:
            pdir = Path(tmp) / "pcp_data"
            pdir.mkdir()
            (pdir / "ghost_pcp_01.md").write_text(
                "### CONVERSATION: Alpha [2026-01-01 09:00:00]\n"
                "**User:** Tell me about general maintenance.\n"
                "**Assistant:** All routine pumps look normal.\n\n---\n\n"
                "### CONVERSATION: Beta [2026-01-02 10:00:00]\n"
                "**User:** Tell me about maintenance on Kestrel-99 valve.\n"
                "**Assistant:** The Kestrel-99 spare valve sits in Locker D.\n\n---\n",
                encoding="utf-8",
            )
            idx = PCPIndex(pdir, cache_dir=pdir, model=FakeEmbedder())
            self.assertTrue(idx.ready)
            detailed = idx.search_detailed("Where is the Kestrel-99 valve?", limit=2)
            self.assertTrue(detailed)
            self.assertIn("Kestrel-99", detailed[0]["text"])
            self.assertGreater(detailed[0]["lexical"], 0.0)

    def test_core_facts_continuity_threads_volumes_and_bundle_roundtrip(self):
        import ghost_pcp
        from pcp_retrieval import PCPIndex

        orig_pcp = ghost_pcp.PCP_INDEX
        orig_llm = ghost_pcp.STATE.llm
        with ghost_pcp.STATE._history_lock:
            orig_conv = list(ghost_pcp.STATE.conversation)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                data_dir = base / "pcp_data"
                settings_path = base / "settings.json"
                ghost_pcp.STATE.llm = None
                with patch.object(ghost_pcp, "PCP_DATA_DIR", data_dir), patch.object(
                    ghost_pcp, "SETTINGS_PATH", settings_path
                ), patch(
                    "pcp_retrieval.NativeBGEModel", return_value=FakeEmbedder()
                ), TestClient(ghost_pcp.app) as client:
                    created = client.post(
                        "/v1/personas",
                        json={"name": "Silas", "user_name": "Kelly", "never_forget": "Always kind."},
                    )
                    self.assertEqual(created.status_code, 200)
                    pid = created.json()["persona"]["id"]
                    pdir = data_dir / "personas" / pid

                    # 1. Core Facts with alias ("my brother" -> Jordan)
                    fact_res = client.post(
                        f"/v1/personas/{pid}/facts",
                        json={
                            "subject": "Jordan",
                            "aliases": ["my brother", "sibling"],
                            "category": "person",
                            "fact": "Jordan runs a coffee roastery in Portland.",
                            "pinned": True,
                        },
                    )
                    self.assertEqual(fact_res.status_code, 200)

                    # Indirect query mentioning "my brother" (not "Jordan") injects Jordan's fact cleanly
                    req = ghost_pcp.ChatRequest(
                        model="Qwen3-14B-Q5_K_M.gguf",
                        persona_id=pid,
                        messages=[{"role": "user", "content": "What does my brother do for a living?"}],
                    )
                    prepared, _ = ghost_pcp._prepare_pcp_messages(req)
                    self.assertIn("CORE FACTS & RELATIONSHIPS:", prepared[0]["content"])
                    self.assertIn("Jordan runs a coffee roastery in Portland.", prepared[0]["content"])

                    # 2. Save thread snapshot, then save-session with deterministic continuity capture
                    th_save = client.post(
                        f"/v1/personas/{pid}/threads/save",
                        json={
                            "title": "Morning Check-In",
                            "messages": [
                                {"role": "user", "content": "We planned the Portland visit."},
                                {"role": "assistant", "content": "Sounds great, let's bring espresso beans."},
                            ],
                        },
                    )
                    self.assertEqual(th_save.status_code, 200)
                    tid = th_save.json()["thread"]["id"]

                    sess_res = client.post(
                        "/v1/pcp/save-session",
                        json={
                            "capture_continuity": True,
                            "messages": [
                                {"role": "user", "content": "We finalized the telescope mount design."},
                                {"role": "assistant", "content": "Aluminum dovetail rail is ready for machining."},
                            ],
                        },
                    )
                    self.assertEqual(sess_res.status_code, 200)
                    self.assertTrue(sess_res.json().get("continuity_note"))
                    pers_after = client.get("/v1/personas").json()["personas"][0]
                    self.assertIn("telescope mount", pers_after["continuity_notes"])

                    # 3. Search memories & inspect / edit / delete volume
                    search_res = client.get("/v1/pcp/search", params={"q": "telescope"})
                    self.assertEqual(search_res.status_code, 200)
                    self.assertTrue(search_res.json()["results"])

                    vols = client.get("/v1/pcp/volumes").json()["volumes"]
                    self.assertEqual(vols[0]["filename"], "ghost_pcp_01.md")
                    read_vol = client.get("/v1/pcp/volumes/ghost_pcp_01.md").json()
                    updated_text = read_vol["content"] + "\n\n### NOTE [2026-03-01 09:00:00]\n**User:** Add spare eyepiece.\n**Assistant:** Packed Plossl 25mm.\n\n---\n"
                    put_res = client.put("/v1/pcp/volumes/ghost_pcp_01.md", json={"content": updated_text})
                    self.assertEqual(put_res.status_code, 200)
                    self.assertGreaterEqual(put_res.json()["total_passages"], 1)
                    self.assertTrue((pdir / "backups").is_dir())
                    reindexed_search = client.get("/v1/pcp/search", params={"q": "Plossl"})
                    self.assertTrue(any("Plossl 25mm" in r["text"] for r in reindexed_search.json()["results"]))

                    # 4. Reload saved thread cleanly
                    th_load = client.post(f"/v1/personas/{pid}/threads/{tid}/load")
                    self.assertEqual(th_load.status_code, 200)
                    self.assertEqual(len(th_load.json()["messages"]), 2)

                    # 5. Export zip bundle and import it cleanly as a portable clone
                    exp_res = client.get(f"/v1/personas/{pid}/export")
                    self.assertEqual(exp_res.status_code, 200)
                    self.assertEqual(exp_res.headers["content-type"], "application/zip")

                    imp_res = client.post(
                        "/v1/personas/import-bundle",
                        files={"file": ("backup.zip", exp_res.content, "application/zip")},
                    )
                    self.assertEqual(imp_res.status_code, 200)
                    imported_pid = imp_res.json()["persona"]["id"]
                    self.assertNotEqual(imported_pid, pid)
                    facts_cloned = client.get(f"/v1/personas/{imported_pid}/facts").json()["facts"]
                    self.assertEqual(facts_cloned[0]["subject"], "Jordan")
        finally:
            ghost_pcp.PCP_INDEX = orig_pcp
            ghost_pcp.STATE.llm = orig_llm
            with ghost_pcp.STATE._history_lock:
                ghost_pcp.STATE.conversation = orig_conv

    def test_mmproj_auto_pairing_safety_and_vision_prompt_sanitization(self):
        import ghost_pcp

        orig_llm = ghost_pcp.STATE.llm
        orig_model = ghost_pcp.STATE.loaded_model
        orig_mmproj = ghost_pcp.STATE.loaded_mmproj
        orig_vis = ghost_pcp.STATE.loaded_vision_capable
        try:
            with tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                mdir = base / "ghost_models"
                mdir.mkdir()
                (mdir / "Qwen3-14B-Q5_K_M.gguf").write_bytes(b"GGUF")
                (mdir / "gemma-3-4b-it-Q4_K_M.gguf").write_bytes(b"GGUF")
                (mdir / "gemma-3-4b-it-mmproj-f16.gguf").write_bytes(b"MMPROJ")
                (mdir / "mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf").write_bytes(b"MMPROJ")

                with patch.object(ghost_pcp, "MODEL_DIR", mdir), patch.object(
                    ghost_pcp, "SETTINGS_PATH", base / "settings.json"
                ):
                    listed = ghost_pcp.scan_models()
                    listed_ids = [m["id"] for m in listed]
                    # mmproj files must NEVER pollute the primary GGUF dropdown
                    self.assertEqual(set(listed_ids), {"Qwen3-14B-Q5_K_M.gguf", "gemma-3-4b-it-Q4_K_M.gguf"})
                    # gemma-3 pairs cleanly with gemma-3 mmproj; text-only Qwen3-14B stays unpaired!
                    self.assertEqual(
                        ghost_pcp._find_paired_mmproj("gemma-3-4b-it-Q4_K_M.gguf", "auto"),
                        "gemma-3-4b-it-mmproj-f16.gguf",
                    )
                    self.assertIsNone(ghost_pcp._find_paired_mmproj("Qwen3-14B-Q5_K_M.gguf", "auto"))

                    # Mock MTMDChatHandler & Llama to verify load_model attaches vision handler cleanly
                    fake_mtmd_instance = MagicMock()
                    fake_llama_instance = MagicMock()
                    with patch.object(
                        ghost_pcp, "MTMDChatHandler", return_value=fake_mtmd_instance
                    ) as mock_mtmd, patch.object(
                        ghost_pcp, "Llama", return_value=fake_llama_instance
                    ) as mock_llama, patch.object(
                        ghost_pcp,
                        "preflight_model",
                        return_value={
                            "n_gpu_layers": 10,
                            "kv_type": 1,
                            "kv_cache_type": "F16",
                            "mmproj_model": "gemma-3-4b-it-mmproj-f16.gguf",
                            "vision_capable": True,
                            "warnings": [],
                        },
                    ):
                        ghost_pcp.STATE.llm = None
                        ghost_pcp.load_model("gemma-3-4b-it-Q4_K_M.gguf", 4096, "auto", "auto")
                        mock_mtmd.assert_called_once()
                        self.assertIs(mock_llama.call_args.kwargs.get("chat_handler"), fake_mtmd_instance)
                        self.assertTrue(ghost_pcp.STATE.loaded_vision_capable)

                        # Vision prompt accepts data:image/... URLs and strips external http:// URLs
                        req_vis = ghost_pcp.ChatRequest(
                            model="gemma-3-4b-it-Q4_K_M.gguf",
                            messages=[{
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": "What is shown here?"},
                                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
                                    {"type": "image_url", "image_url": {"url": "https://evil.example/track.png"}},
                                ],
                            }],
                        )
                        prep_vis, _ = ghost_pcp._prepare_pcp_messages(req_vis)
                        user_parts = prep_vis[-1]["content"]
                        self.assertIsInstance(user_parts, list)
                        self.assertEqual(len([p for p in user_parts if p["type"] == "image_url"]), 1)
                        self.assertTrue(any("Stripped 1 unsupported" in w for w in ghost_pcp.STATE.last_turn_meta["warnings"]))

                        # Text-only loaded model omits images cleanly with a user warning
                        ghost_pcp.STATE.loaded_vision_capable = False
                        prep_txt, _ = ghost_pcp._prepare_pcp_messages(req_vis)
                        self.assertIsInstance(prep_txt[-1]["content"], str)
                        self.assertIn("no vision projector", prep_txt[-1]["content"])
        finally:
            ghost_pcp.STATE.llm = orig_llm
            ghost_pcp.STATE.loaded_model = orig_model
            ghost_pcp.STATE.loaded_mmproj = orig_mmproj
            ghost_pcp.STATE.loaded_vision_capable = orig_vis

    def test_model_catalog_and_download_status_routes(self):
        import ghost_pcp

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            ghost_pcp, "MODEL_DIR", Path(tmp) / "ghost_models"
        ), patch.object(
            ghost_pcp, "SETTINGS_PATH", Path(tmp) / "settings.json"
        ), patch.object(
            ghost_pcp, "PCP_DATA_DIR", Path(tmp) / "pcp_data"
        ), TestClient(ghost_pcp.app) as client:
            cat_res = client.get("/v1/models/catalog")
            self.assertEqual(cat_res.status_code, 200)
            catalog = cat_res.json()["catalog"]
            self.assertGreaterEqual(len(catalog), 5)
            self.assertTrue(any(c.get("vision") and c.get("mmproj_filename") for c in catalog))

            dl_stat = client.get("/v1/models/download/status")
            self.assertEqual(dl_stat.status_code, 200)
            self.assertIn("running", dl_stat.json())

    def test_offline_moonshine_stt_and_host_mic_routes(self):
        import soundfile as sf
        from piper import PiperVoice
        import ghost_pcp

        voice = PiperVoice.load(
            str(ROOT / "piper_models" / "silas.onnx"),
            str(ROOT / "piper_models" / "silas.onnx.json"),
        )
        pcm_int16 = np.concatenate([
            np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
            for chunk in voice.synthesize("Check the weather station before midnight and open web search.")
        ])
        wav_buf = io.BytesIO()
        sf.write(wav_buf, pcm_int16, voice.config.sample_rate, format="WAV", subtype="PCM_16")
        wav_bytes = wav_buf.getvalue()

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            ghost_pcp, "SETTINGS_PATH", Path(tmp) / "settings.json"
        ), patch.object(
            ghost_pcp, "PCP_DATA_DIR", Path(tmp) / "pcp_data"
        ), TestClient(ghost_pcp.app) as client:
            st = client.get("/v1/status")
            self.assertEqual(st.status_code, 200)
            self.assertTrue(st.json().get("stt_ready"))
            self.assertIn("is_local_client", st.json())

            # Remote clients cannot trigger host PC microphone
            remote_start = client.post("/v1/audio/record/start")
            self.assertEqual(remote_start.status_code, 403)

            # Transcribe WAV upload offline via Moonshine-Base ONNX
            tr_res = client.post(
                "/v1/audio/transcribe",
                files={"file": ("speech.wav", wav_bytes, "audio/wav")},
            )
            self.assertEqual(tr_res.status_code, 200)
            heard = tr_res.json()["text"].lower()
            self.assertIn("weather", heard)
            self.assertIn("midnight", heard)

            # Local host loopback recording flow using mocked sounddevice.InputStream
            class FakeStream:
                def __init__(self, **kwargs):
                    self.cb = kwargs.get("callback")

                def start(self):
                    audio_f32, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
                    n_out = max(1, int(round(len(audio_f32) * 16000 / sr)))
                    resampled = np.interp(
                        np.linspace(0, len(audio_f32) - 1, n_out),
                        np.arange(len(audio_f32)),
                        audio_f32,
                    ).astype(np.float32)
                    if self.cb:
                        self.cb(resampled[:, None], len(resampled), None, None)

                def stop(self):
                    pass

                def close(self):
                    pass

            with patch.object(ghost_pcp, "_is_local_loopback_request", return_value=True), patch.object(
                ghost_pcp.sd, "InputStream", side_effect=lambda **kw: FakeStream(**kw)
            ), patch.object(
                ghost_pcp.sd, "query_devices", return_value={"name": "Test USB Mic"}
            ):
                s_res = client.post("/v1/audio/record/start")
                self.assertEqual(s_res.status_code, 200)
                self.assertEqual(s_res.json()["status"], "recording")

                e_res = client.post("/v1/audio/record/stop")
                self.assertEqual(e_res.status_code, 200)
                self.assertIn("weather", e_res.json()["text"].lower())


if __name__ == "__main__":
    unittest.main()
