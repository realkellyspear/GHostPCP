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
from unittest.mock import patch

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
            bge_dir.mkdir()
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
                ("ghost_models", "README.md"),
            ):
                target = temp_root / directory / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(name, encoding="utf-8")
            (temp_root / "tests").mkdir()
            (temp_root / "tests" / "test_pcp.py").write_bytes(Path(__file__).read_bytes())

            with patch.object(package_release, "ROOT", temp_root), patch(
                "pcp_retrieval._resolve_bge_model_dir", return_value=bge_dir
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
            "renderChatAttachments",
            "saveAttachmentToArchive",
            "removeChatAttachment",
            "formatUserMessageHtml",
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
                with patch.object(ghost_pcp, "PCP_DATA_DIR", out_dir), TestClient(ghost_pcp.app) as client:
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
                with patch.object(ghost_pcp, "PCP_DATA_DIR", out_dir), TestClient(ghost_pcp.app) as client:
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


if __name__ == "__main__":
    unittest.main()
