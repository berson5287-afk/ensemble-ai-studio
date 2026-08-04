"""Download progress: totals across layers, rate, and what gets shown."""

from __future__ import annotations

from aichatlab.pulling import (
    Progress,
    human_bytes,
    human_eta,
    looks_like_model_name,
)

MB = 1024 * 1024


def test_progress_sums_every_layer_not_just_the_newest():
    """A model is several layers pulled in turn.  Reading only the newest
    line makes the bar leap back to zero each time a layer finishes."""
    progress = Progress()
    progress.update({"status": "pulling", "digest": "a",
                     "total": 100 * MB, "completed": 100 * MB}, now=0.0)
    progress.update({"status": "pulling", "digest": "b",
                     "total": 100 * MB, "completed": 10 * MB}, now=1.0)

    assert progress.completed == 110 * MB
    assert progress.total == 200 * MB
    assert 0.5 < progress.fraction < 0.6


def test_a_finished_layer_stays_finished():
    progress = Progress()
    progress.update({"digest": "a", "total": 100 * MB, "completed": 50 * MB},
                    now=0.0)
    progress.update({"digest": "a", "total": 100 * MB, "completed": 100 * MB},
                    now=1.0)
    progress.update({"digest": "b", "total": 50 * MB, "completed": 0}, now=2.0)

    assert progress.completed == 100 * MB
    assert progress.fraction < 1.0


def test_success_is_recognised():
    progress = Progress()
    progress.update({"status": "pulling manifest"}, now=0.0)
    assert not progress.done

    progress.update({"status": "success"}, now=1.0)

    assert progress.done
    assert "downloaded" in progress.describe("qwen3:8b", now=1.0)


def test_lines_without_a_digest_only_move_the_status():
    progress = Progress()
    progress.update({"status": "verifying sha256 digest"}, now=0.0)

    assert progress.total == 0
    assert "verifying" in progress.describe("qwen3:8b", now=0.0)


def test_the_rate_is_not_guessed_from_the_first_instant():
    progress = Progress()
    progress.update({"digest": "a", "total": 100 * MB, "completed": 10 * MB},
                    now=0.0)

    assert progress.rate(now=0.1) == 0.0, "too early to say anything"
    assert progress.eta(now=0.1) == 0.0


def test_a_resumed_layer_does_not_report_an_absurd_rate():
    """A cached layer appears instantly at full size; counting it as freshly
    downloaded would claim gigabytes a second."""
    progress = Progress()
    progress.update({"digest": "cached", "total": 500 * MB,
                     "completed": 500 * MB}, now=0.0)
    progress.update({"digest": "new", "total": 100 * MB,
                     "completed": 10 * MB}, now=10.0)

    assert progress.rate(now=10.0) == MB, "10 MB over 10 seconds"


def test_the_line_says_percent_size_rate_and_eta():
    progress = Progress()
    progress.update({"digest": "a", "total": 100 * MB, "completed": 0},
                    now=0.0)
    progress.update({"digest": "a", "total": 100 * MB, "completed": 25 * MB},
                    now=5.0)

    line = progress.describe("qwen3:8b", now=5.0)

    assert "qwen3:8b" in line
    assert "25%" in line
    assert "MB/s" in line
    assert "left" in line


def test_bytes_read_the_way_a_person_would_say_them():
    assert human_bytes(512) == "512 B"
    assert human_bytes(5 * 1024) == "5.0 KB"
    assert human_bytes(5_200_000_000).endswith("GB")


def test_eta_reads_naturally():
    assert human_eta(30) == "30s left"
    assert human_eta(125) == "2m 05s left"
    assert human_eta(3 * 3600 + 300) == "3h 05m left"


def test_obvious_nonsense_is_caught_before_it_becomes_a_500():
    assert looks_like_model_name("qwen3:8b")
    assert looks_like_model_name("hf.co/user/repo:Q4_K_M")
    assert looks_like_model_name("llama3.2")
    assert not looks_like_model_name("")
    assert not looks_like_model_name("   ")
    assert not looks_like_model_name("pull me a model please")
