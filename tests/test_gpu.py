"""Card detection, and the multi-GPU scheduling switch."""

from __future__ import annotations

from aichatlab import gpu

TWO_1080TI = (
    "0, NVIDIA GeForce GTX 1080 Ti, 11264, 512\n"
    "1, NVIDIA GeForce GTX 1080 Ti, 11264, 0\n"
)

ONE_CARD = "0, NVIDIA GeForce RTX 4090, 24564, 1024\n"


def runner_for(output: str):
    return lambda _command: output


def test_two_cards_are_read_as_two_cards_not_one_big_one():
    """The whole point: 2 × 11 GB is not 22 GB for a single model."""
    cards = gpu.probe(runner_for(TWO_1080TI))

    assert len(cards) == 2
    assert gpu.largest_card(cards) == 11264 * gpu.MIB
    assert gpu.total_vram(cards) == 2 * 11264 * gpu.MIB


def test_used_memory_is_read_when_it_is_there():
    cards = gpu.probe(runner_for(TWO_1080TI))

    assert cards[0].used == 512 * gpu.MIB
    assert cards[0].free == (11264 - 512) * gpu.MIB
    assert cards[1].free == cards[1].total


def test_a_missing_nvidia_smi_is_not_an_error():
    assert gpu.probe(runner_for("")) == []
    assert gpu.total_vram([]) == 0
    assert gpu.largest_card([]) == 0
    assert gpu.describe([]) == ""


def test_malformed_lines_are_skipped_rather_than_crashing():
    cards = gpu.probe(runner_for(
        "garbage\n0, Card A, 8192, 0\nnot, a, number, here\n"))

    assert len(cards) == 1
    assert cards[0].name == "Card A"


def test_card_sizes_are_shown_the_way_the_card_is_named():
    """11,264 MiB is 11.8 decimal GB. Calling a GTX 1080 Ti a "12 GB card"
    reads as wrong and costs the reader their trust in the rest."""
    assert gpu.human_vram(11264 * gpu.MIB) == "11 GB"
    assert gpu.human_vram(24564 * gpu.MIB) == "24 GB"


def test_describe_names_the_layout_the_way_a_person_would():
    assert "2 ×" in gpu.describe(gpu.probe(runner_for(TWO_1080TI)))
    assert "11 GB each" in gpu.describe(gpu.probe(runner_for(TWO_1080TI)))
    assert "22 GB total" in gpu.describe(gpu.probe(runner_for(TWO_1080TI)))
    assert "4090" in gpu.describe(gpu.probe(runner_for(ONE_CARD)))


# -- the spread switch -----------------------------------------------------
def test_the_switch_is_read_from_the_environment():
    assert gpu.spread_enabled({"OLLAMA_SCHED_SPREAD": "1"})
    assert gpu.spread_enabled({"OLLAMA_SCHED_SPREAD": "true"})
    assert not gpu.spread_enabled({"OLLAMA_SCHED_SPREAD": "0"})
    assert not gpu.spread_enabled({})


def test_it_is_only_worth_raising_with_more_than_one_card():
    two = gpu.probe(runner_for(TWO_1080TI))
    one = gpu.probe(runner_for(ONE_CARD))

    assert gpu.worth_offering(two, {})
    assert not gpu.worth_offering(one, {}), "one card cannot be spread across"
    assert not gpu.worth_offering(two, {"OLLAMA_SCHED_SPREAD": "1"}), \
        "already on"


def test_the_offer_explains_it_in_terms_of_this_machine():
    note = gpu.offer_note(gpu.probe(runner_for(TWO_1080TI)))

    assert "11 GB" in note and "22 GB" in note
    assert "OLLAMA_SCHED_SPREAD" in note


def test_a_model_that_fits_one_card_says_nothing(monkeypatch):
    cards = gpu.probe(runner_for(TWO_1080TI))

    assert gpu.placement_advice(cards, 5 * 10**9) == ""


def test_a_model_that_needs_both_cards_says_so(monkeypatch):
    monkeypatch.setattr(gpu, "spread_enabled", lambda *a, **k: False)
    cards = gpu.probe(runner_for(TWO_1080TI))

    advice = gpu.placement_advice(cards, 15 * 10**9)

    assert "would fit across both" in advice
    assert "OLLAMA_SCHED_SPREAD" in advice


def test_a_model_too_big_for_the_whole_machine_is_called_out(monkeypatch):
    monkeypatch.setattr(gpu, "spread_enabled", lambda *a, **k: True)
    cards = gpu.probe(runner_for(TWO_1080TI))

    advice = gpu.placement_advice(cards, 30 * 10**9)

    assert "run on the processor" in advice
    assert "smaller model" in advice


def test_nothing_is_said_when_the_hardware_is_unknown():
    assert gpu.placement_advice([], 30 * 10**9) == ""
    assert gpu.placement_advice(gpu.probe(runner_for(TWO_1080TI)), 0) == ""


def test_enabling_it_says_ollama_must_be_restarted(monkeypatch):
    """The variable is read at start-up, so setting it changes nothing until
    the server is restarted — leaving that out would look like a broken fix."""
    monkeypatch.setattr(gpu.platform, "system", lambda: "Windows")
    monkeypatch.setattr(gpu.os, "environ", {})

    worked, message = gpu.enable_spread(runner=lambda _cmd: "SUCCESS")

    assert worked
    assert "restart" in message.lower() or "open it again" in message.lower()


def test_a_failure_to_set_it_says_how_to_do_it_by_hand(monkeypatch):
    monkeypatch.setattr(gpu.platform, "system", lambda: "Windows")

    worked, message = gpu.enable_spread(runner=lambda _cmd: "")

    assert not worked
    assert "by hand" in message


def test_on_linux_it_explains_the_service_file_instead(monkeypatch):
    monkeypatch.setattr(gpu.platform, "system", lambda: "Linux")

    worked, message = gpu.enable_spread()

    assert not worked
    assert "systemctl" in message


def test_it_looks_beyond_the_path_before_giving_up(monkeypatch):
    """A GUI app launched from a shortcut does not always inherit the PATH a
    command prompt has, and a silent "no cards" is indistinguishable from a
    machine that genuinely has none."""
    monkeypatch.setattr(gpu.platform, "system", lambda: "Windows")
    monkeypatch.setattr(gpu.shutil, "which", lambda _name: None)
    monkeypatch.setattr(gpu.os.path, "exists",
                        lambda path: path == gpu.WINDOWS_PATHS[1])

    tried = []

    def runner(command):
        tried.append(command[0])
        return TWO_1080TI if command[0] == gpu.WINDOWS_PATHS[1] else ""

    cards = gpu.probe(runner)

    assert len(cards) == 2
    assert gpu.WINDOWS_PATHS[1] in tried


def test_the_bare_name_is_still_tried_first(monkeypatch):
    monkeypatch.setattr(gpu.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    tried = []

    def runner(command):
        tried.append(command[0])
        return TWO_1080TI

    assert len(gpu.probe(runner)) == 2
    assert tried == ["nvidia-smi"], "one call is enough when it works"


def test_no_cards_anywhere_is_still_just_an_empty_list(monkeypatch):
    monkeypatch.setattr(gpu.shutil, "which", lambda _name: None)
    monkeypatch.setattr(gpu.os.path, "exists", lambda _path: False)

    assert gpu.probe(lambda _command: "") == []


def test_the_short_form_fits_a_narrow_sidebar():
    """The full description loses its own ending in a 390px column, taking
    the "turn on" link off-screen with it."""
    cards = gpu.probe(runner_for(TWO_1080TI))

    short = gpu.short_describe(cards)

    assert short == "2 × GTX 1080 Ti · 22 GB"
    assert len(short) < len(gpu.describe(cards))


def test_vendor_noise_is_stripped_from_card_names():
    assert gpu.short_name("NVIDIA GeForce GTX 1080 Ti") == "GTX 1080 Ti"
    assert gpu.short_name("NVIDIA RTX A4000") == "RTX A4000"
    assert gpu.short_name("Quadro P2000") == "Quadro P2000"


def test_one_card_reads_naturally_in_the_short_form():
    assert gpu.short_describe(gpu.probe(runner_for(ONE_CARD))) == \
        "RTX 4090 · 24 GB"


# -- what is actually free -------------------------------------------------
# Straight from a real session: both cards mostly consumed by a second Ollama
# on the same machine, leaving a 22 GB box with 5 GB to work with.
OCCUPIED = (
    "0, NVIDIA GeForce GTX 1080 Ti, 11264, 8561\n"
    "1, NVIDIA GeForce GTX 1080 Ti, 11264, 8564\n"
)


def test_free_memory_is_counted_not_just_capacity():
    cards = gpu.probe(runner_for(OCCUPIED))

    assert gpu.used_vram(cards) == (8561 + 8564) * gpu.MIB
    assert gpu.free_vram(cards) == (11264 - 8561 + 11264 - 8564) * gpu.MIB
    assert gpu.largest_free(cards) == (11264 - 8561) * gpu.MIB


def test_an_idle_machine_is_not_reported_as_busy():
    assert not gpu.looks_occupied(gpu.probe(runner_for(TWO_1080TI)))
    assert gpu.free_note(gpu.probe(runner_for(TWO_1080TI))) == ""


def test_a_machine_with_someone_else_on_it_says_so():
    note = gpu.free_note(gpu.probe(runner_for(OCCUPIED)))

    assert "already in use" in note
    assert "on one card" in note
    assert "OLLAMA_KEEP_ALIVE" in note


def test_a_model_is_judged_against_free_space_not_total(monkeypatch):
    """Saying "9.7 GB fits an 11 GB card" is true and useless when 8.5 GB of
    that card is already gone — and it is exactly the wrong answer, because
    the model goes to the CPU and nothing explains why."""
    monkeypatch.setattr(gpu, "spread_enabled", lambda *a, **k: True)
    cards = gpu.probe(runner_for(OCCUPIED))

    advice = gpu.placement_advice(cards, int(9.7 * 10**9))

    assert advice, "a model that cannot fit must not pass silently"
    assert "free" in advice
    assert "processor" in advice


def test_the_same_model_on_empty_cards_says_nothing(monkeypatch):
    monkeypatch.setattr(gpu, "spread_enabled", lambda *a, **k: True)
    cards = gpu.probe(runner_for(TWO_1080TI))

    assert gpu.placement_advice(cards, int(9.7 * 10**9)) == ""


def test_a_model_that_needs_both_cards_free_is_described_that_way(monkeypatch):
    monkeypatch.setattr(gpu, "spread_enabled", lambda *a, **k: True)
    # Half of each card gone: 5.5 GB free per card, 11 GB between them.
    half = ("0, GTX 1080 Ti, 11264, 5632\n"
            "1, GTX 1080 Ti, 11264, 5632\n")
    cards = gpu.probe(runner_for(half))

    advice = gpu.placement_advice(cards, int(9.5 * 10**9))

    assert "no single card has that free" in advice
    assert "between them" in advice
