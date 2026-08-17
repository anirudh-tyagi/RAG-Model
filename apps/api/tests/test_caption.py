"""Figure captioning: markers inlined, unresolvable images dropped, cap honoured."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rag.config import Settings
from rag.ingest.caption import caption_images
from rag.ingest.parse import ImageRef, ParsedDocument, ParsedPage

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 64


def make_image(image_dir: Path, name: str) -> Path:
    image_dir.mkdir(parents=True, exist_ok=True)
    path = image_dir / name
    path.write_bytes(PNG_BYTES)
    return path


def parsed_with(image_dir: Path, *names: str) -> ParsedDocument:
    links = " ".join(f"![]({name})" for name in names)
    document = ParsedDocument(
        pages=[ParsedPage(page=1, markdown=f"Intro text. {links} Closing text.")]
    )
    for name in names:
        document.images.append(ImageRef(target=name, path=make_image(image_dir, name), page=1))
    return document


async def test_caption_replaces_the_image_link_with_a_marker(
    tmp_path: Path, settings: Settings, fake_llm: Any
) -> None:
    image_dir = tmp_path / "images"
    parsed = parsed_with(image_dir, "fig1.png")

    captions = await caption_images(parsed, image_dir, settings, fake_llm)

    assert len(captions) == 1
    markdown = parsed.pages[0].markdown
    assert "![" not in markdown
    assert "[Figure:" in markdown
    # Surrounding prose survives.
    assert "Intro text." in markdown
    assert "Closing text." in markdown


async def test_every_resolvable_figure_is_captioned(
    tmp_path: Path, settings: Settings, fake_llm: Any
) -> None:
    image_dir = tmp_path / "images"
    parsed = parsed_with(image_dir, "a.png", "b.png", "c.png")

    captions = await caption_images(parsed, image_dir, settings, fake_llm)

    assert {c.target for c in captions} == {"a.png", "b.png", "c.png"}
    assert len(fake_llm.captions) == 3


async def test_captions_carry_the_page_number(
    tmp_path: Path, settings: Settings, fake_llm: Any
) -> None:
    image_dir = tmp_path / "images"
    parsed = ParsedDocument(pages=[ParsedPage(page=9, markdown="![](fig.png)")])
    parsed.images.append(ImageRef(target="fig.png", path=make_image(image_dir, "fig.png"), page=9))

    captions = await caption_images(parsed, image_dir, settings, fake_llm)

    assert captions[0].page == 9


async def test_unresolvable_image_link_is_removed_not_captioned(
    tmp_path: Path, settings: Settings, fake_llm: Any
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir(parents=True)
    # Link present in the markdown, but no such file was ever written.
    parsed = ParsedDocument(pages=[ParsedPage(page=1, markdown="Before ![](ghost.png) after.")])

    captions = await caption_images(parsed, image_dir, settings, fake_llm)

    assert captions == []
    assert "ghost.png" not in parsed.pages[0].markdown
    assert "![" not in parsed.pages[0].markdown
    assert fake_llm.captions == []


async def test_disabling_captioning_strips_links_without_calling_the_model(
    tmp_path: Path, settings: Settings, fake_llm: Any
) -> None:
    image_dir = tmp_path / "images"
    parsed = parsed_with(image_dir, "fig1.png")
    disabled = settings.model_copy(update={"caption_images": False})

    captions = await caption_images(parsed, image_dir, disabled, fake_llm)

    assert captions == []
    assert "![" not in parsed.pages[0].markdown
    assert fake_llm.captions == []


async def test_caption_cap_limits_the_number_of_vision_calls(
    tmp_path: Path, settings: Settings, fake_llm: Any
) -> None:
    image_dir = tmp_path / "images"
    parsed = parsed_with(image_dir, "a.png", "b.png", "c.png", "d.png")
    capped = settings.model_copy(update={"max_captions_per_doc": 2})

    captions = await caption_images(parsed, image_dir, capped, fake_llm)

    assert len(captions) == 2
    assert len(fake_llm.captions) == 2


async def test_a_failing_figure_does_not_abort_the_rest(tmp_path: Path, settings: Settings) -> None:
    """One bad figure should cost one caption, not the whole document."""
    from rag.chat.llm import LlmError

    class HalfBrokenLlm:
        configured = True

        def __init__(self) -> None:
            self.seen: list[str] = []

        async def describe_image(self, image_path: Any, prompt: str) -> str:
            name = Path(image_path).name
            self.seen.append(name)
            if name == "b.png":
                raise LlmError("vision model choked")
            return f"Caption for {name}"

    llm = HalfBrokenLlm()
    image_dir = tmp_path / "images"
    parsed = parsed_with(image_dir, "a.png", "b.png", "c.png")

    captions = await caption_images(parsed, image_dir, settings, llm)

    assert {c.target for c in captions} == {"a.png", "c.png"}
    assert len(llm.seen) == 3


async def test_progress_callback_reports_completion_counts(
    tmp_path: Path, settings: Settings, fake_llm: Any
) -> None:
    image_dir = tmp_path / "images"
    parsed = parsed_with(image_dir, "a.png", "b.png")
    seen: list[tuple[int, int]] = []

    await caption_images(
        parsed,
        image_dir,
        settings,
        fake_llm,
        on_progress=lambda done, total: seen.append((done, total)),
    )

    assert [done for done, _ in seen] == [1, 2]
    assert all(total == 2 for _, total in seen)


async def test_repeated_image_is_captioned_once(
    tmp_path: Path, settings: Settings, fake_llm: Any
) -> None:
    """A logo repeated on every page shouldn't cost one vision call per page."""
    image_dir = tmp_path / "images"
    path = make_image(image_dir, "logo.png")
    parsed = ParsedDocument(
        pages=[
            ParsedPage(page=1, markdown="![](logo.png) page one"),
            ParsedPage(page=2, markdown="![](logo.png) page two"),
        ]
    )
    parsed.images.append(ImageRef(target="logo.png", path=path, page=1))

    captions = await caption_images(parsed, image_dir, settings, fake_llm)

    assert len(captions) == 1
    assert len(fake_llm.captions) == 1
    # Both pages still get a marker.
    assert all("[Figure:" in page.markdown for page in parsed.pages)
