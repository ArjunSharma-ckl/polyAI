from __future__ import annotations

import base64

import pytest

from polyai.exceptions import InvalidInputError
from polyai.response import AIResponse, AITokens


def test_response_shape_and_string_conversion() -> None:
    response = AIResponse(
        text="hello",
        provider="openai",
        model="gpt-4o",
        mode="text",
        tokens=AITokens(1, 2, 3),
        raw={"ok": True},
    )

    assert str(response) == "hello"
    assert response.to_dict()["tokens"]["total"] == 3
    assert "AIResponse" in repr(response)


def test_save_image_and_audio(tmp_path) -> None:
    response = AIResponse(
        text="",
        provider="openai",
        model="dall-e-3",
        mode="image",
        image_data=b"image",
        audio_data=b"audio",
    )
    image_path = tmp_path / "image.png"
    audio_path = tmp_path / "audio.mp3"

    response.save_image(str(image_path))
    response.save_audio(str(audio_path))

    assert image_path.read_bytes() == b"image"
    assert audio_path.read_bytes() == b"audio"
    assert response.to_dict()["image_data"] == base64.b64encode(b"image").decode("ascii")


def test_save_missing_media_raises() -> None:
    response = AIResponse(text="", provider="openai", model="gpt-4o", mode="text")

    with pytest.raises(InvalidInputError):
        response.save_audio("missing.mp3")


def test_retry_callback() -> None:
    response = AIResponse(text="old", provider="openai", model="gpt-4o", mode="text")
    fresh = AIResponse(text="new", provider="openai", model="gpt-4o", mode="text")
    response.with_retry(lambda: fresh)

    assert response.retry().text == "new"
