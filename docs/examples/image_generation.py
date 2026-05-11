from polyai import AI

# Image mode normalizes provider output into AIResponse.
response = AI(
    "openai",
    "dall-e-3",
    prompt="A bright product photo of a ceramic coffee mug on a wood table",
    mode="image",
    size="1024x1024",
)

# Some providers return a URL, some bytes. save_image handles both.
print(response.image_url)
response.save_image("mug.png")
