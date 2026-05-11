from polyai import AI

# Pass a local file path or HTTPS URL with mode="vision".
response = AI(
    "openai",
    "gpt-4o",
    prompt="Describe this image for alt text.",
    image="./photo.jpg",
    mode="vision",
)

# The description comes back as normal response text.
print(response.text)
