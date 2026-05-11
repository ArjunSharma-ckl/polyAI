from polyai import AI

# Start a session; later sends include the same history.
chat = AI.chat("anthropic", "claude-sonnet-4-5")

# The first message gives the model something to remember.
chat.send("My name is Alex")
response = chat.send("What is my name?")

# Export helpers return strings and can also write to a file path.
print(response.text)
print(chat.export_markdown())
