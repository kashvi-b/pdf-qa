import anthropic

client = anthropic.Anthropic(
    api_key="PASTE_YOUR_KEY_HERE"
)

response = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=50,
    messages=[{"role": "user", "content": "Hello"}]
)

print(response.content)