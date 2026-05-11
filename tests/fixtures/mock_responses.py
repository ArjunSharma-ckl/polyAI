from __future__ import annotations

OPENAI_CHAT = {
    "id": "chatcmpl_test",
    "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
}

OPENAI_EMBED = {
    "id": "emb_test",
    "data": [{"embedding": [0.1, 0.2, 0.3]}],
    "usage": {"prompt_tokens": 4, "total_tokens": 4},
}

ANTHROPIC_MESSAGE = {
    "id": "msg_test",
    "content": [{"type": "text", "text": "hello"}],
    "usage": {"input_tokens": 2, "output_tokens": 3},
    "stop_reason": "end_turn",
}

GOOGLE_GENERATE = {
    "candidates": [
        {
            "content": {"parts": [{"text": "hello"}]},
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 3, "totalTokenCount": 5},
}
