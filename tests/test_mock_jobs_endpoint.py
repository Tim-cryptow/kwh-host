"""The buyer stand-in endpoint refuses what it cannot tokenize with a 400 and a reason.

On the GPU VM (2026-10-05) the mock platform's tokenizer loaded but its chat template needed
jinja2, which transformers does not install; chat jobs died with a bare 500."""

import asyncio

import httpx

from kwh_host.jobspec import JobInvalid
from kwh_host.platform.mock import ChallengePool, Continuation, MockPlatform, Router, Settings, create_app


class NoChatTemplate:
    chat_error = "apply_chat_template requires jinja2 to be installed."

    def encode(self, text):
        return [128000] + [ord(c) for c in text]

    def chat(self, messages):
        raise JobInvalid(f"chat messages need the chat template, which is unavailable here: {self.chat_error}")


def test_chat_without_a_template_is_refused_with_the_reason():
    pool = ChallengePool([Continuation(id="c1", prompt_text="hello", prompt_tokens=2, continuation_token_ids=[3],
                                       reference_mean_logprob=-1.0)])
    platform = MockPlatform(pool, Settings(accept_uncertified=True, allow_bare_metal=True))
    app = create_app(platform, Router(platform), tokenizer=NoChatTemplate())

    async def go():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://mock") as c:
            health = (await c.get("/healthz")).json()
            chat = await c.post("/v1/mock/jobs", json={"requests": [{"messages": [{"role": "user", "content": "hi"}]}]})
            return health, chat
    health, chat = asyncio.run(go())
    assert health["tokenizer"] is True and health["chat"] is False
    assert chat.status_code == 400 and "jinja2" in chat.json()["reason"] and chat.json()["status"] == "failed"
