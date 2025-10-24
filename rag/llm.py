from __future__ import annotations

from langchain_community.llms import Ollama


class LocalLLM:
    def __init__(
        self,
        model: str = "llama3.2:3b",
        temperature: float = 0.2,
        top_p: float = 0.95,
        num_predict: int = 512,
    ):
        self._llm = Ollama(
            model=model, temperature=temperature, top_p=top_p, num_predict=num_predict
        )

    def generate(self, prompt: str) -> str:
        return self._llm.invoke(prompt)
