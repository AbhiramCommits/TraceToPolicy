"""Deterministic fake generator for offline tests (no model weights required)."""

from __future__ import annotations

from tracetopolicy.inference import Generation, Generator


class ScriptedGenerator(Generator):
    """Returns responses from a list (cycled) or a callable.

    ``responses`` may be:
      * a list of SQL strings, returned in order (and then repeated), or
      * a callable ``(prompt: str, index: int) -> str``.
    """

    backend = "scripted"

    def __init__(self, responses, name: str = "scripted"):
        self.responses = responses
        self.name = name
        self.max_new_tokens = 96
        self.calls = 0

    def render(self, messages):
        return messages[-1]["content"]

    def generate(self, prompts, max_new_tokens=96, temperature=0.0, top_p=1.0, seed=None):
        out = []
        for prompt in prompts:
            if callable(self.responses):
                text = self.responses(prompt, self.calls)
            else:
                text = self.responses[self.calls % len(self.responses)]
            self.calls += 1
            out.append(
                Generation(
                    text=text,
                    prompt_tokens=len(prompt.split()),
                    completion_tokens=max(1, len(text.split())),
                    latency_ms=1.0,
                )
            )
        return out


def gold_generator(tasks):
    """A generator that returns the gold SQL for whichever task is in the prompt."""
    by_question = {t["question"]: t["gold_sql"] for t in tasks}

    def respond(prompt, index):
        for question, sql in by_question.items():
            if question in prompt:
                return sql
        return "SELECT 1"

    return ScriptedGenerator(respond, name="gold")


def scripted_all_wrong(sql: str = "SELECT 1"):
    return ScriptedGenerator([sql], name="wrong")
