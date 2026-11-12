import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from agent_dse.llm import LLM
from agent_dse.path import KEY_JSON_PATH

def main():
    print(f"key.json = {KEY_JSON_PATH}")
    llm = LLM.from_file(path=KEY_JSON_PATH)
    print(f"model    = {llm.model}")
    print(f"url      = {llm.url}")

    answer = llm.chat("You are a terse assistant.", "Reply with exactly the word: Hello World")
    print(f"raw reply = {answer!r}")
    assert "hello" in answer.lower(), f"unexpected reply: {answer!r}"

    payload = llm.chat_json(
        "You output only JSON.",
        'Return exactly {"ok": true, "n": 7}',
    )



    print(f"json reply = {payload}")
    assert payload.get("ok") is True and payload.get("n") == 7

    print("STEP1_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main()) 