"""
What:     A LangChain ChatOpenAI model governed by a Driftlock mission via
          DriftlockCallbackHandler — mid-run kill/downgrade, no DriftlockClient
          on the call path.
Requires: pip install "driftlock[langchain]" langchain-openai  +  OPENAI_API_KEY
Run:      OPENAI_API_KEY=sk-... python examples/langchain_agent_demo.py "renewable energy storage"
"""

from __future__ import annotations

import os
import sys

import driftlock
from driftlock import DriftlockClient, DriftlockConfig, MissionBudgetExceededError
from driftlock.integrations import DriftlockCallbackHandler


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    topic = argv[0] if argv else "renewable energy storage"

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: set OPENAI_API_KEY to run the demo.", file=sys.stderr)
        return 1

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI
    except ImportError:
        print(
            'LangChain is not installed. Install the extra:\n'
            '  pip install "driftlock[langchain]" langchain-openai',
            file=sys.stderr,
        )
        return 1

    # A DriftlockClient is used purely as the persistence target so mission_stats
    # and the CLI can see the LangChain-driven calls.
    dl_client = DriftlockClient(
        api_key=os.environ["OPENAI_API_KEY"],
        config=DriftlockConfig(log_level="WARNING", log_json=False),
    )

    handler = DriftlockCallbackHandler(client=dl_client, endpoint="lc_research")
    llm = ChatOpenAI(model="gpt-4o", callbacks=[handler], max_tokens=200)

    subtasks = [
        f"Summarize the current state of {topic}.",
        f"List the main challenges in {topic}.",
        f"Describe recent breakthroughs in {topic}.",
        f"Outline the outlook for {topic}.",
    ]

    print(f"\nLangChain + Driftlock — topic: {topic!r}\n")
    with driftlock.mission(
        "langchain_agent",
        budget_usd=0.03,
        expected_calls=len(subtasks),
        on_exceed="kill",
    ) as mission:
        try:
            for st in subtasks:
                resp = llm.invoke([
                    SystemMessage(content="You are a concise research analyst."),
                    HumanMessage(content=st),
                ])
                print(f"  • {st[:48]:<50} spent=${mission.spent:.4f}")
                _ = resp.content
        except MissionBudgetExceededError as exc:
            print(f"\n  🛑 Mission killed mid-run: {exc.rule_name}\n")

    print("\n" + "=" * 60)
    print(
        f"Mission: ${mission.spent:.4f} spent | {mission.call_count} calls | "
        f"status={mission.status}"
    )
    print(f"Inspect:  driftlock mission {mission.mission_id}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
