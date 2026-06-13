"""
What:     The smallest possible Driftlock integration — wrap the OpenAI client,
          make one call, read aggregated cost/token stats. No framework needed.
Requires: OPENAI_API_KEY (makes one real ~$0.00001 call to gpt-4o-mini).
Run:      OPENAI_API_KEY=sk-... python examples/basic_usage.py
"""

import json
import os

from driftlock import DriftlockClient, DriftlockConfig

client = DriftlockClient(
    api_key=os.environ["OPENAI_API_KEY"],
    config=DriftlockConfig(log_json=False),  # human-readable logs for the terminal
)

# Identical call signature to openai.OpenAI().chat.completions.create()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is 2 + 2?"}],
    _dl_endpoint="math_demo",
)

print("\nReply:", response.choices[0].message.content)

# Pull aggregated stats from local SQLite
print("\n--- Stats ---")
print(json.dumps(client.stats(), indent=2))
