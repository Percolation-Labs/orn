# Quickly trying out Inception Mercury

(Abridged recap from the original. Keep this file as a style reference only; do not quote verbatim in new drafts.)

Inception Labs recently introduced their commercial scale diffusion language model. I tested the API to see how streaming diffusion works in a notebook, and the visual rendering of the "filling in the blanks" idea is conceptually compelling.

I used `mercury-coder-small` with the Percolate agent framework. For a database search and a tool call the model finished in about four seconds, with roughly one second of that being the final output generation.

For comparison with OpenAI's recent releases, mercury-coder-small came in around four seconds, GPT-4.1 around ten, and GPT-4.1-mini around six on the same task. Rough numbers, one run each, so take them as a vibe check.

A couple of quirks showed up. The model objected to a temperature of 0.01 that other models accept without complaint. On some more complex tasks the stream produced blank content and did not explain what confused it.

It's too early for me to say how useful the Mercury models will be in practice but based on speed and just being conceptually fascinating, this seems like one to watch.
