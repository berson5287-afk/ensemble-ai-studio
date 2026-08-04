# Prior art: does anything else do this?

Research done August 2026, ahead of interview questions like *"isn't this just
Open WebUI?"* — which someone will ask.

## The short answer

Every individual capability in AI Chat Lab exists somewhere else. The
**combination** does not appear to, and two specific pieces are genuinely hard
to find in any end-user application: structured multi-agent patterns beyond
"answer in parallel then merge", and benchmarking built into the same window as
the chat.

Nothing found is a straight competitor. Nothing found makes this project
redundant. But the honest framing is not "nobody has done this" — it is "the
pieces are scattered across a dozen tools, and this one puts a specific set of
them together."

## The closest things that exist

**Open WebUI** is the giant in this space (140,000+ GitHub stars) and the one
an interviewer is most likely to name. Its Multi-Model Chat sends "your prompt
to two or more selected models at the same time", and the docs describe a
Mixture of Agents workflow where "the synthesizer model reads all the draft
answers and combines them into one final, polished response." It also has web
search and RAG. That covers broadcast and judge-style synthesis, and it covers
them in a far more polished product than this one.

Two real differences are worth knowing precisely. First, Open WebUI's
multi-server support is **load balancing, not pooling**: you can add multiple
Ollama instances and it "will distribute requests between them using a random
selection strategy", which requires the model IDs to match across instances so
they merge into a single entry. That treats a second server as an interchangeable
replica of the first. AI Chat Lab treats a local box and a networked box as two
distinct pools with their own connection status, which is the right model when
the networked machine has a 20B that the laptop cannot run. It has been a
long-standing request on their tracker (issue #788, "multiple Ollama servers in
one webui + load balancing"). Second, the Mixture of Agents behaviour is
available largely through community **pipe functions and pipelines** rather than
as a core mode — a plugin framework, which is a strength for them and a
different thing from six modes in the mode dropdown.

**big-AGI's Beam** is the closest single feature to this project's design. It
sends one prompt to several models simultaneously, shows them side by side, then
offers four fusion strategies: Fusion, Checklist, Compare and Custom. That is
broadcast plus judge synthesis, done well. What it does not do, per its own
documentation, is debate rounds, models responding to each other, critique
chains, or any performance benchmarking.

**Msty's Split Chat** lets you "chat with multiple models at once, making it
easy to compare responses." It is comparison only — the documentation makes no
mention of models seeing each other's answers, of a judge, or of synthesis.

## The pieces that exist separately

Benchmarking local models is well served, but always as its own tool:
`llm-benchmark`, `ollama-benchmark`, OllaMan's token-speed test, and hosted
leaderboards like Ollama TPS. None of them is the window you are chatting in,
which means comparing "which of my models is actually worth using" is a context
switch rather than a click.

Local research over SearXNG is well served by **Perplexica**, an open-source
Perplexity alternative that pairs SearXNG with Ollama. It is a search engine
with a chat interface rather than a multi-model orchestrator, and it does not
run the same question across several models or check its own citations against
what was actually retrieved.

Multi-agent debate is the most striking gap. The GitHub `multi-agent-debate`
topic is almost entirely **research frameworks and paper code** — mallm,
DAR, cultural_debate, minerva, debate-or-vote — plus a small number of
developer tools like CodeAgora (five models debating a code review). These are
libraries and experiments, not applications a non-programmer opens and uses.
The patterns are well studied in the literature and largely absent from the
desktop.

## Where that leaves this project

| Capability | Who else has it |
| --- | --- |
| Broadcast one prompt to many models | Open WebUI, big-AGI Beam, Msty, LibreChat |
| Synthesise the answers (judge / MoA) | big-AGI Beam, Open WebUI (via community pipes) |
| Two model pools with separate identities | Not found — Open WebUI load-balances replicas instead |
| Debate rounds with a judge | Research frameworks only |
| Critique chain (draft → critique → revise) | Research frameworks only |
| Open-ended model conversation you can join | Rare; mostly toy scripts |
| Benchmarking inside the chat app | Not found — always a separate tool |
| SearXNG research with citation verification | Perplexica does the search half |
| Attachment pinning across trim and compaction | Not found as a described feature |

The last row is the kind of thing nobody advertises because it only matters
once you have hit it — which is exactly why it is worth mentioning in an
interview.

## How to answer the interview question

Do not claim novelty of category. Claim the engineering, because that is where
the defensible work is and it is all in the repository history: streaming with
real cancellation that closes the connection so the server stops generating; a
context budget tied to the `num_ctx` actually requested so the two cannot drift;
attachments that survive trimming, saving, reloading and compaction; a UI-free
core with 499 tests that need neither a network nor a display.

The most honest and most persuasive line is the bug list in the README. "I
found that raising my own trimming budget above the window I was requesting was
worse than not raising it at all, because the server then discards the overflow
oldest-first without any of the care my own trimmer takes" is a sentence very
few candidates can say about their own code. Breadth of features is not the
argument. Knowing exactly why something broke is.

## Honest risks

Open WebUI is better resourced, better looking and more capable in almost every
direction, and an interviewer who uses it daily will know that. Get ahead of it:
say plainly that it is the obvious production choice, that this project is a
harness built to explore orchestration patterns and the failure modes underneath
them, and that you learned more from the four days of context-management bugs
than you would have from installing theirs.

---

## Sources

- [Multi-Model Chats — Open WebUI docs](https://docs.openwebui.com/features/chat-conversations/chat-features/multi-model-chats/)
- [Starting with Ollama — Open WebUI docs](https://docs.openwebui.com/getting-started/quick-start/connect-a-provider/starting-with-ollama/)
- [feat: multiple Ollama servers in one webui + load balancing — Open WebUI issue #788](https://github.com/open-webui/open-webui/issues/788)
- [Mixture of Agents Pipe Function — Open WebUI Community](https://openwebui.com/f/srossitto79/moa_pipe)
- [Beam: Better Decisions, Lower Risk, with Multi-Model AI Reasoning — Big-AGI](https://big-agi.com/blog/beam-multi-model-ai-reasoning)
- [Split Chat — Msty Studio docs](https://docs.msty.ai/studio/conversations/split-chat)
- [multi-agent-debate — GitHub topic](https://github.com/topics/multi-agent-debate)
- [llm-benchmark — GitHub](https://github.com/MinhNgyuen/llm-benchmark)
- [Perplexica — GitHub](https://github.com/OmarElKadri/Perplexica)
- [Best Ollama & Local LLM Frontends 2026: 8 Compared](https://www.promptquorum.com/local-llms/best-local-llm-frontends)
