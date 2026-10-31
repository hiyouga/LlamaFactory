---
name: llamafactory-v1-docs
description: Write, review, update, build, or preview LlamaFactory v1 documentation. Use for docs work grounded in src/llamafactory/v1; not for running model training.
---

# LlamaFactory v1 Documentation

## Scope

- Work in the repository containing both `docs/` and `src/llamafactory/v1/`. Check the branch and existing changes first.
- Treat this branch's v1 implementation as the source of truth; do not infer behavior from v0 or upstream code.
- Maintain Chinese and English counterparts for pages touched by the task unless the user limits the language. Align relative page paths, structure, technical meaning, examples, and navigation; translate prose naturally while preserving code identifiers and commands. Report any remaining language gaps.
- Preserve unrelated edits and explicitly deferred issues. Review-only requests stay read-only; build-only requests do not rewrite content.

## Writing

- Explain features directly: behavior, configuration, execution, and constraints. Avoid assumed user goals, selection advice, rhetorical questions, and “use case” columns unless requested.
- Keep content ownership clear: feature guides explain usage; configuration pages define fields and semantics; developer guides explain internals. Link between them instead of repeating full sections.
- Use tables for factual comparisons. Keep examples short and distinguish complete configurations from partial snippets without repetitive setup reminders.
- Explain defaults, omitted versus explicit values, precedence, units, and conditional behavior—not just parameter names.
- In developer guides, show responsibilities, inputs, outputs, state ownership, and call order. Use a small example or tensor shapes when abstractions need clarification.
- Use one H1 per page, meaningful headings, language-tagged code blocks, relative links, and the appropriate `toctree` entry for new pages.

## Source and Examples

- Trace relevant configuration parsing, plugin registration, and call sites. A declared field or interface alone does not establish support.
- Distinguish related concepts, such as quantization plugins, backends, bit widths, and formats. Verify current support rather than hardcoding assumptions into this skill.
- Keep model IDs, adapter paths, model directories, and filenames consistent across training, inference, and export examples. Match each downstream input to the preceding output.
- Parse changed YAML examples and check their placement and composition. Examples described as runnable need the required imports, interfaces, and registration.
- Document confirmed behavior and limitations. Report unresolved implementation questions separately; do not insert speculation or conversation history into public documentation.

## Build and Preview

Use an existing Python environment with `docs/requirements.txt` installed. Build both languages from the repository root, or just the language explicitly requested:

```bash
for doc_language in zh en; do
  env LC_ALL=C LANG=C python3 -m sphinx -b html -n -W --keep-going "docs/$doc_language" "docs/_build/html/$doc_language" || exit 1
done
```

Review the diff and build diagnostics; check changed pages, links, anchors, and table/code rendering. Fix issues introduced by authorized edits and rebuild. For review-only or build-only requests, report existing failures without expanding the scope. Do not disable strict checks or run model training merely to validate documentation.

When a preview is requested, reuse a server serving this output or start one on an available local port:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory docs/_build/html
```

Open the relevant pages under `/zh/` and `/en/` on the preview server and verify the served content and language-switch links. Do not stop unrelated servers or publish the site automatically.

Report the changed files and completed checks briefly; include a preview link if available. Distinguish a successful documentation build from validated model execution.
