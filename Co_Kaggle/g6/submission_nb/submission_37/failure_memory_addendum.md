
S37 FAILURE MEMORY (model-guided; no new tools):
Use the most recent tool RESULT as the source of truth. Before issuing a
tool call after an error, compare its tool name, filepath, exact arguments
and known file state with the last failed operation. If they are unchanged,
DO NOT repeat that operation: it cannot provide new evidence.

- edit_file with old_string not found or ambiguous: do not guess another
  replacement immediately. Read the CURRENT target lines once; copy a
  short, unique anchor from that result and change the edit arguments.
- Missing arguments or malformed native tool call: regenerate ONE complete
  native call with the required named fields; do not print a tool call as text.
- Failed search/read: change the path, range, or search term. Do not reread
  identical output without a source change.
- After two failed edits at one location, stop that approach; find a different
  verified location or preserve an already valid patch. Do not overwrite
  existing source through write_file or shell mutation as a workaround.

Keep only the latest failed-operation signature and reason in working
context; replace it when new evidence arrives. Do not spend tool calls
writing a ledger or narrating a plan. A successful tool response alone is
not progress: require a relevant new fact, accepted edit, or behavioral
check. Once a plausible patch exists, verify the requested behavior and
submit rather than repeating investigation.
