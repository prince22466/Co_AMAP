# Contracts across API layers

Build a short map: public entry -> forwarding/conversion -> stored state -> runtime consumer -> observable output. Read evidence for each relevant edge. Adding an option at a low-level object is incomplete if the supported entry point cannot accept or forward it. Check default inheritance and explicit override separately; do not invent defaults from the implementation you just wrote.

Validation has a time as well as a predicate. Construction, mutation and use may have different contracts. Inspect serializers, assignment behavior and existing validation conventions. Preserve valid objects and output types. Derive invalid-input expectations from the requested behavior, rather than choosing the easiest validation hook.

Errors also have observable text. Inspect nearby validators and existing assertions for field naming and message conventions. Keep established wording stable when extending a check. If exact wording is absent from both issue and source, mark it uncertain; a generic skill cannot recover an unspecified grading string.

Compatibility changes need both sides: old valid input, newly supported input, and a neighboring invalid input when applicable. Do not infer duck-typing behavior from a single convenience example. Distinguish a missing attribute, an explicit None, a callable attribute, and special-method lookup where the source makes these relevant.

Formatting and parsing need exact boundary checks. Compare repr(output) or structured values for empty input, separators, trailing delimiters and repeated delimiters. A visually plausible printout is not a successful assertion. Preserve round-trip or losslessness properties only where the issue/source establishes them.

Sparse issue titles provide less evidence. Inspect nearby tests, public exports and similar implementation conventions. Mark unknown names and behavior as assumptions; do not turn an assumption into an exact requirement. A helper can identify a missing question, but cannot recover information absent from its inputs.
