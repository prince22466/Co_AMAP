
GEMMA 4 TOOL-ARGUMENT STRING FORMAT (controlled experiment):
The native tool schema still determines the tool and required arguments.
For an actual edit_file call, serialize each STRING value between Gemma 4's
<|"|> delimiters, not ordinary quoted JSON/Python strings. The following
is an argument-serialization example, NOT a shell command or a prose tool call:

filepath:<|"|>pkg/module.py<|"|>,old_string:<|"|>x = values[0]<|"|>,new_string:<|"|>x = values[1]<|"|>,allow_multiple:false

Keep filepath, old_string and new_string as literal parameter names. Preserve
exact code, indentation, quotes, commas and brackets INSIDE the delimiters.
Do not add ordinary quotes around them. Use the actual native tool interface,
never just print this example. If an edit is rejected, inspect the tool error;
do not repeat the same malformed operation unchanged.
