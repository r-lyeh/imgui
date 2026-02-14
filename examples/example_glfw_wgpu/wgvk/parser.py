import sys
import os
import clang.cindex

def find_instrumentation_points(cursor, source_file):
    points = []
    for node in cursor.get_children():
        if (node.kind == clang.cindex.CursorKind.FUNCTION_DECL and
                node.is_definition() and
                node.location.file and
                node.location.file.name == source_file):
            
            if node.spelling.startswith("wgpu"):
                body = None
                for child in node.get_children():
                    if child.kind == clang.cindex.CursorKind.COMPOUND_STMT:
                        body = child
                        break
                
                if body:
                    entry_offset = body.extent.start.offset + 1
                    exit_offset = body.extent.end.offset - 1

                    macro_indent_str = ""
                    body_children = list(body.get_children())
                    if body_children:
                        first_stmt_col = body_children[0].extent.start.column
                        macro_indent_str = ' ' * (first_stmt_col - 1)
                    else:
                        closing_brace_col = body.extent.end.column
                        macro_indent_str = ' ' * (closing_brace_col - 1 + 4)

                    closing_brace_indent_str = ' ' * (body.extent.end.column - 1)

                    points.append(
                        (entry_offset, exit_offset, macro_indent_str, closing_brace_indent_str)
                    )

        points.extend(find_instrumentation_points(node, source_file))
        
    return points

def main():
    if len(sys.argv) < 2:
        print("Usage: python instrument.py <c_file_path> [clang_args...]", file=sys.stderr)
        sys.exit(1)

    source_file = sys.argv[1]
    clang_args = sys.argv[2:]

    if not os.path.exists(source_file):
        print(f"Error: File not found at '{source_file}'", file=sys.stderr)
        sys.exit(1)

    try:
        index = clang.cindex.Index.create()
        tu = index.parse(source_file, args=clang_args)
        
        has_errors = False
        for diag in tu.diagnostics:
            if diag.severity >= clang.cindex.Diagnostic.Error:
                print(f"Parsing Error: {diag.spelling}", file=sys.stderr)
                has_errors = True
        
        if has_errors:
            print("\nInstrumentation aborted due to parsing errors.", file=sys.stderr)
            print("Please provide correct include paths (-I, -isystem) if needed.", file=sys.stderr)
            sys.exit(1)

    except clang.cindex.TranslationUnitLoadError as e:
        print(f"Error: Could not load the translation unit. Details: {e}", file=sys.stderr)
        sys.exit(1)

    instrument_points = find_instrumentation_points(tu.cursor, source_file)

    if not instrument_points:
        print("No 'wgpu' function definitions found to instrument.")
        return

    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read()

    modifications = []
    for entry_off, exit_off, macro_indent, brace_indent in instrument_points:
        entry_text = f"{macro_indent}ENTRY();\n"
        exit_text = f"{macro_indent}EXIT();\n{brace_indent}"

        modifications.append((entry_off - 1, entry_text))
        modifications.append((exit_off - 2, exit_text))

    modifications.sort(key=lambda x: x[0], reverse=True)

    for offset, text in modifications:
        content = content[:offset] + text + content[offset:]

    base, ext = os.path.splitext(source_file)
    output_path = f"{base}_instrumented{ext}"
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Successfully wrote instrumented file to '{output_path}'")
    except IOError as e:
        print(f"Error writing to file '{output_path}': {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()