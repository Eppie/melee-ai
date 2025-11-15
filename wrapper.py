#!/usr/bin/env python3
"""
Wrapper script that instruments None checks in Python code.
Usage: python wrapper.py target_script.py [args...]
"""

import ast
import os
import runpy
import sys
import tempfile
import traceback
from pathlib import Path

# Project root - only instrument files within this directory
PROJECT_ROOT = Path("/Users/eppie/PycharmProjects/nano-melee").resolve()


class NoneCheckInstrumenter(ast.NodeTransformer):
    """AST transformer that instruments 'is None' and 'is not None' checks."""

    def __init__(self, filename):
        self.filename = filename
        self.counter = 0

    def _should_instrument(self, node):
        """Determine if a comparison should be instrumented."""
        # Check if it's a comparison with 'is None' or 'is not None'
        if not isinstance(node, ast.Compare):
            return False

        # Must have exactly one comparator
        if len(node.ops) != 1 or len(node.comparators) != 1:
            return False

        op = node.ops[0]
        comparator = node.comparators[0]

        # Check if comparing with None using 'is' or 'is not'
        if not (isinstance(op, (ast.Is, ast.IsNot)) and
                isinstance(comparator, ast.Constant) and
                comparator.value is None):
            return False

        return True

    def _get_var_name(self, node):
        """Extract variable name from the comparison node."""
        left = node.left
        if isinstance(left, ast.Name):
            return left.id
        elif isinstance(left, ast.Attribute):
            # For things like obj.attr, get full dotted name
            parts = []
            current = left
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return '.'.join(reversed(parts))
        elif isinstance(left, ast.Subscript):
            # For subscripts like dict[key], approximate
            return ast.unparse(left)
        else:
            return ast.unparse(left)

    def _create_check_code(self, var_name, check_type, original_node):
        """Create the instrumentation code as AST nodes."""
        self.counter += 1
        check_id = f"check_{self.counter}"

        # Get line number from original node
        lineno = getattr(original_node, 'lineno', 1)
        col_offset = getattr(original_node, 'col_offset', 0)

        # Create a helper function to pretty print locals
        # This function will be defined inline and then called
        pretty_print_locals_code = """
def _pretty_print_locals(locals_dict):
    '''Pretty print locals, one per line, truncated to single line.'''
    for key, value in sorted(locals_dict.items()):
        # Skip the pretty print function itself
        if key == '_pretty_print_locals':
            continue
        try:
            str_val = str(value)
            # Replace newlines with spaces and truncate if too long
            str_val = str_val.replace('\\n', ' ').replace('\\r', ' ')
            max_len = 200
            if len(str_val) > max_len:
                str_val = str_val[:max_len] + '...'
            print(f'  {key} = {str_val}', file=sys.stdout, flush=True)
        except Exception:
            print(f'  {key} = <unprintable>', file=sys.stdout, flush=True)
"""
        
        # Parse the helper function
        pretty_print_tree = ast.parse(pretty_print_locals_code)
        pretty_print_func = pretty_print_tree.body[0]
        ast.fix_missing_locations(pretty_print_func)
        
        # Create a check key for tracking (filename, line_number, variable_name)
        check_key = f"{self.filename}:{lineno}:{var_name}"
        # Escape the check_key for use in Python code
        check_key_repr = repr(check_key)
        
        # The print statement we want to inject
        # We only print if the variable IS None AND we haven't printed this check before
        # First, check if we should print (only once per unique check)
        # Use globals() for tracking (works in all contexts)
        check_tracking_code = f"""
# Track which checks have been printed (using globals())
if '_none_check_printed' not in globals():
    globals()['_none_check_printed'] = set()

check_key = {check_key_repr}
_none_check_printed = globals()['_none_check_printed']
if check_key not in _none_check_printed:
    _none_check_printed.add(check_key)
    _should_print = True
else:
    _should_print = False
"""
        
        # Parse the tracking code
        tracking_tree = ast.parse(check_tracking_code)
        tracking_stmts = tracking_tree.body
        for stmt in tracking_stmts:
            ast.fix_missing_locations(stmt)
        
        # The print statement we want to inject (wrapped in if _should_print)
        print_header = ast.Expr(
            value=ast.Call(
                func=ast.Name(id='print', ctx=ast.Load()),
                args=[
                    ast.JoinedStr(values=[
                        ast.Constant(value=f"\n{'=' * 60}\n"),
                        ast.Constant(value=f"[NONE-CHECK] File: {self.filename}\n"),
                        ast.Constant(value=f"Line: {lineno}\n"),
                        ast.Constant(value=f"Variable: {var_name}\n"),
                        ast.Constant(value=f"Check type: {check_type}\n"),
                        ast.Constant(value=f"Locals:\n"),
                    ])
                ],
                keywords=[]
            ),
            lineno=lineno,
            col_offset=col_offset
        )
        
        # Call the pretty print function
        print_locals = ast.Expr(
            value=ast.Call(
                func=ast.Name(id='_pretty_print_locals', ctx=ast.Load()),
                args=[
                    ast.Call(
                        func=ast.Name(id='locals', ctx=ast.Load()),
                        args=[],
                        keywords=[]
                    )
                ],
                keywords=[]
            ),
            lineno=lineno,
            col_offset=col_offset
        )
        
        # Wrap print statements in if _should_print
        print_block = ast.If(
            test=ast.Name(id='_should_print', ctx=ast.Load()),
            body=[print_header, print_locals],
            orelse=[],
            lineno=lineno,
            col_offset=col_offset
        )
        ast.fix_missing_locations(print_block)
        
        # Combine: tracking code, define function, then conditional print statements
        print_call = tracking_stmts + [pretty_print_func, print_block]

        # Use fix_missing_locations on each statement
        for stmt in print_call:
            ast.fix_missing_locations(stmt)
            # Ensure end_lineno is valid (must be >= lineno)
            if hasattr(stmt, 'end_lineno') and stmt.end_lineno and stmt.end_lineno < stmt.lineno:
                stmt.end_lineno = stmt.lineno

        return print_call

    def visit_If(self, node):
        """Visit If nodes and instrument None checks."""
        # First, recursively visit child nodes
        node = self.generic_visit(node)

        # Check if the test is a None comparison we should instrument
        if self._should_instrument(node.test):
            var_name = self._get_var_name(node.test)
            op = node.test.ops[0]

            # Determine check type
            is_none_check = isinstance(op, ast.Is)
            check_type = "is None" if is_none_check else "is not None"

            # Create a conditional that only logs when the variable IS None
            # For "if X is None", we log inside the if block
            # For "if X is not None", we need to check in an else or with a negated condition

            if is_none_check:
                # If checking "is None", add logging at start of if body
                log_stmts = self._create_check_code(var_name, check_type, node.test)
                node.body = log_stmts + node.body
                return node
            else:
                # If checking "is not None", we need to log when it IS None
                # We'll wrap with a check: if X is None: log()
                var_node = node.test.left

                # Create: if var_name is None: log()
                compare_node = ast.Compare(
                    left=var_node,
                    ops=[ast.Is()],
                    comparators=[ast.Constant(value=None)]
                )
                # Set location for Compare node
                compare_node.lineno = node.test.lineno
                compare_node.col_offset = node.test.col_offset
                
                none_check = ast.If(
                    test=compare_node,
                    body=self._create_check_code(var_name, check_type, node.test),
                    orelse=[],
                    lineno=node.lineno,
                    col_offset=node.col_offset
                )

                # Fix location info - ensure end_lineno is valid
                ast.fix_missing_locations(none_check)
                # Ensure end_lineno is valid (must be >= lineno)
                if hasattr(none_check, 'end_lineno') and none_check.end_lineno and none_check.end_lineno < none_check.lineno:
                    none_check.end_lineno = none_check.lineno

                # We can't return multiple statements from a single node visit
                # Instead, we'll add the check at the beginning of the parent block
                # For now, just instrument inline with a try-except wrapper
                # Better approach: add logging inside an else branch

                # Actually, let's keep it simple: add check to else branch
                if node.orelse:
                    # There's already an else/elif, prepend to it
                    if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                        # It's an elif, can't easily inject here
                        pass
                    else:
                        # Regular else block
                        node.orelse = self._create_check_code(var_name, check_type, node.test) + node.orelse
                else:
                    # No else block, create one with just the logging
                    node.orelse = self._create_check_code(var_name, check_type, node.test)

        return node

    def visit_While(self, node):
        """Visit While nodes - similar to If but we're more careful."""
        # For while loops, we generally don't want to instrument
        # as they can create excessive logging
        node = self.generic_visit(node)
        return node


def instrument_code(source_code, filename):
    """Instrument the source code with None-check logging."""
    try:
        tree = ast.parse(source_code, filename=filename)
        instrumenter = NoneCheckInstrumenter(filename)
        new_tree = instrumenter.visit(tree)

        # Fix missing locations
        ast.fix_missing_locations(new_tree)

        # Validate and fix all line and column ranges
        def validate_ranges(node):
            """Ensure all nodes have valid line and column ranges."""
            for child in ast.walk(node):
                # Fix line ranges (end_lineno must be >= lineno)
                if hasattr(child, 'lineno') and hasattr(child, 'end_lineno'):
                    if child.end_lineno and child.end_lineno < child.lineno:
                        child.end_lineno = child.lineno
                # Fix column ranges (end_col_offset must be >= col_offset)
                if hasattr(child, 'col_offset') and hasattr(child, 'end_col_offset'):
                    if child.end_col_offset is not None and child.col_offset is not None:
                        if child.end_col_offset < child.col_offset:
                            child.end_col_offset = child.col_offset

        validate_ranges(new_tree)

        # Return source code as string instead of compiled code
        return ast.unparse(new_tree)
    except SyntaxError as e:
        print(f"Syntax error in {filename}: {e}", file=sys.stdout, flush=True)
        raise


def should_instrument_file(filepath):
    """Determine if a file should be instrumented based on its location."""
    try:
        resolved = Path(filepath).resolve()
        # Check if file is within project root
        return resolved.is_relative_to(PROJECT_ROOT)
    except (ValueError, OSError):
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python wrapper.py target_script.py [args...]", file=sys.stdout, flush=True)
        sys.exit(1)

    target_script = sys.argv[1]

    if not os.path.exists(target_script):
        print(f"Error: {target_script} not found", file=sys.stdout, flush=True)
        sys.exit(1)

    # Set up the execution environment
    # Update sys.argv to remove wrapper.py
    sys.argv = sys.argv[1:]

    # Change working directory to target script's directory
    target_dir = os.path.dirname(os.path.abspath(target_script))
    if target_dir:
        os.chdir(target_dir)

    # Add target directory to path
    sys.path.insert(0, target_dir if target_dir else '.')

    # Check if we should instrument this file
    if not should_instrument_file(target_script):
        print(f"Note: {target_script} is outside project root, running without instrumentation", file=sys.stdout, flush=True)
        # Just run it normally using runpy
        try:
            runpy.run_path(target_script, run_name='__main__')
        except Exception as e:
            print(f"\nError executing {target_script}: {e}", file=sys.stdout, flush=True)
            traceback.print_exc()
            sys.exit(1)
    else:
        print(f"Instrumenting {target_script} for None-check logging...", file=sys.stdout, flush=True)
        # Read and instrument the code
        with open(target_script, 'r') as f:
            source_code = f.read()

        instrumented_code = instrument_code(source_code, target_script)

        # Write instrumented code to a temporary file and execute it
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp_file:
            tmp_file.write(instrumented_code)
            tmp_file_path = tmp_file.name

        try:
            # Execute the temporary file using runpy
            runpy.run_path(tmp_file_path, run_name='__main__')
        except Exception as e:
            print(f"\nError executing {target_script}: {e}", file=sys.stdout, flush=True)
            traceback.print_exc()
            sys.exit(1)
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except OSError:
                pass


if __name__ == '__main__':
    main()