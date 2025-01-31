"""
This script extracts function and module docstrings from Python files in a given directory 
and generates an HTML documentation file. 

Features:
- Recursively scans directories to collect documentation.
- Extracts both module-level and function-level docstrings.
- Generates an interactive HTML file with expandable sections for better readability.
"""

import os
import ast

def extract_function_docs_from_file(file_path):
    """
    Extracts module and function docstrings from a Python file.
    
    Args:
        file_path (str): Path to the Python file.
    
    Returns:
        dict: A dictionary containing the module docstring and a list of function docstrings.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as file:
            content = file.read()
            tree = ast.parse(content)
    except SyntaxError as e:
        print(f"❌ Syntax error in file: {file_path}\nError: {e}")
        return {}
    
    module_docstring = ast.get_docstring(tree) or "No module documentation provided."

    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):  
            function_name = node.name
            args = [arg.arg for arg in node.args.args]
            docstring = ast.get_docstring(node) or "No documentation provided."
            functions.append({
                "name": function_name,
                "args": args,
                "docstring": docstring
            })

    return {
        "module_docstring": module_docstring,
        "functions": functions
    }

def extract_function_docs_from_directory(directory):
    """
    Traverses a directory and extracts docstrings from Python files.
    Args:
        directory (str): Path to the directory to traverse.
    Returns:
        dict: Tree-like dictionary representing directories, files, and their functions.
    """
    EXCLUDED_DIRS = {"articles_KGG", ".git", "models", "my_env", "__pycache__", "articles", "results", "data"}  

    def build_tree(current_path):
        tree = {}
        for entry in sorted(os.listdir(current_path)):
            full_path = os.path.join(current_path, entry)
            if os.path.isdir(full_path):
                if entry in EXCLUDED_DIRS: 
                    continue  
                tree[entry] = build_tree(full_path)
            elif entry.endswith(".py"):
                tree[entry] = extract_function_docs_from_file(full_path)
        return tree

    return build_tree(directory)

def generate_html_documentation(tree, output_file):
    """
    Generates an HTML file containing module documentation in a directory tree structure.
    Args:
        tree (dict): Directory tree with files and functions.
        output_file (str): Path to the output HTML file.
    """
    def render_tree(tree, parent_id="root"):
        html = "<ul>"
        for key, value in tree.items():
            node_id = f"{parent_id}_{key.replace('.', '_').replace('/', '_')}"

            if isinstance(value, dict) and "functions" not in value:  
                html += f"<li><span onclick=\"toggleVisibility('{node_id}')\" style=\"cursor: pointer; color: #0056b3;\">📁 {key}</span>"
                html += f"<div id='{node_id}' style='display: none; margin-left: 20px;'>"
                html += render_tree(value, node_id)
                html += "</div></li>"

            elif isinstance(value, dict) and "functions" in value:  
                html += f"<li><span onclick=\"toggleVisibility('{node_id}')\" style=\"cursor: pointer; color: #0056b3;\">📄 {key}</span>"
                html += f"<div id='{node_id}' style='display: none; margin-left: 20px;'>"

                html += f"<div class='module-docstring'><strong>Module Documentation:</strong><pre>{value['module_docstring']}</pre></div>"

                for func in value.get("functions", []):
                    html += f"<div class='function'>"
                    html += f"<h3>{func['name']}({', '.join(func['args'])})</h3>"
                    html += f"<div class='docstring'><pre>{func['docstring']}</pre></div>"
                    html += "</div>"

                html += "</div></li>"

        html += "</ul>"
        return html

    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Module Documentation</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }
            h1 { color: #333; }
            ul { list-style-type: none; padding-left: 20px; }
            .function { margin-bottom: 20px; }
            .function h3 { color: #333; }
            .docstring { background: #f9f9f9; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        </style>
        <script>
            function toggleVisibility(elementId) {
                const element = document.getElementById(elementId);
                if (element.style.display === "none") {
                    element.style.display = "block";
                } else {
                    element.style.display = "none";
                }
            }
        </script>
    </head>
    <body>
        <h1>Module Documentation</h1>
        """ + render_tree(tree) + """
    </body>
    </html>
    """

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(html_content)

    print(f"HTML documentation generated: {output_file}")

# Example usage
directory = "./"
output_file = "documentation.html"

modules_tree = extract_function_docs_from_directory(directory)
generate_html_documentation(modules_tree, output_file)