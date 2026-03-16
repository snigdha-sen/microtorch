def setup_notebook_environment():
    import sys
    from pathlib import Path

    project_root = Path.cwd().parent.resolve()

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))



def add_project_path_to_notebook():
    import sys
    from pathlib import Path

    sys.path.append(str(Path.cwd().parent))


