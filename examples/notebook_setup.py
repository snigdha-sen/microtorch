def setup_notebook_environment():
    import sys
    from pathlib import Path

    project_root = Path.cwd().parent.resolve()

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    simulation_path = project_root / "simulation_data"

    return project_root, simulation_path



