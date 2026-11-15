"""Shared helpers for building & executing OIBSIP notebooks."""
import os
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


def md(src):
    """Create a markdown cell."""
    return nbf.v4.new_markdown_cell(src)


def code(src):
    """Create a code cell."""
    return nbf.v4.new_code_cell(src)


def build_notebook(cells, title=None):
    """Assemble a list of cell objects into an nbformat notebook."""
    nb = nbf.v4.new_notebook()
    nb.cells = list(cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {
        "name": "python",
        "version": "3.10+",
    }
    nb.metadata["nbformat"] = 4
    nb.metadata["nbformat_minor"] = 5
    return nb


def execute(notebook, save_path, timeout=900):
    """Execute a notebook in-place and write it to disk.

    ``notebook`` must be an nbformat Notebook object. ``save_path`` is where
    the executed notebook gets written; its parent directory becomes the
    kernel working directory so relative paths in the notebook resolve the
    same way they do when a user opens the notebook in Jupyter from there.
    """
    if not isinstance(notebook, nbf.NotebookNode):
        raise TypeError(
            "execute() expects an nbformat Notebook, got " + type(notebook).__name__
        )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    client = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(save_path.parent)}},
    )
    try:
        client.execute()
    except Exception:
        # Write the partial notebook so the failing cell can be inspected.
        nbf.write(notebook, str(save_path))
        print("EXECUTION FAILED — partial notebook written to", save_path)
        raise

    nbf.write(notebook, str(save_path))
    print("Saved executed notebook:", save_path)
    return notebook
