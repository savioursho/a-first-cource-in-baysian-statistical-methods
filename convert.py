from pathlib import Path
from tqdm.auto import tqdm
import subprocess

notebooks_dir = Path("./notebooks").resolve()
list_notebooks = list(str(path) for path in notebooks_dir.rglob("*.ipynb"))


def convert_notebooks(to: str, output_dir: str):
    args = [
        "jupyter",
        "nbconvert",
        " ".join(list_notebooks),
        "--to",
        to,
        "--output-dir",
        output_dir,
    ]
    subprocess.run(args)


def main():
    convert_notebooks(to="html", output_dir="html")
    # convert_notebooks(to="markdown", output_dir="markdown")


if __name__ == "__main__":
    main()
