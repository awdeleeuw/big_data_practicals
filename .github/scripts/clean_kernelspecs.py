import nbformat
import glob

for nb_path in glob.glob("**/*.ipynb", recursive=True):
    with open(nb_path) as f:
        nb = nbformat.read(f, as_version=4)
    nb['metadata']['kernelspec'] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python"
    }
    with open(nb_path, 'w') as f:
        nbformat.write(nb, f)

