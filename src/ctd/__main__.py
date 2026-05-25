"""Allow `python -m ctd` as an alias for the installed `ctd` entry point."""

from .cli import app

if __name__ == "__main__":
    app()
