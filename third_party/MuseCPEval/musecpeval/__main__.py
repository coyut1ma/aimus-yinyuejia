"""``python -m musecpeval`` — same CLI as the ``musecpeval`` console script."""
from .runner import main

if __name__ == "__main__":
    raise SystemExit(main())
