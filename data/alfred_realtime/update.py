import sys

from download import main

if __name__ == "__main__":
    if "--update" not in sys.argv:
        sys.argv.append("--update")
    main()
