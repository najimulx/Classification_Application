import streamlit as st
from aeroreach.ui.app import main


def run():
    # Streamlit will call this file. Delegate to the project's main() function.
    main()


if __name__ == "__main__":
    run()
