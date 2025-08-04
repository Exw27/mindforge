import argparse

def get_parser():
    """Creates and returns the argument parser."""
    parser = argparse.ArgumentParser(description="A tool like Ollama, but in Python with Hugging Face Transformers.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a model")
    run_parser.add_argument("model", type=str, help="The name of the model to run")
    run_parser.add_argument("prompt", type=str, nargs='?', default=None, help="Optional one-shot prompt; omit to start chat")
    run_parser.add_argument("--html", action="store_true", help="Allow HTML/markup in outputs")

    # Pull command
    pull_parser = subparsers.add_parser("pull", help="Pull a model from the Hugging Face Hub")
    pull_parser.add_argument("model", type=str, help="The name of the model to pull")

    # List command
    list_parser = subparsers.add_parser("list", help="List all downloaded models")

    # Remove command
    rm_parser = subparsers.add_parser("rm", help="Remove a downloaded model")
    rm_parser.add_argument("model", type=str, help="The name of the model to remove")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Serve an OpenAI-compatible API for all models")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create a custom model from a Modelfile")
    create_parser.add_argument("model", type=str, help="The name for the new custom model")
    create_parser.add_argument("-f", "--file", type=str, required=True, help="The path to the Modelfile")

    return parser
