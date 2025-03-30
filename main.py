import argparse
import logging
import uuid

from colorlog import ColoredFormatter

from binge_buddy.front_end import FrontEnd


def configure_logging(log_file="app.log"):
    # Console handler with color
    console_handler = logging.StreamHandler()
    console_formatter = ColoredFormatter(
        "%(log_color)s[%(asctime)s] %(levelname)s in %(module)s:%(reset)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        },
    )
    console_handler.setFormatter(console_formatter)

    # File handler (no color)
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    # Root logger setup
    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler, file_handler],
    )

    # Optional: silence noisy loggers (like requests, urllib3)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def generate_session_id():
    return str(uuid.uuid4())


def main(mode, user_id, session_id):
    configure_logging()
    front_end = FrontEnd(mode=mode, user_id=user_id, session_id=session_id)
    front_end.run_flask()


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Start the Binge Buddy FrontEnd.")

    # Add arguments to the parser
    parser.add_argument(
        "--mode",
        choices=["semantic", "episodic"],
        required=True,
        help="Set the mode for the FrontEnd (semantic or episodic)",
    )
    parser.add_argument("--user", required=True, help="Provide the user ID string")

    # Parse arguments
    args = parser.parse_args()

    session_id = str(generate_session_id())

    main(mode=args.mode, user_id=args.user, session_id=session_id)
