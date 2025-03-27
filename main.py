import argparse
import uuid

from binge_buddy.front_end import FrontEnd


def generate_session_id():
    return str(uuid.uuid4())


def main(mode, user_id, session_id):
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
