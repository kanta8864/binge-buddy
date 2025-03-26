from binge_buddy.front_end import FrontEnd

def main_semantic():
    front_end = FrontEnd(mode="semantic")
    front_end.run_flask()

def main_episodic():
    front_end = FrontEnd(mode="episodic")
    front_end.run_flask()

if __name__ == "__main__":
    main_semantic()
    # main_episodic()
