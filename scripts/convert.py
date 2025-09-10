import json
import numpy as np


def main():
    data = np.load("tts/voices.bin", allow_pickle=True)
    voices_dict = data.item()

    with open("tts/voices.json", "w") as f:
        json.dump(voices_dict, f, indent=2)
    print("Direct conversion successful")


if __name__ == "__main__":
    main()
