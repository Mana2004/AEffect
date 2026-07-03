import cv2
import os
import tkinter as tk
from App import FXCanvas


def load_assets():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    assets_dir = os.path.join(script_dir, 'image_filters', 'assets')

    prop_names = ['hat', 'mustache', 'glasses', 'butterfly']
    assets = {}

    print(f"Searching for assets in: {assets_dir}")

    for prop in prop_names:
        path = os.path.join(assets_dir, f"{prop}.png")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"WARNING: Failed to load {prop}.png at {path}")
        else:
            print(f"SUCCESS: Loaded {prop}.png (Shape: {img.shape})")

        assets[prop] = img

    return assets

def main():
    root = tk.Tk()

    print("Loading assets...")
    loaded_assets = load_assets()

    app = FXCanvas(root, "Computer Vision & Mathematical Engine V2", assets=loaded_assets)

    root.mainloop()


if __name__ == "__main__":
    main()