import cv2
import glob
import os
import sys

# Point this at a folder of ARCADE test images (stenosis test_case_segmentation/images works well)
IMAGE_FOLDER = sys.argv[1] if len(sys.argv) > 1 else r"F:\cardiosense\ml-training\vessel_analysis\data\stenosis\test_case_segmentation\images"
OUTPUT_VIDEO = sys.argv[2] if len(sys.argv) > 2 else r"F:\cardiosense\test_angiogram.mp4"
NUM_FRAMES = 30
FPS = 10

images = sorted(glob.glob(os.path.join(IMAGE_FOLDER, "*.png")))[:NUM_FRAMES]

if not images:
    print(f"No images found in {IMAGE_FOLDER}")
    sys.exit(1)

print(f"Found {len(images)} images, building video...")

first_frame = cv2.imread(images[0])
h, w = first_frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (w, h))

for img_path in images:
    frame = cv2.imread(img_path)
    if frame.shape[:2] != (h, w):
        frame = cv2.resize(frame, (w, h))
    writer.write(frame)

writer.release()
print(f"Test video saved to: {OUTPUT_VIDEO}")
print(f"{len(images)} frames at {FPS} fps (~{len(images)/FPS:.1f} seconds)")
