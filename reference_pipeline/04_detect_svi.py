# ---------------------------------------------------------------------------
# REFERENCE ONLY -- not runnable from this repository.
#
# This script belongs to the upstream pipeline that produced the analysis
# dataset. It requires a Google Street View API key and/or the panorama store
# held on the lab server, neither of which is redistributable. It is included
# to document how the published features were derived, not to be re-executed.
#
# Set PERSONALITY_SVI_ROOT and SVI_IMAGE_DIR if you adapt it to your own data.
# ---------------------------------------------------------------------------

from zensvi.cv import ObjectDetector
import os

SVI_IMAGE_DIR = os.environ.get("SVI_IMAGE_DIR", "/path/to/streetview")

detector = ObjectDetector(
    text_prompt="American flag .",  # specify the object(s) (e.g., single type: "building", multi-type: "car . tree")
    box_threshold=0.45,
    text_threshold=0.25
)
place_list = ["Austin, Texas", "Dallas, Texas", "Houston, Texas", "San Antonio, Texas"]
for place in place_list:
    place_clean = place.lower().replace(", ", "_").replace(" ", "_")
    # get a list of all the folders in f"data/raw/{place_clean}/gsv_panorama/"
    # folders_list = os.listdir(f"data/raw/{place_clean}/gsv_panorama/")
    folders_list = os.listdir(f"{SVI_IMAGE_DIR}/{place_clean}/gsv_panorama/")
    for folder in folders_list:
        # make sure to create the output directory
        os.makedirs(f"data/processed/detection/{place_clean}/{folder}/", exist_ok=True)
        # input_dir = f"data/raw/{place_clean}/gsv_panorama/{folder}/"
        input_dir = f"{SVI_IMAGE_DIR}/{place_clean}/gsv_panorama/{folder}/"
        output_dir = f"data/processed/detection/{place_clean}/{folder}/"
        output_image_dir = f"data/processed/detection/{place_clean}/{folder}/images/"

        detector.detect_objects(
            dir_input=input_dir,
            dir_summary_output=output_dir,
            # dir_image_output=output_image_dir,
            group_by_object = True,
            max_workers= 8,
            save_format="csv" # or "csv"
        )