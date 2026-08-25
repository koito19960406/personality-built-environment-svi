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

from zensvi.download import GSVDownloader
import os 

api_key = os.environ["GSV_API_KEY"]
downloader = GSVDownloader(grid=True, grid_size = 100, gsv_api_key=api_key)
place_list = ["Austin, Texas", "Dallas, Texas", "Houston, Texas", "San Antonio, Texas"]
for place in place_list:
    place_clean = place.lower().replace(", ", "_").replace(" ", "_")
    output_dir = f"data/raw/{place_clean}/"
    # make sure to create the output directory
    os.makedirs(output_dir, exist_ok=True)
    # downloader.download_svi(output_dir,
    #                         input_place_name=place,
    #                         augment_metadata=True,
    #                         )
    downloader.update_metadata(input_pid_file=f"data/raw/{place_clean}/gsv_pids.csv",
                        output_pid_file=f"data/raw/{place_clean}/gsv_pids_augmented.csv",
                        verbosity=1
                        )
