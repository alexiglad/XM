## Instructions for downloading each of the datasets

### Note on `ffprobe` for Computer Vision datasets

We use ffprobe to read the duration of each video in our video dataloaders. While opencv-python does not require a library installation external to Python, we found opencv to be more unreliable than ffprobe at reading video durations.

#### ffprobe setup: 
On a linux system, you can `apt install ffmpeg` to get ffprobe. If ffprobe is in your path, the training scripts should run without any additional setup for ffprobe.

Or if you have a conda env you can do `conda install -c conda-forge ffmpeg` and see if that works.

If none of those work, you can download ffprobe at the [ffmpeg download site](https://ffmpeg.org//download.html). Once downloaded, extract the binaries and provide the path to the ffprobe file by either setting:
- the environment variable `FFPROBE_PATH=<path_to_ffprobe>`
- or the command-line argument `--ffprobe_path=<path_to_ffprobe>`

### Kinetics-400

To download Kinetics-400, use the scripts at https://github.com/cvdfoundation/kinetics-dataset. Set the command-line argument `--dataset_dir=<path_to_dataset>`. Then set the `$K400_DIR` env variable.

### Something-something-v2
SSv2 is distributed by Qualcomm in separate files at [this link](https://www.qualcomm.com/developer/software/something-something-v-2-dataset/downloads). Download the video files and the labels. After downloading all the video files, you can concatenate them and unzip them as a single file:

```
cat 20bn-something-something-v2-* > ssv2_archive
tar -xvf ssv2_archive
```
Also unzip the labels: ``unzip 20bn-something-something-download-package-labels.zip``
Then, set the command-line argument `--dataset_dir=<path_to_dataset>`, or preferably set the `$SSV2_DIR` env variable.
