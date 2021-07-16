# Steps to run the model
## Setup the dataset
Place the dataset in the current working directory where the model will be run <br>
The dataset can be downloaded from [here](https://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/)
>Make sure each character set is present in a separate folder and images are in **png** format<br>
>Example: For English alphabet set, there has to be 26 folders where each folder corresponds to each alphabet set

## Installing the dependencies
Install the required python packages by going to the root of the directory and executing the following command<br>
`pip install -r requirements.txt`

## Preprocessing the data
To run the *TFrecord.py* for generating tfrecords, execute the following command in terminal
<br>
`python3 TFrecord.py`
<br>

## Training the model
To train the model, execute *train.py* with the following command
<br>
```python3 train.py --epoch 300 --batchsize 128 --maxitr 100000 --checkpoint 0```
<br>
> Arguments list
> * epoch: The number of passes of the entire training dataset
> * batchsize: The amount of images that are grouped into single batch.
> * maxitr: The maximum number of iterations for which the model should run
> * checkpoint: The latest checkpoint from where the model starts to train (This will restore all the trained variables into the model)

## Image completion
1. Start the interface by running *gui.py*
<br>
`python3 gui.py`
2. Select the required mask type from the dropdown list and browse the image file which has to be completed.
3. Click on **Run** and check the status in the progress bar.
4. After completion, the image will be displayed in the interface and it will be stored at *\<CurrentWorkingDirectory\>\ObjectConstruction\complete\\* folder
<br>
5. Click on **Exit** to close the interface.
<br>

## Directory Structure
```
GAN
├── readme.md
├── requirements.txt            # essential packages to run the model
├── ObjectConstruction
│   └── checkpoints             # stores all the checkpoints
│   └── complete                # stores the completed image
│   └── complete_src            # stores the image to be completed
│   └── images                  # stores intermediate training images, loss plot image
│   └── gan.py                  # generator and discriminator code
│   └── gui.py                  # interface code
│   └── main.py                 # image completion code
│   └── train.py                # training code
│   └── utils.py                # utility functions like psnr, image difference and image similarity
│   └── helping_functions.py    # functions for saving, cropping and finding inverse of images
└── TFrecord_generator
    ├── dataset                 # contains the dataset
    └── TFrecord_Files          # contains generated tfrecord files
    └── TFrecord.py             # code for generating tfrecords
```
