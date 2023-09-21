README.txt

Configuration Guide for Windows Installation

1. Python Installation
    Please install python version 3.11.4
    Refer pytohon official homepage(https://www.python.org)

2. Virtual Environment Setting(Optional)
    If you want to make virtural environment, please use Anaconda or athoer software

3. Package Installation
Followings are list of package I used in this repository. I recommend use pip instead of conda.
There were errors during installation of package on MAC os

Before installing packages, updata pip

Please install those packages's latest version

pip install torch
pip install torchvision
pip install numpy
pip install tqdm
pip install pandas
pip install PyWavelets
pip install jupyter

4. Download My Repository Files

5. Put RAW Data Files in "./Data/raw_data"

You can download "Preprocessed data in Python Format" from https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html
Put files from s01.dat ~ s32.dat in raw_data directory.

6. Please Excute "dat_to_image.py"
**train set is data of participant 1~24
**test set isi data of participant 25~32
"dat_to_image.py" extracts features form EEG data and transform it to 9 x 9, 3 channel image.

7. You can RUN four .ipynb files!!!

