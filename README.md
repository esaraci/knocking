# :microscope: Digital Forensics Project - 18/19
Project 1 - Packet classification for mobile applications

> The large expansion of SSL/TLS in the last years has made it harder
for attackers to collect clear text information through packet sniffing or,
more in general, through network traffic analysis. The main reason for this
is that SSL/TLS encrypts the traffic between two endpoints, which means
that even though packets can still be easily captured, no useful information
can be inferred from the packet’s content without having the encryption
keys.
> The authors of [1] and [2] showed that by training a machine learning
algorithm with encrypted traffic data, one could correctly classify which
actions a user performed while using some of the most common Android
applications such as Facebook, Gmail, or Twitter. This could easily lead,
through correlation attacks, to the full deanonimization of fake, privacy
preserving identities.
> In this work I try to reproduce the results achieved in [1] and [2] by
implementing the classification model described in the papers.

For more information read the papers in \[1\] and \[2\] or my [REPORT](report/main.pdf).
___

## :notebook_with_decorative_cover: Instructions

#### `clustering.py`

```bash
$ chmod +x clustering.py
$ ./clustering.py DATASET

# alternatively
$ python clustering.py DATASET
```
:warning: Running `clustering.py` with more then 1000 flows might take a long time.

#### `classifier.py`

```bash
$ chmod +x 
$ ./classifier.py DATASET

# alternatively
$ python classifier.py DATASET
```

DATASET is a value taken from `{dropbox, evernote, facebook, gmail, gplus, twitter}`

DATASER can be set in the source code when debugging, just comment the lines

```python
if len(sys.argv) < 2:
    exit("Usage: ./{clustering, classifier}.py APPNAME")
else:
    ENV_TASK = sys.argv[1]
```

 and set 

```python
ENV_TASK = "facebook"
```
___

## :open_file_folder: File and folder structure

#### Folders

- `dataset/`: contains the datasets saved by `clustering.py` and used by `classifier.py`
- `images/`: images of the confusion matrixes for each task
- `report/`: `.tex` and `.pdf` file of the report
- `results/`: `.txt` files with *precision*, *recall*, and *f1 measure* for each task,



#### Files

- `README.md`: this file
- `apps_total_plus_filtered.csv`: original dataset, provided by [3]
- `classifier.py`: loads one of the dataset in  `datasets/` and performs classification with Random Forest
- `clustering.py`: loads the original dataset and produces the files cotained in `datasets/`
- `executor.py`: script used to execute `clustering.py` and `classifier.py`



____

## :books: Referecenes

[1] Mauro Conti, Luigi V. Mancini, Riccardo Spolaor, and Nino Vincenzo Verde. 2015. Can't You Hear Me Knocking: Identification of User Actions on Android Apps via Traffic Analysis. In Proceedings of the 5th ACM Conference on Data and Application Security and Privacy (CODASPY '15). ACM, New York, NY, USA, 297-304. DOI: https://doi.org/10.1145/2699026.2699119

[2] Conti, M., Mancini, L. V., Spolaor, R., \& Verde, N. V. (2016). Analyzing android encrypted network traffic to identify user actions. IEEE Transactions on Information Forensics and Security, 11(1), 114-125.

[3] https://spritz.math.unipd.it/projects/analyzinguseractionsandroid/
