# Digital Forensics Project - 18/19
## Project 1 - Packet classification for mobile applications
___

## Instructions

#### `clustering.py`

```bash
$ chmod +x clustering.py
$ ./clustering.py DATASET

# alternatively
$ python clustering.py DATASET
```

DATASET is a value taken from `{dropbox, evernote, facebook, gmail, gplus, twitter}`

DATASER can be set in the source code when debugging, just comment the lines

```python
if len(sys.argv) < 2:
    exit("Usage: ./clustering.py APPNAME")
else:
    ENV_TASK = sys.argv[1]
```

 and set 

```python
ENV_TASK = "facebook"
```

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
    exit("Usage: ./classifier.py APPNAME")
else:
    ENV_TASK = sys.argv[1]
```

 and set 

```python
ENV_TASK = "facebook"
```

___

## File and folder structure

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
- `utils.py`:  auxiliary script used for printing statistics and debugging purposes, can be ignored



____

## Referecenes

[1] Mauro Conti, Luigi V. Mancini, Riccardo Spolaor, and Nino Vincenzo Verde. 2015. Can't You Hear Me Knocking: Identification of User Actions on Android Apps via Traffic Analysis. In Proceedings of the 5th ACM Conference on Data and Application Security and Privacy (CODASPY '15). ACM, New York, NY, USA, 297-304. DOI: https://doi.org/10.1145/2699026.2699119

[2] Conti, M., Mancini, L. V., Spolaor, R., \& Verde, N. V. (2016). Analyzing android encrypted network traffic to identify user actions. IEEE Transactions on Information Forensics and Security, 11(1), 114-125.

[3] https://spritz.math.unipd.it/projects/analyzinguseractionsandroid/
