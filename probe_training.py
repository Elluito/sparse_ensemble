# https://github.com/delve-team/phd-lab/blob/master/phd_lab/experiments/probe_training.py


PROBE_PERFORMANCE_SAVEFILE = "probe_performances.csv"



import os
import numpy as np
import pandas as pd
from typing import List, Tuple
from itertools import product
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression as LogisticRegressionModel
import pickle
from multiprocessing import Pool
from attr import attrs, attrib
from phd_lab.experiments.utils import config
from joblib import Parallel, delayed
from joblib import Memory
if not os.path.exists(".memcache"):
    os.makedirs(".memcache")
memory = Memory(".memcache", verbose=0)


@attrs(auto_attribs=True, slots=True)
class PseudoArgs:
    """The pseudo args configurng the training of a probe.
    Args:
        model_name: The name of the model
        folder:     The folder containing the latent representation
        mp:         Number of processes to use of multiprocessing
        overwrite:  Overwrite existing results
    """
    model_name: str
    folder: str
    mp: int
    save_path: str = attrib(init=False)
    overwrite: bool = False
    verbose: int = 100

    def __attrs_post_init__(self):
        self.save_path = open(os.path.join(self.folder, "model_pointer.txt"), "r").read()


def filter_files_by_string_key(files: List[str], key: str) -> List[str]:
    """Filter files by a substring

    Args:
        files: list of filepaths
        key:   the substring the file most contain in order to be not filtered
    Returns:
         The filtered list of paths
    """
    return [file for file in files if key in file]


def seperate_labels_from_data(files: List[str]) -> Tuple[List[str], List[str]]:
    """Seperate taining data files and label files.

    Args:
        files: the list of files to seperate
    Returns:
        data files and label files in this order.
    """
    data_files = [file for file in files if '-labels' not in file]
    label_file = [file for file in files if '-labels' in file]
    return data_files, label_file


def get_all_npy_files(folder: str) -> List[str]:
    """Get all npy-files from a folder.
    Args:
        folder:     The target folder.
    Returns:
         The npy-files as full filepaths
    """
    all_files = os.listdir(folder)
    filtered_files = filter_files_by_string_key(all_files, '.p')
    full_paths = [os.path.join(folder, file) for file in filtered_files]
    return full_paths


def obtain_all_dataset(folder: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Build the training and test datasets from all available latent representations.
    Args:
        folder:     The folder containing labels and latent representations
    Returns:
        training and test datasets
    """
    all_files = get_all_npy_files(folder)
    data, labels = seperate_labels_from_data(all_files)
    train_data, train_label = filter_files_by_string_key(data, 'train-'), filter_files_by_string_key(labels, 'train-')
    eval_data, eval_label = filter_files_by_string_key(data, 'eval-'), filter_files_by_string_key(labels, 'eval-')
    train_set = [elem for elem in product(train_data, train_label)]
    eval_set = [elem for elem in product(eval_data, eval_label)]
    return train_set, eval_set


def loadall(filename: str) -> np.ndarray:
    """Load a pickle file batch-wise
    Args:
        filename:   the filename to load
    Returns:
        the data as numpy-array
    """
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def load(filename: str) -> np.ndarray:
    """Load a large pickle file batch-wise  and append the data.
    Args:
        filename:   the filenam to load the data from.
    Returns:
        the data as numpy array
    """
    return np.vstack([batch for batch in loadall(filename)])


def get_data_annd_labels(data_path: str, label_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load the dataset and labels ready for training.
    Args:
        data_path:      The path the data is stored
        label_path:     The path to the label data
    Returns:
        training data and labels, ready for training
    """
    return load(data_path), np.squeeze(load(label_path))

@memory.cache
def fit_with_cache(data: np.ndarray, labels: np.ndarray, verbose: int = 100):
    model = LogisticRegressionModel(
        multi_class='multinomial', n_jobs=12, solver='saga', verbose=verbose
    ).fit(data, labels)
    return model


def train_model(data_path: str, labels_path: str, verbose: int = 0) -> LogisticRegressionModel:
    """Train a logistic regression model on latent representations and labels from the original dataset.
    Args:
        data_path:      the training data
        labels_path:    the labels
    Returns:
        A logistic regression model fitted on the provided data
    """
    print('Loading training data from', data_path)
    data, labels = get_data_annd_labels(data_path, labels_path)
    print('Training data obtained with shape', data.shape, "Starting training für verbosity", verbose)
    return fit_with_cache(data, labels, verbose=verbose)



def obtain_accuracy(model: LogisticRegressionModel, data_path, label_path: str) -> float:
    """Optain the probe performance from a fitted logistic regression models.

    Args:
        model:          the fitted model
        data_path:      the path to the evaluation data
        label_path:     the path to the evaluation labels
    Returns:
         the accuracy
    """
    data, labels = get_data_annd_labels(data_path, label_path)
    print('Loaded data:', data_path)
    print('Evaluating with data of shape', data.shape)
    preds = model.predict(data)
    return accuracy_score(labels, preds)


def train_model_for_data(train_set: Tuple[str, str], eval_set: Tuple[str, str], verbose: int = 0):
    """Train a logistic regression model and evaluate it on the test set.
    Args:
        train_set:   the training dataset
        eval_set:    the evaluation dataset
    """
    print('Training model')
    model = train_model(*train_set, verbose=verbose)
    print('Obtaining metrics')
    train_acc = obtain_accuracy(model, *train_set)
    eval_acc = obtain_accuracy(model, *eval_set)
    print(os.path.basename(train_set[0]))
    print('Train acc', train_acc)
    print('Eval acc:', eval_acc)
    return train_acc, eval_acc


def train_model_for_data_mp(args):
    """Train models on latent representation distributed on multiple processes.
    Args:
        args: PseudodoArgs
    """
    return train_model_for_data(*args)


def main(args: PseudoArgs):
    """Execute model training on latent representations
    Args:
        args:  The configuration of the training as PseudoArgs object
    """
    if os.path.exists(os.path.join(args.save_path, config.PROBE_PERFORMANCE_SAVEFILE)):
        print('Detected existing results', os.path.join(args.save_path, config.PROBE_PERFORMANCE_SAVEFILE))
        if args.overwrite:
            print("overwriting is enabled. Training will continue and previous results will be overwritten.")
        else:
            print("overwriting is disabled, stopping...")
            return
    names, t_accs, e_accs = [], [], []
    train_set, eval_set = obtain_all_dataset(args.folder)
    if len(train_set) != len(eval_set):
        raise FileNotFoundError(f"Number of training sets ({len(train_set)}) does not"
                                f"match the number of test sets ({len(eval_set)})."
                                f"Make sure the datas has been extracted correctly. Consider rerunning "
                                f"extraction.")
    fargs = []
    for train_data, eval_data in zip(train_set, eval_set):
        names.append(os.path.basename(train_data[0][:-2]))
        if args.mp == 0:
            print('Multiprocessing is disabled starting training...')
            train_acc, eval_acc = train_model_for_data(train_data, eval_data)
            t_accs.append(train_acc)
            e_accs.append(eval_acc)
            pd.DataFrame.from_dict(
                {
                    'name': names,
                    'train_acc': t_accs,
                    'eval_acc': e_accs
                }
            ).to_csv(os.path.join(args.save_path, config.PROBE_PERFORMANCE_SAVEFILE), sep=';')
        else:
            fargs.append((train_data, eval_data, args.verbose))

    if args.mp != 0:
        p = Parallel(n_jobs=args.mp, verbose=1000)
        results = p(delayed(train_model_for_data_mp)(farg) for farg in fargs)
        for i, result in enumerate(results):
            t_accs.append(result[0])
            e_accs.append(result[1])
        pd.DataFrame.from_dict(
            {
                'name': names,
                'train_acc': t_accs,
                'eval_acc': e_accs
            }
        ).to_csv(os.path.join(args.save_path, config.PROBE_PERFORMANCE_SAVEFILE), sep=';')


def parse_model(model_name, shape, num_classes) ->  str:
    """Get the name of the model
    Args:
        model_name:         factory name of the model
        shape:              input shape (may be any 2 tupe of ints)
        num_classes:        the number of classes of the dataset (may be any integer)
    Returns:
        the actual name of the model
    """
    try:
        from phd_lab.experiments.utils.config import MODEL_REGISTRY as models
        model = models.__dict__[model_name](input_size=shape, num_classes=num_classes)
    except KeyError:
        raise NameError("%s doesn't exist." % model_name)
    return model.name


