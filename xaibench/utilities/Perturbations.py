from nlpaug.augmenter.word import SpellingAug, SynonymAug, RandomWordAug
from nlpaug.augmenter.char import KeyboardAug
import pandas as pd


def spelling_perturb(instances, n_words=3):
    if isinstance(instances, pd.core.series.Series):
        instances = instances.tolist()
    aug = SpellingAug(
        aug_max=n_words,
        aug_min=n_words,
    )
    return aug.augment(instances)


def synonym_perturb(instances, n_words=3):
    if isinstance(instances, pd.core.series.Series):
        instances = instances.tolist()
    aug = SynonymAug(
        aug_max=n_words,
        aug_min=n_words,
    )
    return aug.augment(instances)


def typo_perturb(instances, n_words=3):
    if isinstance(instances, pd.core.series.Series):
        instances = instances.tolist()
    aug = KeyboardAug(
        aug_word_min=n_words,
        aug_word_max=n_words,
    )
    return aug.augment(instances)


def baseline_perturb(instances, n_words=3, action='delete'):
    if isinstance(instances, pd.core.series.Series):
        instances = instances.tolist()
    aug = RandomWordAug(
        aug_min=n_words,
        aug_max=n_words,
        action=action,
    )
    return aug.augment(instances)
