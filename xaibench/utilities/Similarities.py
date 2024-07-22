

def jaccard_similarity(list1, list2):
    '''Returns the Jaccard similarity between two lists.

    Parameters
    ----------
    list1, list2 : List
        Lists to compare.

    Returns
    -------
    out: Int
        Jaccard similarity between the two lists.

    '''
    list1, list2 = set(list1), set(list2)
    intersection = len(list(list1.intersection(list2)))
    union_ = (len(list(list1.union(list2))))
    return intersection / union_
