def bias_assignment(art_source):
    """A function for assigning political bias based on 
    left and right wing think-tanks/news sources

    Args:
        art_source (string): a Pandas series string

    Returns:
        string: Left or Right wing based on string input
    """
    assert isinstance(art_source, str)
    left = ["Jacobin", "Brookings Institute"]
    right = ['Heritage Commentary', 'Heritage Report', 'American Mind']
    if art_source in left:
        return "left-wing"
    else:
        return "right-wing"