"""A collection of tools that are required to help other scripts.
"""

import sys

def progress_bar(total: int, current: int, prefix: str = "", suffix: str = "") -> None:
    """Displays a text-based rogress bar.

    Creates a text based progress bar. The effects are achieved with terminal text manipulation.

    Args: 
        total : The total number of tasks, representing 100% of the bar.
        current : The number of tasks that have been completed.
        prefix : An optional string to display at the beginning of the progress bar.
        suffix : An optional string to display at the end of the progress bar.

    Returns:
        None
    """
    bar_length = 50
    progress = float(current) / total
    arrow = '=' * int(round(progress * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    
    sys.stdout.write(f"\r{prefix} [{arrow}{spaces}] {int(progress*100)}% {suffix}")
    sys.stdout.flush()
    
    if current == total:
        sys.stdout.write("\n")

