import sys

def progress_bar(total, current, prefix="", suffix=""):
    bar_length = 50
    progress = float(current) / total
    arrow = '=' * int(round(progress * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    
    sys.stdout.write(f"\r{prefix} [{arrow}{spaces}] {int(progress*100)}% {suffix}")
    sys.stdout.flush()
    
    if current == total:
        sys.stdout.write("\n")