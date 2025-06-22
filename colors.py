import warnings
class Colors:
    RED = '\033[91m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    ENDC = '\033[0m'  # 結束顏色
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_green(text):
    print(f"{Colors.GREEN}{text}{Colors.ENDC}")
def print_yellow(text):
    print(f"{Colors.YELLOW}{text}{Colors.ENDC}")
def print_red(text):
    print(f"{Colors.RED}{text}{Colors.ENDC}")
def warn_with_color(message, category=UserWarning):
        print_red(f"WARNING: {message}")
        warnings.warn(message, category)

