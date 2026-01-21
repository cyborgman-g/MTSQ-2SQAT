import os
from pathlib import Path
from datetime import datetime

class Logger:
    def __init__(self, exp_name: str, base_dir: str = None):
        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Results")

        self.exp_name = exp_name.upper()
        self.chkpt_folder = os.path.join(base_dir, self.exp_name)

        if not os.path.isdir(self.chkpt_folder):
            os.makedirs(self.chkpt_folder)

        self.log_file = os.path.join(self.chkpt_folder, "LOG_FILE.txt")

        with open(self.log_file, "w") as F:
            F.write(f"{'='*60}\n")
            F.write(f"{self.exp_name} Experiment Results\n")
            F.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            F.write(f"{'='*60}\n\n")

    def write(self, *args, **kwargs):
        data = " ".join([str(i) for i in args])

        with open(self.log_file, "a") as F:
            F.write(f">>  {data}\n")

    def section(self, title: str):
        header = f"\n{'='*60}\n{title}\n{'='*60}"
        with open(self.log_file, "a") as F:
            F.write(header + "\n")

    def result(self, key: str, value, precision: int = 4):
        if isinstance(value, float):
            value_str = f"{value:.{precision}f}"
        else:
            value_str = str(value)

        self.write(f"{key}: {value_str}")

    def error(self, message: str):
        error_msg = f"[ERROR] {message}"
        with open(self.log_file, "a") as F:
            F.write(f">>  {error_msg}\n")

    def get_checkpoint_path(self, filename: str) -> str:
        return os.path.join(self.chkpt_folder, filename)

    def close(self):
        with open(self.log_file, "a") as F:
            F.write(f"\n{'='*60}\n")
            F.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            F.write(f"{'='*60}\n")
