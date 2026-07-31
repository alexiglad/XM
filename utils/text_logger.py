import logging
import os
import sys
import json
# Assemble all the code into a single function for easy logger instantiation

# Custom Stream Handler to control console printing
class Tee(object):
    def __init__(self, terminal, logfile):
        self.terminal = terminal
        if not os.path.exists(logfile):
            open(logfile, 'w').close()
        self.log = open(logfile, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()


class CustomStreamHandler(logging.StreamHandler):
    def emit(self, record):
        print_to_console = getattr(record, 'print_to_console', True)
        if print_to_console:
            super().emit(record)



# Custom Logger to override info method
class CustomLogger(logging.Logger):
    def info(self, msg, *args, print_to_console=True, **kwargs):
        if self.isEnabledFor(logging.INFO):
            if 'extra' in kwargs:
                kwargs['extra']['print_to_console'] = print_to_console
            else:
                kwargs['extra'] = {'print_to_console': print_to_console}
            self._log(logging.INFO, msg, args, **kwargs)

def setup_custom_logger(log_filename, name="custom_logger", print_console=True, base_lor_dir="./logs/"):
    # Create an instance of CustomLogger
    custom_logger = CustomLogger(name=name)
    custom_logger.setLevel(logging.INFO)
    
    # File handler for logging
    file_handler = logging.FileHandler(base_lor_dir + log_filename)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    custom_logger.addHandler(file_handler)
    
    # Console handler for logging
    if print_console:
        logfile_full_path = os.path.join(base_lor_dir, log_filename)
        tee = Tee(sys.stdout, logfile_full_path)  # Now using Tee to log to console and file
        console_handler = CustomStreamHandler(stream=tee)  # Pass Tee as the stream to CustomStreamHandler
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        custom_logger.addHandler(console_handler)
    
    return custom_logger

class JsonlLogger(logging.Logger):
    def __init__(self, name, log_filename, base_log_dir="./logs/"):
        super().__init__(name)
        self.log_filename = os.path.join(base_log_dir, log_filename)
        os.makedirs(base_log_dir, exist_ok=True)  # Ensure the base log directory exists
        
        # Open the file in write mode to ensure all existing data is overwritten
        self.file = open(self.log_filename, 'w', encoding='utf-8') #NOTE can set to append mode if preferred instead of overwriting

    def log_data(self, data):
        """
        Logs a dictionary or a JSON-serializable object as a new line in the .jsonl file.
        """
        if not isinstance(data, (dict, list)):
            raise ValueError("Only dictionary or list data can be logged in JSONL format.")
        
        # Write the JSON string to the file, ensuring it's in JSONL format
        self.file.write(json.dumps(data) + '\n')
        self.file.flush()  # Ensure data is written to the file immediately

    def close(self):
        """Closes the file when logging is complete."""
        self.file.close()

def setup_jsonl_logger(log_filename, name="jsonl_logger", base_log_dir="./logs/"):
    return JsonlLogger(name=name, log_filename=log_filename, base_log_dir=base_log_dir)
