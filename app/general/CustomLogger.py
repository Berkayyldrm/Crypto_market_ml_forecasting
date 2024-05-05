import logging

class CustomLogger:
    def __init__(self, name, log_file, level=logging.DEBUG):
        self.log_file = log_file
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        if not self.logger.handlers:
            formatter = logging.Formatter('%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s', '%D---%H:%M:%S')
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)

    def get_logger(self):
        return self.logger
    
    def clear_log_file(self):
        open(self.log_file, 'w').close()
