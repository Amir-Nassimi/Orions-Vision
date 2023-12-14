import argparse

from utils import FrameUtils
from load_stream import Camera
from database import DatabaseManager
from person_reidentification import PersonReidentification
from person_processing_manager import PersonProcessingManager


class Execution:
    def __init__(self, **kwargs):
        
        self._utils_manager = FrameUtils()
        self._stream_manager = Camera(kwargs.get('fps'))
        self._db_manager = DatabaseManager(kwargs.get('no_db_limit'))
        self._per_ident_manager = PersonReidentification(kwargs.get('model_name'))
        self._tracking_manager = PersonProcessingManager(self._db_manager, self._per_ident_manager)

    @property
    def db_manager(self):
        return self._db_manager

    @property
    def utils_manager(self):
        return self._utils_manager

    @property
    def per_ident_manager(self):
        return self._per_ident_manager

    @property
    def tracking_manager(self):
        return self._tracking_manager

    def run(self): pass
        # Your execution code here


def main():
    parser = argparse.ArgumentParser(description='Person Re-identification Application')
    parser.add_argument('--fps', required=False, default=30, type=int, help='The frame capture rate per second')
    parser.add_argument('--model_name', required=False, default='model.pth.tar-80', type=str, help='The name of the model. Caution: the model MUST be inside the ./models directory.')
    parser.add_argument('--no_db_limit', required=False, default=15, type=int, help='When the database is full, determine the number of data items to retain while discarding the oldest to make room for the newest ones')
    
    args = parser.parse_args()

    executer = Execution(**vars(args))
    executer.run()

if __name__ == "__main__":
    main()
