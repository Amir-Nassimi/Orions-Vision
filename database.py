import time
from singleton_decorator import singleton


@singleton
class DatabaseManager:
    def __init__(self, no_data_keep = 15):
        self.database = []
        self.data_keep = no_data_keep
        self.database_unknowns = []

    def add_person(self, person_emb, flag, indx=None):
        if flag:
            if indx is None:
                indx = f'{len(self.database) + 1}'
            else:
                if 'Unknown ' in indx:
                    indx = indx.replace('Unknown ', '')
        
            data = {
                'Id': f'{indx}',
                'Embds': [person_emb],
                'Last_Update': time.gmtime()
            }
        
            self.database.append(data)
            return indx
        
        else:
            if indx is None:
                indx = f'{len(self.database_unknowns) + 1}'
            else:
                if 'Unknown ' in indx:
                    indx = indx.replace('Unknown ', '')
        
            data = {
                'Id': f'Unknown {indx}',
                'Embds': [person_emb],
                'Last_Update': time.gmtime()
            }
        
            self.database_unknowns.append(data)
            return f'Unknown {indx}'

    def time_passed(self, current_time, seconds=2):
        old_second = current_time[5]
        current_second = time.gmtime()[5]

        return current_second - old_second >= seconds