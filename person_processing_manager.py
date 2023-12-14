class PersonProcessingManager:
    def __init__(self, database_manager, person_reidentification):
        self.database_manager = database_manager
        self.person_reidentification = person_reidentification

    def database_update(self, persons):
        for data in self.database_manager.database:
            if self.database_manager.time_passed(data['Last_Update']):
                flag_ = False
                for indx in persons:
                    if persons[indx]['Label'] == data['Id']:
                        flag_ = True
                        break
                    else:
                        continue

                if flag_:
                    if len(data['Embds']) > (self.database_manager.data_keep + 1):
                        data['Embds'].pop(0)

                    new_embd = Person_ReIdentification(persons[indx]['Img'], flag_known=False, flag_unknown=False)
                    data['Embds'].append(new_embd)
                    data['Last_Update'] = time.gmtime()
                else:
                    continue
            else:
                continue

        for data in self.database_manager.database_unknowns:
            if self.database_manager.time_passed(data['Last_Update']):
                flag_ = False
                for indx in persons:
                    if persons[indx]['Label'] == data['Id']:
                        flag_ = True
                        break
                    else:
                        continue

                if flag_:
                    if len(data['Embds']) > (self.database_manager.data_keep + 1):
                        data['Embds'].pop(0)

                    new_embd = Person_ReIdentification(persons[indx]['Img'], flag_known=False, flag_unknown=False)
                    data['Embds'].append(new_embd)
                    data['Last_Update'] = time.gmtime()
                else:
                    continue
            else:
                continue