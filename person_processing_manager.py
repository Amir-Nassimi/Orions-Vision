import cv2
import time
import torch
import numpy as np
from datetime import datetime

from image_processor import ImageProcessor

class PersonProcessingManager:
    def __init__(self, database_manager, person_reidentification):
        self.thresh = 8.5
        self.thresh_param = 3
        
        self.pre_proc = ImageProcessor()
        self.database_manager = database_manager
        self.person_reidentification = person_reidentification

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

                    new_embd = self.Person_ReIdentification(persons[indx]['Img'], flag_known=False, flag_unknown=False)
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

                    new_embd = self.Person_ReIdentification(persons[indx]['Img'], flag_known=False, flag_unknown=False)
                    data['Embds'].append(new_embd)
                    data['Last_Update'] = time.gmtime()
                else:
                    continue
            else:
                continue

    def Person_ReIdentification(self, img, flag_known=True, flag_unknown=False):
        img = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2RGB)
        
        img = self.pre_proc.pre_proc_light(img)
        img_ = self.person_reidentification.pre_proc(img)

        x = self.person_reidentification.reidentification_model(img_.unsqueeze(0).float().to(self.device)).detach().cpu()
        
        if flag_known:
            distns = []
            for data in self.database_manager.database:
                for embd in data['Embds']:
                    dist = torch.cdist(x, embd, p=2.0)[0][0]
                    distns.append((data['Id'],dist))
            return distns,x
        else: pass
                
        if flag_unknown:
            distns = []
            for data in self.database_manager.database_unknowns:
                for embd in data['Embds']:
                    dist = torch.cdist(x, embd, p=2.0)[0][0]
                    distns.append((data['Id'],dist))
            return distns,x
        else: pass
        
        return x
    
    def Identify_Persons(self, persons, old_frame_info, labels):
        for person_num in persons:
            distns,emb = self.Person_ReIdentification(persons[person_num]['Img'],flag_known=True,flag_unknown=False)
            label,dist,flag = self.person_reidentification.Check_Similarity(distns,self.thresh)

            if flag: 
                persons[person_num],_,_,_,_,labels,old_frame,persons,flag_state = self.Check_And_Go(persons[person_num],label,emb,'Known',dist,labels,old_frame_info,persons)
                if flag_state:
                    labels.append(label)
                    persons[person_num]['Label'] = label
                    persons[person_num]['Status'] = 'Identified'
                    persons[person_num]['Last_PID_Run'] = datetime.now()
                else: continue

            else:
                distns,emb = self.Person_ReIdentification(persons[person_num]['Img'],flag_known=False,flag_unknown=True)
                label,dist,flag = self.person_reidentification.Check_Similarity(distns,self.thresh)

                if flag: 
                    persons[person_num],_,_,_,_,labels,old_frame,persons,flag_state = self.Check_And_Go(persons[person_num],label,emb,'Unknown',dist,labels,old_frame_info,persons)
                    
                    if flag_state:
                        labels.append(label)
                        persons[person_num]['Label'] = label
                        persons[person_num]['Status'] = 'Identified'
                        persons[person_num]['Last_PID_Run'] = datetime.now()
                    else: continue

                else:
                    label,dist,flag = self.person_reidentification.Check_Similarity(distns,self.thresh + self.thresh_param)

                    if flag:
                        persons[person_num],_,_,_,_,labels,old_frame,persons,flag_state = self.Check_And_Go(persons[person_num],label,emb,'Unknown',dist,labels,old_frame_info,persons)
                        if flag_state:
                            labels.append(label)
                            persons[person_num]['Label'] = label
                            persons[person_num]['Status'] = 'Identified'
                            persons[person_num]['Last_PID_Run'] = datetime.now()
                        else: continue

                    else: 
                        label = self.database_manager.add_person(emb,False)
                        
                        labels.append(label)
                        persons[person_num]['Label'] = label
                        persons[person_num]['Status'] = 'Identified'
                        persons[person_num]['Last_PID_Run'] = datetime.now()

        return persons

    def Check_And_Go(self, current_person, current_label, current_emb, state, current_dist, prev_labels, old_frame, new_persons):
        if current_label in prev_labels:
            flag_ = False
            for indx in old_frame:
                if current_label == old_frame[indx]['Label']:
                    flag_ = True
                    break
                else: continue
                    
            if flag_:
                if 'Unknown' in current_label:
                    label = self.database_manager.add_person(current_emb,False)

                    prev_labels.append(label)
                    current_person['Label'] = label
                    current_person['Status'] = 'Identified'
                    current_person['Last_PID_Run'] = datetime.now()

                    return current_person,label,current_emb,'Unknown',9e9,prev_labels,old_frame,new_persons,False
                
                else:
                    distns,emb = self.Person_ReIdentification(current_person['Img'],flag_known=False,flag_unknown=True)
                    label,dist,flag = self.person_reidentification.Check_Similarity(distns,self.thresh)

                    print(f"1 - current label: {current_label} - label: {label}")

                    if flag: return self.Check_And_Go(current_person,label,emb,'Unknown',dist,prev_labels,old_frame,new_persons)
                    else:
                        label,dist,flag = self.person_reidentification.Check_Similarity(distns,self.thresh + self.thresh_param)
                        print(f"1 - current label: {current_label} - label: {label}")

                        if flag: return self.Check_And_Go(current_person,label,emb,'Unknown',dist,prev_labels,old_frame,new_persons)
                        else: 
                            label = self.database_manager.add_person(emb,False)

                            prev_labels.append(label)
                            current_person['Label'] = label
                            current_person['Status'] = 'Identified'
                            current_person['Last_PID_Run'] = datetime.now()

                            return current_person,label,emb,'Unknown',9e9,prev_labels,old_frame,new_persons,False         
                            
            else:   
                for indx in new_persons:
                    if new_persons[indx]['Label'] == current_label: break
                    else: continue
                    
                if ('Unknown' in current_label) and not ('Unknown' == current_label):
                    distns,emb = self.Person_ReIdentification(new_persons[indx]['Img'],flag_known=True,flag_unknown=False)
                    label,dist,flag = self.person_reidentification.Check_Similarity(distns,self.thresh)
                    
                    if not flag:
                        distns,emb = self.Person_ReIdentification(new_persons[indx]['Img'],flag_known=False,flag_unknown=True)
                        label,dist,flag = self.person_reidentification.Check_Similarity(distns,self.thresh)
                        print(f"2 - current label: {current_label} - label: {label}")
                        
                        if not flag: 
                            label,dist,flag = self.person_reidentification.Check_Similarity(distns,self.thresh + self.thresh_param)
                            print(f"2 - current label: {current_label} - label: {label}")
                            
                    if dist < current_dist:     
                        label = self.database_manager.add_person(current_emb,False)

                        prev_labels.append(label)
                        current_person['Label'] = label
                        current_person['Status'] = 'Identified'
                        current_person['Last_PID_Run'] = datetime.now()

                        return current_person,label,current_emb,'Unknown',9e9,prev_labels,old_frame,new_persons,False  

                    else:
                        current_person['Label'] = label
                        current_person['Status'] = 'Identified'
                        current_person['Last_PID_Run'] = datetime.now()

                        label = self.database_manager.add_person(emb,False)

                        prev_labels.append(label)
                        new_persons[indx]['Label'] = label
                        new_persons[indx]['Status'] = 'Identified'
                        new_persons[indx]['Last_PID_Run'] = datetime.now()

                        return current_person,label,emb,'Unknown',9e9,prev_labels,old_frame,new_persons,False  
                    
                else:
                    persons_dists = []
                    persons_dists.append(current_dist)
                    
                    distns,_ = self.Person_ReIdentification(new_persons[indx]['Img'],flag_known=True,flag_unknown=False)
                    try: indx,dist = min(distns, key=lambda x: x[1])
                    except ValueError: dist = 9e9
                        
                    persons_dists.append(dist)   
                        
                    arg = np.argmin(persons_dists)
                    if arg == 0:
                        current_person['Label'] = label
                        current_person['Status'] = 'Identified'
                        current_person['Last_PID_Run'] = datetime.now()
                        
                        person_to_investigate = new_persons[indx]
                        
                    else: person_to_investigate = current_person
                        
                    distns,emb = self.Person_ReIdentification(person_to_investigate['Img'],flag_known=False,flag_unknown=True)
                    label,dist,flag = self.person_reidentification.Check_Similarity(distns,self.thresh)
                    print(f"3 - current label: {current_label} - label: {label}")
                    if flag: return self.Check_And_Go(person_to_investigate,label,emb,'Unknown',dist,prev_labels,old_frame,new_persons)
                    else:
                        label,dist,flag = self.person_reidentification.Check_Similarity(distns,self.thresh + self.thresh_param)
                        print(f"3 - current label: {current_label} - label: {label}")
                        if flag: return self.Check_And_Go(person_to_investigate,label,emb,'Unknown',dist,prev_labels,old_frame,new_persons)
                        else: 
                            label = self.database_manager.add_person(emb,False)
                            
                            prev_labels.append(label)
                            person_to_investigate['Label'] = label
                            person_to_investigate['Status'] = 'Identified'
                            person_to_investigate['Last_PID_Run'] = datetime.now()
                            
                            return current_person,label,emb,'Unknown',9e9,prev_labels,old_frame,new_persons,False
        else: 
            prev_labels.append(current_label)
            return current_person,current_label,current_emb,state,current_dist,prev_labels,old_frame,new_persons,True