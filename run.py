import cv2
import argparse
import numpy as np
from datetime import datetime
from math import hypot as Dist

from utils import FrameUtils
from load_stream import Camera
from database import DatabaseManager
from person_detection import Detection
from person_reidentification import PersonReidentification
from person_processing_manager import PersonProcessingManager


class Execution:
    def __init__(self, **kwargs):
        self._utils_manager = FrameUtils()
        self._sfr = kwargs.get('skipp_frame_rate')
        self._thrsh_estimate = kwargs.get('thresh_on_estimate')
        self._db_manager = DatabaseManager(kwargs.get('no_db_limit'))
        self._streamer_manager = Camera(kwargs.get('filepath'), kwargs.get('fps'))
        self._per_ident_manager = PersonReidentification(kwargs.get('tracking_model_name'))
        self._tracking_manager = PersonProcessingManager(self._db_manager, self._per_ident_manager)
        self._detector_manager = Detection(kwargs.get('threshold'), kwargs.get('detector_model_name'))

    @property
    def detector_manager(self):
        return self._detector_manager

    @property
    def db_manager(self):
        return self._db_manager

    @property
    def sfr(self):
        return self._sfr
    
    @property
    def thrsh_estimate(self):
        return self._thrsh_estimate
    
    @property
    def streamer_manager(self):
        return self._streamer_manager

    @property
    def utils_manager(self):
        return self._utils_manager

    @property
    def per_ident_manager(self):
        return self._per_ident_manager

    @property
    def tracking_manager(self):
        return self._tracking_manager

    def run(self): 
        self.streamer_manager.run()

        while not self.streamer_manager.exit_signal.is_set():
            counter, count_skipped = 0, 0
            new_frame,all_labels = {}, []

            try: frame = self.streamer_manager.frames_queue.get()
            except AttributeError: break

            if counter != 0 and (counter % self.sfr) == 0: 
                count_skipped += 1
                if count_skipped > 1: 
                    counter += 1
                    count_skipped = 0
                    
                else: continue
                    
            else: counter += 1

            imgs = self.detector_manager.Object_Detection(frame)

            if len(imgs) == 0: pass
            else: new_frame = self.utils_manager.crop_persons(imgs,new_frame)
                
            if len(new_frame) == 0:
                cv2.imshow('Tracking', np.array(frame))
    
                if cv2.waitKey(1) & 0xFF == ord('q'): break  
                else: continue        
            
            if len(old_frame) == 0 or (counter % self.thrsh_estimate) == 0:
                new_frame = self.tracking_manager.Identify_Persons(new_frame, [], all_labels)

            elif len(old_frame) <= len(new_frame):
                for i in old_frame:
                    distances_compares = []
                    cor_old = [(old_frame[i]['Coordination'][0]+old_frame[i]['Coordination'][2])/2, (old_frame[i]['Coordination'][1]+old_frame[i]['Coordination'][3])/2]
                    
                    for j in new_frame:
                        cor_current = [(new_frame[j]['Coordination'][0]+new_frame[j]['Coordination'][2])/2, (new_frame[j]['Coordination'][1]+new_frame[j]['Coordination'][3])/2]     
                        dist_ = Dist(cor_old[0]-cor_current[0],cor_old[1]-cor_current[1])              
                        distances_compares.append(dist_)

                    j = np.argmin(distances_compares)
                    
                    new_frame[j]['Status'] = 'Estimate'
                    new_frame[j]['Label'] = old_frame[i]['Label']
                    new_frame[j]['Last_PID_Run'] = datetime.now()
                    
                    all_labels.append(old_frame[i]['Label'])
                    
                new_frame_2 = {}
                for indx in new_frame:
                    if new_frame[indx]['Status'] == '-': new_frame_2[indx] = new_frame[indx].copy()
                    else: continue
                
                new_frame_2 = self.tracking_manager.Identify_Persons(new_frame_2,old_frame,all_labels)
                
                new_frame = self.utils_manager.merge_frame_dictionaries(new_frame,new_frame_2)
                
            else: 
                for i in new_frame:
                    distances_compares = []
                    cor_old = [(new_frame[i]['Coordination'][0]+new_frame[i]['Coordination'][2])/2, (new_frame[i]['Coordination'][1]+new_frame[i]['Coordination'][3])/2]
                    
                    for j in old_frame:
                        cor_current = [(old_frame[j]['Coordination'][0]+old_frame[j]['Coordination'][2])/2, (old_frame[j]['Coordination'][1]+old_frame[j]['Coordination'][3])/2]     
                        
                        dist_ = Dist(cor_old[0]-cor_current[0],cor_old[1]-cor_current[1])

                        distances_compares.append(dist_)

                    j = np.argmin(distances_compares)

                    new_frame[i]['Status'] = 'Estimate'
                    new_frame[i]['Label'] = old_frame[j]['Label']
                    new_frame[i]['Last_PID_Run'] = datetime.now()
            
            for i in new_frame:
                frame = self.utils_manager.write_on_image(f"{new_frame[i]['Label']}", new_frame[i]['Coordination'], frame)
            
            self.tracking_manager.Databse_Update(new_frame)
            old_frame = new_frame.copy()
            
            cv2.imshow('Tracking', np.array(frame))
            if cv2.waitKey(1) & 0xFF == ord('q'): break  
                
            print(f"{counter}**************************")

        self.streamer_manager.stop_threads()


def main():
    parser = argparse.ArgumentParser(description="Orion's Vision Tracking System")
    
    parser.add_argument('--skip_frame_rate', required=False, default=0, type=int, help='Skip frame rate')
    parser.add_argument('--threshold', required=False, type=float, default=45, help='Threshold for detection')
    parser.add_argument('--fps', required=False, default=30, type=int, help='The frame capture rate per second')
    parser.add_argument('--filepath', required=True, type=str, help='File path of video or rtsp adrs of camera')
    parser.add_argument('--detector_model_name', required=False, default='yolov8x.pt', help='Name to YOLO model weights. Caution: the model MUST be inside the ./models directory.')
    parser.add_argument('--thresh_on_estimate', required=False, type=int, default=int(5e2), help='Threshold on estimate for updating the database based on backbone tracking algorithm')
    parser.add_argument('--tracking_model_name', required=False, default='model.pth.tar-80', type=str, help='The name of the model. Caution: the model MUST be inside the ./models directory.')
    parser.add_argument('--no_db_limit', required=False, default=15, type=int, help='When the database is full, determine the number of data items to retain while discarding the oldest to make room for the newest ones')
    
    args = parser.parse_args()

    executer = Execution(**vars(args))
    executer.run()

if __name__ == "__main__":
    main()