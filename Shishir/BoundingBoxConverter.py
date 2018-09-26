import numpy as np
import glob

class BoundingBoxConverter:
    
    def __init__(self, path_to_annotations, path_to_maps):
        self.path_to_annotations = path_to_annotations
        self.path_to_maps = path_to_maps
    
    def convert(self, data_path):
        #print(self.path_to_annotations)
        anots_file_contents = ''
        for filepath in glob.glob(self.path_to_annotations+'D*'):
            #print("Filepath:", filepath)
            filename = filepath.split('/')[-1]
            head,sep,tail = filename.partition('.')
            head = head.replace('D', '')
            image_filepath = self.path_to_maps+'D'+head+'.tiff'
            anot_filename = self.path_to_annotations+'D'+head+'.npy'
            
            anot_line = data_path + 'D'+head+'.tiff'
            
            A = np.load(anot_filename).item()

            # the following code shows how to loop through all the items in the numpy dictionary.
            # each dictionary can have at most 3 keys: vertices, name and link_to.

            for j in A.keys():
                # copy the list of vertices for jth dictionary element
                vertices = np.array(A[j]['vertices'])
                
                # find left most, right most, top most, and bottom most verticies
                x_min = np.min(vertices[:, 0])
                x_max = np.max(vertices[:, 0])
                y_min = np.min(vertices[:, 1])
                y_max = np.max(vertices[:, 1])

                # add annotation to line
                anot_line += " " + ",".join([str(int(x_min)), str(int(y_min)), str(int(x_max)), str(int(y_max)), "0"])
                
            anots_file_contents += anot_line + "\n"

        with open("train.txt", "w") as text_file:
            text_file.write(anots_file_contents)