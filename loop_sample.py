from glob import glob
import pandas as pd

images_list = glob.glob('*.png')
save_for_confusion_matrix = {'img_name':[],
                             'bbox_x1':[],
                             'bbox_y1':[],
                             'bbox_x2':[],
                             'bbox_y2:'[],
                             'pred_label':[]}

for idx, img in enumerate(images_list):
    pass
    net = load_net("cfg/tiny-yolo.cfg", "tiny-yolo.weights", 0)
    meta = load_meta("cfg/coco.data")
    r = detect(net, meta, "data/dog.jpg")
    # Do reading
    # Do detection
    # add into dict


# Done
df = pd.DataFrame(save_for_confusion_matrix)
df.to_csv('your_filename.csv')
