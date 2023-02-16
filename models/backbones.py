# function to load different models


from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

def load_fasterrcnn(cnn_backbone, num_classes, anchor_gen):


    # initialize model by class

    if cnn_backbone == 'resnet':
        model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
        model.rpn.anchor_generator = anchor_gen
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model