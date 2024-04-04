# function to load different models


from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

def load_fasterrcnn(cnn_backbone, num_classes, anchor_gen, pretrain_wts=None):
    """
    If using weights you pretrained, set the value of `pretrain_wts` to the full file path for the checkpoint being opened
    
    """


    # initialize model by class

    if cnn_backbone == 'resnet':
        model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
        if pretrain_wts is not None:
            # make sure you are getting a pt or pth file
            assert pretrain_wts.endswith(".pth") | pretrain_wts.endswith(".pt")
            # open checkpoint
            chkpt = torch.load(pretrain_wts, map_location=device)
            # load weights
            model.load_state_dict(chkpt['state_dict'])

        model.rpn.anchor_generator = anchor_gen
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)



    return model