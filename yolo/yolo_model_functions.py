## - YOLO Model Functions


# Define class size range based on model type
def class_range(model_type):
    if model_type == 'species':
        # testing smaller numbers w data augmentation, speed up training time, train more species
        max_per_category = 2100
        min_per_category = 300
    if model_type == 'family':
        max_per_category = 2000
        min_per_category = 200
    if model_type == 'general':
        max_per_category = 30000
        min_per_category = 5000
    if model_type == 'pig_only':
        max_per_category = 1000
        min_per_category = 300
    return max_per_category, min_per_category


# data wrangling
def wrangle_df(df, IMAGE_ROOT):
    # confirm bounding boxes are listed as proportions
    assert df['YMax'].max() <= 1
    assert df['XMax'].max() <= 1

    # fix special characters in file names
    # 07.22.22 AB: uncomment this line on CURC, comment out when local
    # df['filename'] = df['filename'].str.replace("'","").replace(" ","")
    # TODO: ask Ryan to remove special characters from folder names

    # cross check filenames to image files in directory
    # for local jobs, need to be connected to VPN here
    extant = [os.path.join(dp, f).replace(os.sep, '/') for dp, dn, fn in os.walk(IMAGE_ROOT) for f in fn]
    extant = [x.replace(IMAGE_ROOT + '/', '/').strip('/') for x in extant]

    # filter df for existing images
    df = df[df['filename'].isin(extant)]

    # exclude partial images
    df = df[df['partial.image'] == False]

    # exclude species with poor annotations
    # TODO: review these images for updated bboxes
    drops = ['Wood_Stork', 'Sandhill_Crane', 'Great_Egret',
             'Great_Blue_Heron', 'Crested_Caracara', 'Black-crowned_Night_Heron', 'Caribou']
    df = df.loc[~df['common.name'].isin(drops)].reset_index(drop=True)

    # swap y-axis for images where bboxes originate in LL corner (pytorch looks for UL and LR corners)
    df['YMin_org'] = df['YMin']
    df['YMax_org'] = df['YMax']
    df.drop(['YMax', 'YMin'], axis=1, inplace=True)
    df['YMax'] = np.where(df['bbox.origin'] == 'LL', (1 - df['YMin_org']), df['YMax_org'])
    df['YMin'] = np.where(df['bbox.origin'] == 'LL', (1 - df['YMax_org']), df['YMin_org'])

    # check for bboxes that are too small
    df['YDiff'] = df['YMax'] - df['YMin']
    df['XDiff'] = df['XMax'] - df['XMin']
    df_tooSmall = df[(df['YDiff'] < 0.001) | (df['XDiff'] < 0.001)]
    assert df_tooSmall.shape[0] == 0, "Remove image files where bboxes are too small"

    # create new columns for YOLO bboxes
    # Note: this formula applies to *normalized* bbox coordinates!
    df['x_center'] = (df['XMin'] + df['XMax']) / 2
    df['y_center'] = (df['YMin'] + df['YMax']) / 2
    df['w_bbox'] = df['XMax'] - df['XMin']
    df['h_bbox'] = df['YMax'] - df['YMin']

    # update column names for common name
    df['common.name_org'] = df['common.name']
    df['common.name'] = df['common.name.general']

    # combine squirrel species into one group
    squirrels = ["Aberts_Squirrel", 'American_Red_Squirrel', 'Douglas_Squirrel', 'Eastern_Fox_Squirrel',
                 'Eastern_Gray_Squirrel', 'Fox_Squirrel', 'Golden-Mantled_Ground_Squirrel', 'Gray_Squirrel',
                 'Northern_Flying_Squirrel', 'Rock_Squirrel', 'Squirrel', 'Western_Gray_Squirrel']
    df.loc[df['common.name'].isin(squirrels), 'common.name'] = 'squirrel_spp'

    # change the taxonomic classifications for vehicle
    df.loc[df['common.name'] == 'Vehicle', ['genus', 'species', 'family', 'order', 'class']] = 'vehicle'

    # add category for general model
    conditions = [(df['class'] == 'Mammalia') & (df['common.name'] != "Human") & (df['common.name'] != "Vehicle"),
                  (df['class'] == "Aves"),
                  (df['common.name'] == 'Human'),
                  (df['common.name'] == "Vehicle")]
    choices = ['mammal', 'bird', 'human', 'vehicle']
    df['general_category'] = np.select(conditions, choices, default=np.NAN)
    return df



# Create PyTorch dataset
class DetectDataset(torch.utils.data.Dataset):
    def __init__(self, df, w, h, image_dir, transform):
        self.image_dir = image_dir
        self.df = df
        self.image_infos = df.filename.unique()
        self.w = w
        self.h = h
        self.transform = transform

    def __getitem__(self, item):
        # create image id
        image_id = self.image_infos[item]
        # create full path to open each image file
        img_path = os.path.join(self.image_dir, image_id).replace("\\", "/")
        # open image, convert to numpy array
        # resize here so bboxes can also be resized before transformations
        img = Image.open(img_path).convert("RGB").resize((self.w, self.h), resample=Image.Resampling.BILINEAR)
        img = np.array(img, dtype="float32") / 255.
        # extract bbox and label data for image
        data = df[df['filename'] == image_id]
        # extract label names
        labels = data['LabelName'].values.tolist()
        # extract bbox coordinates
        data = data[['XMin', 'YMin', 'XMax', 'YMax']].values
        # convert to absolute values for model input
        data[:, [0, 2]] *= self.w
        data[:, [1, 3]] *= self.h
        # convert coordinates to list
        boxes = data.tolist()
        # convert bboxes and labels to a tensor dictionary
        target = {
            'boxes': torch.tensor(boxes).float(),
            'labels': torch.tensor([label2target[i] for i in labels]).long()
        }
        if self.transform is not None:
            augmented = self.transform(image=img, bboxes=target['boxes'], labels=labels)
            img = augmented['image'].to(device).float()
            target['boxes'] = augmented['bboxes']
        target['boxes'] = torch.tensor(target['boxes']).float()  # for some reason, ToTensorV2() isn't working
        return img, target

    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def __len__(self):
        return len(self.image_infos)


# define data augmentation pipelines
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Affine(rotate=(-20, 20), fit_output=True, p=0.3),
    A.Affine(shear=(-20, 20), fit_output=True, p=0.3),
    A.RandomBrightnessContrast(brightness_by_max=True, p=0.3),
    A.RandomSizedBBoxSafeCrop(height=307, width=408, erosion_rate=0.2, p=0.5),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0),
)
val_transform = A.Compose([
    ToTensorV2(),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
)

# Plot images
COLORS = np.random.randint(0, 255, size=(80, 3), dtype="uint8")


# manually adjust score threshold?
def show_img_bbox(img, targets, score_threshold=0.7):
    # convert image format to PIL
    if torch.is_tensor(img):
        img = to_pil_image(img)
    elif isinstance(img, np.ndarray):
        img = Image.fromarray(img, 'RGB')
    # convert bboxes and labels to np array
    if torch.is_tensor(targets):
        boxes = targets.numpy()[:, 1:]
    else:
        # if using the input of `output`.
        if 'scores' in targets:
            boxes, scores, labels = decode_output(targets, labels_as_numbers=True)
        else:
            boxes = targets['boxes']
            labels = targets['labels']
            scores = [1] * len(boxes)
    # plot image
    draw = ImageDraw.Draw(img)
    # plot bboxes
    for i, tg in enumerate(boxes):
        if scores[i] > score_threshold:
            id_ = int(labels[i])
            bbox = boxes[i]
            xmin, ymin, xmax, ymax = bbox
            color = [int(c) for c in COLORS[id_]]
            name = target2label[id_]
            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=tuple(color), width=3)
            draw.text((xmin, ymin), name, fill=(255, 255, 255, 0))
    plt.imshow(np.array(img))


# define model


# Define PyTorch data loaders
def get_dataloaders(train_df, train_ds, val_ds, model_type, num_classes, batch_size):
    # for pig model, oversample Suidae to give it weight equal to other classes combined
    if model_type == 'pig_only':
        # set up class weights
        s = dict(Counter(train_df['LabelName']))
        # weight all other classes equally by num of classes
        # note : prob a way to do this by samples as well, but ok for now
        s = {x: (1 / (2 * (num_classes - 2))) for x in s}
        # weight pig class by sum of all other samples
        s['Suidae'] = 0.5
        # sanity check weighting was done correctly
        assert sum(s.values()) == 1
        # assign sample weight to each image in the dataset
        sample_weights = train_df.LabelName.map(s).tolist()
        # create weighted random sampler
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        # define data loaders
        train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=train_ds.collate_fn, drop_last=True,
                                  sampler=sampler)
        # do not oversample for validation, just for training
        val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=train_ds.collate_fn, drop_last=True)
    # No oversampling for other models at this time
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=train_ds.collate_fn, shuffle=True,
                                  drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=train_ds.collate_fn, drop_last=True)
    return train_loader, val_loader


# obtain learning rate
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# define checkpoint functions
def create_checkpoint(model, optimizer, epoch, lr_scheduler, loss_history, best_loss, model_type, num_classes,
                      label2target):
    checkpoint = {'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'epoch': epoch + 1,
                  'lr': current_lr,
                  'scheduler': lr_scheduler.state_dict(),
                  'loss_history': loss_history,
                  'best_loss': best_loss,
                  'model_type': model_type,
                  'num_classes': num_classes,
                  'label2target': label2target}
    return checkpoint


def save_checkpoint(checkpoint, checkpoint_file):
    print(" Saving model state")
    torch.save(checkpoint, checkpoint_file)


def load_checkpoint(checkpoint_path):
    print(" Loading saved model state")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['scheduler'])
    epoch = checkpoint['epoch']
    loss_history = checkpoint['loss_history']
    best_loss = checkpoint['best_loss']
    model_type = checkpoint['model_type']
    label2target = checkpoint['label2target']
    return model, optimizer, lr_scheduler, epoch, loss_history, best_loss, model_type, label2target


# make predictions
def decode_output(output, labels_as_numbers=False):
    bbs = output['boxes'].cpu().detach().numpy().astype(np.uint16)
    if labels_as_numbers:
        labels = np.array(output['labels'].cpu().detach().numpy())
    else:
        labels = np.array([target2label[i] for i in output['labels'].cpu().detach().numpy()])
    confs = output['scores'].cpu().detach().numpy()
    ixs = nms(torch.tensor(bbs.astype(np.float32)), torch.tensor(confs), 0.5)
    bbs, confs, labels = [tensor[ixs] for tensor in [bbs, confs, labels]]
    if len(ixs) == 1:
        bbs, confs, labels = [np.array([tensor]) for tensor in [bbs, confs, labels]]
    return bbs.tolist(), confs.tolist(), labels.tolist()


# deploy function
def deploy(df, sample=False, w=408, h=307):
    model.eval()
    model.to(device)
    image_infos = df.filename.unique()
    if sample:
        image_infos = random.sample(list(image_infos), 10)
    gt_df = []
    pred_df = []
    for i in tqdm(range(len(image_infos))):
        dfi = df[df['filename'] == image_infos[i]]
        dsi = DetectDataset(df=dfi, image_dir=IMAGE_ROOT, w=w, h=h, transform=val_transform)
        dli = DataLoader(dsi, batch_size=1, collate_fn=dsi.collate_fn, drop_last=True)
        input, target = next(iter(dli))
        image = list(image.to(device) for image in input)
        output = model(image)[0]
        bbs, confs, labels = decode_output(output, labels_as_numbers=True)
        boxes = bbs
        if len(bbs) == 0:
            pred_df_i = pd.DataFrame({
                'filename': image_infos[i],
                'file_id': image_infos[i][:-4],
                'class_name': 'empty',
                'confidence': 1,
                'bbox': [0, 0, w, h]
            })
        else:
            pred_df_i = pd.DataFrame({
                'filename': image_infos[i],
                'file_id': image_infos[i][:-4],
                'class_name': [target2label[a] for a in labels],
                'confidence': confs,
                'bbox': bbs
            })
        gt_df_i = pd.DataFrame({
            'filename': image_infos[i],
            'file_id': image_infos[i][:-4],
            'class_name': [target2label[a] for a in dsi[0][1]['labels'].tolist()],
            'bbox': dsi[0][1]['boxes'].tolist(),
            'used': False
        })
        gt_df_i = gt_df_i.join(pd.DataFrame(dsi[0][1]['boxes'].tolist()))
        gt_df.append(gt_df_i)
        pred_df.append(pred_df_i)
    gt_df = pd.concat(gt_df)
    pred_df = pd.concat(pred_df)
    gt_df.to_csv(eval_path + "gt_df.csv")
    pred_df.to_csv(eval_path + "pred_df.csv")
    return gt_df, pred_df


# define intersection over union function
def IoU(bbs, tbs):
    """
    Calculates intersection over union
    :param bbs: set of prediction bounding boxes for an image
    :param tbs: set of true target bounding boxes for an image
    :return: intersection over union
    """
    # shape is (N,4) where N is number of bboxes per image
    # Convert coordinates from (x,y,w,h) to (x1, y1, x2, y2)
    pred_xmin = bbs[..., 0:1] - bbs[..., 2:3] / 2
    pred_ymin = bbs[..., 1:2] - bbs[..., 3:4] / 2
    pred_xmax = bbs[..., 2:3] + bbs[..., 2:3] / 2
    pred_ymax = bbs[..., 3:4] + bbs[..., 3:4] / 2
    target_xmin = tbs[..., 0:1] - tbs[..., 2:3] / 2
    target_ymin = tbs[..., 1:2] - tbs[..., 3:4] / 2
    target_xmax = tbs[..., 2:3] + tbs[..., 2:3] / 2
    target_ymax = tbs[..., 3:4] + tbs[..., 3:4] / 2

    # find area of each box
    pred_area = abs((pred_xmax - pred_xmin) * (pred_ymax - pred_ymin))
    target_area = abs((target_xmax - target_xmin) * (target_ymax - target_ymin))

    # find intersection area
    xmin = torch.max(pred_xmin, target_xmin)
    ymin = torch.max(pred_ymin, target_ymin)
    xmax = torch.min(pred_xmax, target_xmax)
    ymax = torch.min(pred_ymax, target_ymax)
    intersect = (xmax - xmin).clamp(0) * (ymax - ymin).clamp(0)

    # find area of union
    union = pred_area + target_area - intersect + 1e-6  # add numeric stabilizer in case union = 0

    return intersect / union
