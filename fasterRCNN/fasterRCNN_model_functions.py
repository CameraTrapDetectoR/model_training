## - CameraTrapDetectoR Model Functions


# Define class size range based on model type
def class_range(model_type):
    if model_type == 'species':
        max_per_category = 10000
        min_per_category = 100
    if model_type == 'family':
        max_per_category = 10000
        min_per_category = 300
    if model_type == 'general':
        max_per_category = 30000
        min_per_category = 5000
    if model_type == 'pig_only':
        max_per_category = 10000
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

    #TODO: include function to filter images with overlapping bboxes with IOU above a certain threshold

    # exclude species with poor annotations
    # TODO: review these images for updated bboxes
    drops = ['Great_Egret', 'Great_Blue_Heron', 'Crested_Caracara', 'Caribou']
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

# define model class dictionary and create representative sample
def define_dictionary(df, model_type):
    """
    Creates balanced sample from total available training images per model_type.
    Sample is representative of target class and database (image source).
    """

    if model_type == 'general':
        # keep images only in the general categories
        df = df[~df['general_category'] != "nan"].reset_index()
        # save original df to filter through later
        original_df = df
        # create dictionary of category labels
        label2target = {l: t + 1 for t, l in
                        enumerate(df['general_category'].unique())}
        # add background class
        label2target['empty'] = 0
        background_class = label2target['empty']
        # reverse dictionary to read into pytorch
        target2label = {t: l for l, t in label2target.items()}
        # remove vehicle class before downsampling
        g = df[df['general_category'] != 'vehicle'].groupby('general_category',
                                                            group_keys=False)
        # downsample to smallest remaining category
        balanced_df = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()))).reset_index(
            drop=True)
        # add back all vehicle images
        df = pd.concat([balanced_df, df[df['general_category'] == 'vehicle']])
        # locate images within df
        df = original_df.loc[original_df['filename'].isin(df['filename'])]
        # rename label column
        df['LabelName'] = df['general_category']
        # split across category and species for train-val split
        columns2stratify = ['general_category' + '_' + 'common.name']
    if model_type == 'family':
        # list families with fewer images than category min
        too_few = list({k for (k, v) in Counter(df['family']).items() if
                        v < min_per_category})
        # remove general bird images
        too_few.append('Bird')
        # exclude those images from the sample
        df = df[~df['family'].isin(too_few)]
        # remove rows if family is nan
        df = df[df['family'].notna()]
        # save original df to filter through later
        original_df = df
        # list families where all images are included
        fewerMax = list({k for (k, v) in Counter(df['family']).items() if
                         v <= max_per_category})
        # list families with more images than category max
        overMax = list({k for (k, v) in Counter(df['family']).items() if
                        v > max_per_category})
        # shuffle rows
        df = df.sample(frac=1, random_state=1).reset_index(drop=True)
        # setup for representative downsampling across database
        balanced_df = pd.DataFrame(columns=list(df.columns))
        thresh = 100
        for label in overMax:
            temp = balanced_df
            # loop through each family
            subset_df = df[df['family'] == label]
            # list db with image numbers above and below threshold
            fewer = list({k for (k, v) in Counter(subset_df['database']).items() if v <= thresh})
            greater = list({k for (k, v) in Counter(subset_df['database']).items() if v > thresh})
            # add all images for db with image counts below threshold
            lt_df = subset_df[subset_df['database'].isin(fewer)]
            # sample equally from remaining db until total images reaches max per category
            size = max_per_category - lt_df.shape[0]
            ht_subset = subset_df[subset_df['database'].isin(greater)]
            ht_df = pd.DataFrame(columns=list(df.columns))
            while ht_df.shape[0] <= size:
                tmp = ht_subset.groupby('database').sample(n=thresh, replace=False)
                ht_df = pd.concat([ht_df, tmp], ignore_index=True)
                ht_df = ht_df.drop_duplicates()
                if ht_df.shape[0] > size:
                    break
            # add family sample to balanced df
            fam_df = pd.concat([lt_df, ht_df], ignore_index=True)
            balanced_df = pd.concat([temp, fam_df], ignore_index=True)
        # combine representative db sample with families taking all images
        df = pd.concat([balanced_df, df[
            df['family'].isin(fewerMax)]])
        # locate images within df
        df = original_df.loc[original_df['filename'].isin(df['filename'])]
        # create dictionary of family labels
        label2target = {l: t + 1 for t, l in enumerate(df['family'].unique())}
        # set background class
        label2target['empty'] = 0
        background_class = label2target['empty']
        # reverse dictionary for pytorch input
        target2label = {t: l for l, t in label2target.items()}
        pd.options.mode.chained_assignment = None
        # standardize label name
        df['LabelName'] = df['family']
        # stratify across species and family for train/val split
        columns2stratify = ['family' + '_' + 'common.name']
    if model_type == 'species':
        # list species with fewer images than category min
        too_few = list({k for (k, v) in Counter(df['common.name']).items() if v < min_per_category})
        # remove general bird images
        too_few.append('Bird')
        # include these species despite small sample sizes
        always_include = ['White-nosed_Coati', 'Collared_Peccary', 'Jaguarundi', 'Margay', 'Jaguar',
                          'Ocelot']
        # always include images from CFT databases
        cft_include = ['Cattle Fever Tick Program', 'Texas A&M']
        # filter always_include species out of the too_few list
        too_few = [e for e in too_few if e not in always_include]
        df = df[~df['common.name'].isin(too_few)]
        # save original df to filter through later
        original_df = df
        # list species with fewer than max cat images
        fewerMax = list({k for (k, v) in Counter(df['common.name']).items() if v <= max_per_category})
        # list species with greater images than max per category
        overMax = list({k for (k, v) in Counter(df['common.name']).items() if v > max_per_category})
        # shuffle rows before sampling
        df = df.sample(frac=1, random_state=1).reset_index(drop=True)
        # initiate representative sample df
        balanced_df = pd.DataFrame(columns=list(df.columns))
        # set threshold for db images to include
        thresh = 50
        # loop over species with num images greater than max per category
        for label in overMax:
            temp = balanced_df
            subset_df = df[df['common.name'] == label]
            # list db with image numbers above and below threshold
            fewer = list({k for (k, v) in Counter(subset_df['database']).items() if v <= thresh})
            greater = list({k for (k, v) in Counter(subset_df['database']).items() if v > thresh})
            # add all images for db with image counts below threshold
            lt_df = subset_df[subset_df['database'].isin(fewer)]
            # sample equally from remaining db until total images reaches max per species
            size = max_per_category - lt_df.shape[0]
            ht_subset = subset_df[subset_df['database'].isin(greater)]
            ht_df = pd.DataFrame(columns=list(df.columns))
            while ht_df.shape[0] <= size:
                tmp = ht_subset.groupby('database').sample(n=thresh, replace=False)
                ht_df = pd.concat([ht_df, tmp], ignore_index=True)
                ht_df = ht_df.drop_duplicates()
                if ht_df.shape[0] > size:
                    break
            # add species sample to balanced df
            spec_df = pd.concat([lt_df, ht_df], ignore_index=True)
            balanced_df = pd.concat([temp, spec_df], ignore_index=True)
        # combine data and drop duplicates
        df = pd.concat(
            [balanced_df, df[df['common.name'].isin(fewerMax)], df[df['database'].isin(cft_include)]])
        df = df.drop_duplicates()
        # locate images within df
        df = original_df.loc[original_df['filename'].isin(df['filename'])]
        # create dictionary of species labels
        label2target = {l: t + 1 for t, l in enumerate(df['common.name'].unique())}
        # set background class
        label2target['empty'] = 0
        background_class = label2target['empty']
        # reverse dictionary for pytorch input
        target2label = {t: l for l, t in label2target.items()}
        pd.options.mode.chained_assignment = None
        # standardize label name
        df['LabelName'] = df['common.name']
        # stratify across species for train/val split
        columns2stratify = ['common.name']
    #TODO: create balanced sample based on species for the pig-only and family models
    if model_type == 'pig_only':
        too_few = list({k for (k, v) in Counter(df['family']).items() if
                        v < min_per_category})  # list families with fewer images than category min
        too_few.append('Bird')  # remove general bird images
        df = df[~df['family'].isin(too_few)]  # exclude those images from the sample
        df = df[df['family'].notna()]  # remove rows if family is nan
        original_df = df  # save original df to filter through later
        fewerMax = list({k for (k, v) in Counter(df['family']).items() if
                         v <= max_per_category})  # list families where all images are included
        overMax = list({k for (k, v) in Counter(df['family']).items() if
                        v > max_per_category})  # list families with more images than category max
        df = df.sample(frac=1, random_state=1).reset_index(drop=True)  # shuffle rows
        balanced_df = pd.DataFrame(columns=list(df.columns))  # setup for representative downsampling across database
        thresh = 100
        for label in overMax:
            temp = balanced_df
            subset_df = df[df['family'] == label]
            # list db with image numbers above and below threshold
            fewer = list({k for (k, v) in Counter(subset_df['database']).items() if v <= thresh})
            greater = list({k for (k, v) in Counter(subset_df['database']).items() if v > thresh})
            # add all images for db with image counts below threshold
            lt_df = subset_df[subset_df['database'].isin(fewer)]
            # sample equally from remaining db until total images reaches max per category
            size = max_per_category - lt_df.shape[0]
            ht_subset = subset_df[subset_df['database'].isin(greater)]
            ht_df = pd.DataFrame(columns=list(df.columns))
            while ht_df.shape[0] <= size:
                tmp = ht_subset.groupby('database').sample(n=thresh, replace=False)
                ht_df = pd.concat([ht_df, tmp], ignore_index=True)
                ht_df = ht_df.drop_duplicates()
                if ht_df.shape[0] > size:
                    break
            # add family sample to balanced df
            fam_df = pd.concat([lt_df, ht_df], ignore_index=True)
            balanced_df = pd.concat([temp, fam_df], ignore_index=True)
        df = pd.concat([balanced_df, df[df['family'].isin(fewerMax)],
                        df[df[
                               'family'] == 'Suidae']]).drop_duplicates()  # combine dfs, including all available pig images
        df = original_df.loc[original_df['filename'].isin(df['filename'])]  # locate images within df
        label2target = {l: t + 1 for t, l in enumerate(df['family'].unique())}  # create dictionary of family labels
        label2target['empty'] = 0  # set background class
        background_class = label2target['empty']
        target2label = {t: l for l, t in label2target.items()}  # reverse dictionary for pytorch input
        pd.options.mode.chained_assignment = None
        df['LabelName'] = df['family']  # standardize label name
        columns2stratify = ['family']
    return df, label2target, target2label, columns2stratify

# split df into training / validation sets
def split_df(df, columns2stratify):
    """
    Takes df, columns2stratify output from the wrangle_df function and splits the dataset by the stratified column.
    70% of total data is allocated to training, while 15% each is allocated to validation and testing.
    """

    df_unique_filename = df.drop_duplicates(subset='filename', keep='first')
    # split 70% of images into training set
    trn_ids, rem_ids = train_test_split(df_unique_filename['filename'], shuffle=True,
                                        stratify=df_unique_filename[columns2stratify],
                                        test_size=0.3, random_state=22)
    train_df = df[df['filename'].isin(trn_ids)].reset_index(drop=True)
    rem_df = df[df['filename'].isin(rem_ids)].reset_index(drop=True)
    rem_unique_filename = rem_df.drop_duplicates(subset='filename', keep='first')
    # split remaining 30% evenly between validation and test sets
    val_ids, test_ids = train_test_split(rem_unique_filename['filename'], shuffle=True,
                                        stratify=rem_unique_filename[columns2stratify],
                                        test_size=0.33, random_state=22)
    val_df = rem_df[rem_df['filename'].isin(val_ids)].reset_index(drop=True)
    test_df = rem_df[rem_df['filename'].isin(test_ids)].reset_index(drop=True)
    return train_df, val_df, test_df

# Create PyTorch dataset
class DetectDataset(torch.utils.data.Dataset):
    """
    Builds dataset with images and their respective targets, bounding boxes and class labels.
    DF must include: filename containing pathway to individual images; bbox ccordinates in format proportional to
    image size (i.e. all bbox coordinates [0,1]) with xmin, ymin corresponding to upper left corner and
    xmax, ymax corresponding to lower right corner.
    Images are resized, channels converted, and augmented according to data augmentation pipelines defined below.
    Bboxes also undergo corresponding data augmentation.
    Each filename corresponds to a 'target' dict of bboxes and labels.
    Images and targets are returned as Tensors.
    """

    def __init__(self, df, image_dir, w, h, transform):
        self.image_dir = image_dir
        self.df = df
        self.image_infos = df.filename.unique()
        self.w = w
        self.h = h
        self.transform = transform

    def __getitem__(self, item):
        #create image id
        image_id = self.image_infos[item]
        # create full path to open each image file
        img_path = os.path.join(self.image_dir, image_id).replace("\\", "/")
        # open image
        img = cv2.imread(img_path)
        # reformat color channels
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # resize image so bboxes can also be converted
        img = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)/255.
        #img = Image.open(img_path).convert("RGB").resize((self.w, self.h), resample=Image.Resampling.BILINEAR)
        #img = np.array(img, dtype="float32")/255.
        # filter df rows for img
        df = self.df
        data = df[df['filename'] == image_id]
        # extract label names
        labels = data['LabelName'].values.tolist()
        # extract bbox coordinates
        data = data[['XMin', 'YMin', 'XMax', 'YMax']].values
        # convert to absolute values for model input
        data[:,[0,2]] *= self.w
        data[:,[1,3]] *= self.h
        # convert coordinates to list
        boxes = data.tolist()
        # convert bboxes and labels to a tensor dictionary
        target = {
            'boxes': boxes,
            'labels': torch.tensor([label2target[i] for i in labels]).long()
        }
        # apply data augmentation
        if self.transform is not None:
            augmented = self.transform(image=img, bboxes=target['boxes'], labels=labels)
            img = augmented['image']
            target['boxes'] = augmented['bboxes']
        target['boxes'] = torch.tensor(target['boxes']).float() #ToTensorV2() isn't working on bboxes
        return img, target

    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def __len__(self):
        return len(self.image_infos)


# define data augmentation pipelines
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Affine(rotate=(-20, 20), fit_output=True, p=0.3),
    A.Affine(shear=(-20,20), fit_output=True, p=0.3),
    A.RandomBrightnessContrast(brightness_by_max=True, p=0.3),
    A.HueSaturationValue(p=0.3),
    A.RandomSizedBBoxSafeCrop(height=307, width=408, erosion_rate=0.2, p=0.5),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
)
val_transform = A.Compose([
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
)

# Plot images
COLORS = np.random.randint(0, 255, size=(80, 3),dtype="uint8")
# manually adjust score threshold?
def show_img_bbox(img, targets, score_threshold=0.7):
    #convert image format to PIL
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
    for i,tg in enumerate(boxes):
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
def get_model(num_classes):
    # initialize model
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model.to(device)

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
        assert math.isclose(sum(s.values()), 1, abs_tol=0.001) == True
        # assign sample weight to each image in the dataset
        train_unique = train_df.drop_duplicates(subset='filename', keep='first')
        sample_weights = train_unique.LabelName.map(s).tolist()
        # create weighted random sampler
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        # define data loaders
        train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=train_ds.collate_fn, drop_last=True,
                                  sampler=sampler)
        # do not oversample for validation, just for training
        val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=train_ds.collate_fn, drop_last=True)
    # Balance class weights for other models
    else:
        # set up class labels
        s = dict(Counter(train_df['LabelName']))
        # weight all other classes equally by num of classes
        s = {x: (1 / (num_classes - 1)) for x in s}
        # sanity check weighting was done correctly
        assert math.isclose(sum(s.values()), 1, abs_tol=0.001) == True
        # assign sample weight to each image in the dataset
        train_unique = train_df.drop_duplicates(subset='filename', keep='first')
        sample_weights = train_unique.LabelName.map(s).tolist()
        # create weighted random sampler
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        # define data loaders
        train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=train_ds.collate_fn, drop_last=True,
                                  sampler=sampler)
        # do not oversample for validation, just for training
        val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=train_ds.collate_fn, drop_last=True)
    return train_loader, val_loader

# try new dataloaders function w/o oversampling pigs
def get_dataloaders_even(train_df, train_ds, val_ds, batch_size, num_classes):
    # set up class labels
    s = dict(Counter(train_df['LabelName']))
    # weight all other classes equally by num of classes
    s = {x: (1 / (num_classes - 1)) for x in s}
    # sanity check weighting was done correctly
    assert math.isclose(sum(s.values()), 1, abs_tol=0.001) == True
    # assign sample weight to each image in the dataset
    train_unique = train_df.drop_duplicates(subset='filename', keep='first')
    sample_weights = train_unique.LabelName.map(s).tolist()
    # create weighted random sampler
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    # define data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, collate_fn=train_ds.collate_fn, drop_last=True,
                              sampler=sampler)
    # do not oversample for validation, just for training
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=train_ds.collate_fn, drop_last=True)
    return train_loader, val_loader

# obtain learning rate
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# define checkpoint functions
def create_checkpoint(model, optimizer, epoch, lr_scheduler, loss_history, best_loss, model_type, num_classes, label2target):
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

def load_checkpoint(checkpoint_file):
    print(" Loading saved model state")
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['scheduler'])
    epoch = checkpoint['epoch']
    loss_history = checkpoint['loss_history']
    best_loss = checkpoint['best_loss']
    model_type = checkpoint['model_type']
    label2target = checkpoint['label2target']
    return model, optimizer, lr_scheduler, epoch, loss_history, best_loss, model_type, label2target

# plot losses
def plot_losses(model_type, loss_history):
    # extract losses and number of epochs
    train_loss = [loss.detach().numpy() for loss in loss_history['train']]
    val_loss = [loss.cpu().numpy() for loss in loss_history['val']]
    epochs = range(1, len(train_loss) + 1)
    # format and plot
    plt.plot(epochs, train_loss, 'bo', label='Train Loss')
    plt.plot(epochs, val_loss, 'b', label='Val Loss')
    plt.title(model_type + "Faster R-CNN Loss History")
    plt.legend()
    plt.figure()

#TODO: update evaluation functions here or delete

# make predictions
def decode_output(output, labels_as_numbers = False):
    output = output[0]
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
def deploy(df, w=408, h=307):
    model.eval()
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
        output = model(image)
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

# filter predictions with low probability scores
def filter_preds(output, threshold):
    """
    filter output based on probability score; exclude predictions less than threshold
    :param output: model output of all predictions for a particular image
    :param threshold: probability score below which to exclude all predictions
    :return:
    """

    # format prediction data
    bbs = output['boxes'].cpu().detach()
    labels = output['labels'].cpu().detach()
    confs = output['scores'].cpu().detach()

    # id indicies of tensors to include in evaluation
    idx = torch.where(confs > threshold)

    # filter to predictions that meet the threshold
    bbs, labels, confs = [tensor[idx] for tensor in [bbs, labels, confs]]

    return bbs, labels, confs


# define intersection over union function
def intersect_over_union(bbs, tbs):
    """
    Calculates intersection over union
    :param bbs: set of prediction bounding boxes for an image
    :param tbs: set of true target bounding boxes for an image
    :return: intersection over union
    """
    # shape is (N,4) where N is number of bboxes per image
    pred_xmin = bbs[...,0:1] # slice tensor to maintain (N,1) shape
    pred_ymin = bbs[...,1:2]
    pred_xmax = bbs[...,2:3]
    pred_ymax = bbs[...,3:4]
    target_xmin = tbs[..., 0:1]
    target_ymin = tbs[..., 1:2]
    target_xmax = tbs[..., 2:3]
    target_ymax = tbs[..., 3:4]

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
    union = pred_area + target_area - intersect + 1e-6 # add numeric stabilizer in case union = 0

    return intersect / union

#TODO: add manual non-max suppression function
def non_max_suppression(bbs, labels, confs, iou_threshold):
    """
    Perform non-maximum suppression on overlapping predictions based on
    provided IoU threshold

    :param bbs: predicted bounding boxes
    :param labels: predicted class labels
    :param confs: probability score
    :param iou_threshold: intersection over union to consider
    :return:
    """

    # boxes are returned from the model sorted by confidence score, so start with the first box
    ref_box = bbs[0]
    ref_class = labels[0]

    # first compare predictions of the same class


