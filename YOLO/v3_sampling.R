# CameraTrapDetectoR V3 sample generation
# Amira Burns
# May 15 2024

library(dplyr)
library(clhs)

# load data
df <- data.table::fread("/path/to/annotations/database")

# count images in each species category
sp_img_cts <- df %>% 
  dplyr::group_by(common.name.combined) %>%
  dplyr::summarise(n_imgs = length(unique(filename)))
sp_samps <- sp_img_cts$common.name.combined[sp_img_cts$n_imgs > 300]

# remove images with less than 300 images 
sp_df <- df %>%
  dplyr::filter(common.name.combined %in% sp_samps)

# set class size
N <- 5000

# set placeholder lists for final dfs
train_list <- list()
val_list <- list()

classes <- unique(sp_df$common.name.combined)

for(i in 1:length(classes)) {
  
  # print message
  print(paste0("Creating training sample for ", classes[i]))
  
  # subset to single class
  class_df <- sp_df[sp_df$common.name.combined == classes[i],]
  
  # take all images if less than class max
  if(nrow(class_df) < N) {
    class_samp <- class_df
  } 
  
  # else use CLHS to downsample across categories of interest
  else {
    # count sitewise images
    site_imgs <- class_df %>%
      dplyr::group_by(site) %>%
      dplyr::summarise(n_imgs = length(unique(filename)))
    
    # immediately extract all images from sites with less than 5% of total
    full_sites <- class_df %>%
      dplyr::filter(site %in% (site_imgs$site[site_imgs$n_imgs < N*0.05]))
    
    # separate out majority-site images to downsample
    maj_sites <- class_df[!(class_df$filename %in% full_sites$filename),]
    
    # define sample size
    Nsamp <- N - nrow(full_sites)
    
    # select columns to sample over
    downsamp <- maj_sites %>%
      dplyr::select(c(common.name, bbox.area, site, aspect.ratio, contrast, complexity, month, day.time))
    
    # run clhs algorithm 
    clhs_ids <- clhs::clhs(downsamp, size = Nsamp, use.cpp = FALSE, simple = TRUE)
    
    # pull all rows associated with those files from the oversampled data frame
    clhs_df <- class_df[clhs_ids,]
    clhs_imgs <- class_df[(class_df$filename %in% clhs_df$filename),]
    
    # combine clhs sample and minority sites
    class_samp <- dplyr::bind_rows(full_sites, clhs_imgs)
  }
  
  # print message
  print(paste0("Splitting ", classes[i], " sample into training/validation sets."))
  
  # pull out cats to stratify in train/val split
  downsamp <- class_samp %>%
    dplyr::select(c(common.name, bbox.area, site, aspect.ratio, contrast, complexity, month, day.time))
  
  # sample 80% for training
  train_clhs_ids <- clhs::clhs(downsamp, size = 0.8*nrow(downsamp), use.cpp = FALSE, simple = TRUE)
  train_samp <- class_samp[train_clhs_ids,]
  train_df <- class_samp[(class_samp$filename %in% train_samp$filename),]
  
  # place remaining images in validation set
  val_df <- class_samp[!(class_samp$filename %in% train_df$filename),]
  
  train_list[[i]] <- train_df
  val_list[[i]] <- val_df
}


# unpack lists into df
train_df <- purrr::reduce(train_list, bind_rows)
val_df <- purrr::reduct(val_list, bind_rows)