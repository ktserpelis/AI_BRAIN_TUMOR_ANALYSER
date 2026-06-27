import nibabel as nb
import numpy as np 
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


### 
# We first implement the whole proccess for one 
# volume in order to refine and perfect it before we do it 
# for all the volumes we are gonna use for training  
###

DATA_PATH = 'data/BraTS2020_TrainingData/'

# Load the images seg, t1ce, t1, t2 using nibabel
# We are not gonna use t1 more info on why is on the report
test_img_seg = nb.load(DATA_PATH + 'BraTS20_Training_001/BraTS20_Training_001_seg.nii')
test_img_t1ce = nb.load(DATA_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t1ce.nii')
test_img_flair = nb.load(DATA_PATH + 'BraTS20_Training_001/BraTS20_Training_001_flair.nii')
test_img_t2 = nb.load(DATA_PATH + 'BraTS20_Training_001/BraTS20_Training_001_t2.nii')

# For validation purposes we print the dtype of the loaded img before converting to float64 type 
print("Original image dtype:", test_img_seg.get_data_dtype())

# We use nibabel getfdata() to make sure the imgs are converted to numpy arrays with float64 objects inside
# This is the final data we will be using 
test_data_seg = test_img_seg.get_fdata()
test_data_flair = test_img_flair.get_fdata()
test_data_t1ce = test_img_t1ce.get_fdata()
test_data_t2= test_img_t2.get_fdata()

# For validation purposes we print the dtype of the final data 
# in order to make sure they are float64 objects
print("Final data dtype:", test_data_flair.dtype)

###
# After loading and casting the data we need to make sure 
# that we normalize the pixels so as not to get huge diferrences in the values
# when they are multiplied with the weight of the NN (neural network)
# 
# We are going to use z-score normalizitation (more on why on the report)
###

data_array = [test_data_flair, test_data_t1ce, test_data_t2]

for data in data_array:

    # We need to make sure the black pixels that dominate the img dont 
    # bias the mean so as to keep the brain pixels in focus (more on report)
    # We use boolean masking to suppress the black pixels 
    data_mask = (data != 0)

    mean = np.mean(data[data_mask])
    std_dv = np.std(data[data_mask])

    # This is the final standardized data
    z_data = (data[data_mask] - mean) / std_dv

    # After z-score we need to check if std_dv = 1 and mean = 0 
    # if this are true then the data is normalized correctly
    z_data_mean = np.mean(z_data)
    z_data_std_dv = np.std(z_data)
    
    assert np.isclose(z_data_mean, 0.0, atol=1e-5), f"ERROR: Mean is {z_data_mean}, not 0"
    assert np.isclose(z_data_std_dv, 1.0, atol=1e-5), f"ERROR: Standard deviation is {z_data_std_dv}, not 1"



###
# We use stack() to combine the volumes on the z axis to channels so as to give the 
# model spatial recognition of the slices , then we divide them to patches (more on why on the report)
#
# We will use float32 accuracy (more on why on the report)
###

# Convert to f32
test_data_flair = test_data_flair.astype(np.float32)
test_data_t1ce = test_data_t1ce.astype(np.float32)
test_data_t2 = test_data_t2.astype(np.float32)

# Combine data 
combined_data = np.stack([test_data_flair, test_data_t1ce, test_data_t2], axis=3)

# Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches. 
# cropping x, y, and z
# combined_x=combined_x[24:216, 24:216, 13:141]

combined_x=combined_data[56:184, 56:184, 13:141] #Crop to 128x128x128x4

# Do the same for mask
test_mask = test_data_seg[56:184, 56:184, 13:141]

# Printing the patches
import random
n_slice=random.randint(0, test_data_seg.shape[2])

plt.figure(figsize=(12, 8))

plt.subplot(231)
plt.imshow(combined_data[:,:,n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(232)
plt.imshow(combined_data[:,:,n_slice, 1], cmap='gray')
plt.title('Image t1ce')
plt.subplot(233)
plt.imshow(combined_data[:,:,n_slice, 2], cmap='gray')
plt.title('Image t2')
plt.subplot(234)
plt.imshow(test_mask[:,:,n_slice])
plt.title('Mask')
plt.show()

###
# In order to finalize the data and get it ready for the ml model we will use
# one-hot encoding to turn the seg data from categorical to binary columns(more on report)
###

print(np.unique(test_data_seg))
# First we convert the 4 to 3 in order to use automated pytorch function
test_data_seg[test_data_seg == 4] = 3
print(np.unique(test_data_seg))

# We convert the numpy array to pytorch tensor
test_data_seg_tensor = torch.from_numpy(test_data_seg)

# one_hot function only works with LongTensor object 
# before conversion our data was float64 
test_data_seg_tensor = test_data_seg_tensor.long()
test_data_seg_one_hot_tensor = F.one_hot(test_data_seg_tensor, num_classes=4)

# To finalize the data we need to permuate the tensor to have the z axis (4) in the front of the shape
# and also we need to convert back to float64
final_test_data_tensor = test_data_seg_one_hot_tensor.permute(3, 0, 1, 2)
# This is the final data we will use for training
final_test_data_tensor = final_test_data_tensor.float()

# Final check for correct tensor shape 
assert final_test_data_tensor.shape == (4, 240, 240, 155), f"Final seg tensor shape is: {final_test_data_tensor}, not (4 , 240, 240, 155)"


# Final assertion checks (shapes, dtypes) TODO