from pie_data import PIE
import numpy as np
from PIL import ImageTransform

pie_path = './'
pie = PIE(data_path=pie_path)
traj_sec_train = pie.generate_data_trajectory_sequence("train")
traj_sec_val = pie.generate_data_trajectory_sequence("val")
traj_sec_test = pie.generate_data_trajectory_sequence("test")

from tqdm import tqdm
from PIL import Image, ImageDraw

def save_npy_clips(traj_sec_dict, save_dir, down_samp_size):
    
    # keys are: dict_keys(['image', 'bbox', 'occlusion', 'intention_prob', 'intention_binary', 'ped_id'])
    images = traj_sec_dict['image']
    pedestrians = traj_sec_dict['ped_id']
    bbox_all = traj_sec_dict['bbox']
    for i in tqdm(range(len(images))):
        cur_image_file_list = images[i]
        cur_ped_id = pedestrians[i][0][0]
        
        # adding the bbox in the frame instead of text prompt
        cur_bbox_list = bbox_all[i]
        # break
        # print(cur_ped_id)
        clip = []
        # print(cur_image_file_list)
        for j, f in enumerate(cur_image_file_list):
            cur_bbox = cur_bbox_list[j]
            image = Image.open(f)
            draw = ImageDraw.Draw(image)
            draw.rectangle(cur_bbox, outline='red', width=3)

            # flip images horizontally
            # image = flip_img_horizontally(image)

            # add gaussian noise
            # image = add_gaussian_noise(image)

            # add perspective augmentation
            image = warp_image(image)

            clip.append(image)
        
        # down-sampling 
        total_frames = len(clip)
        indices = np.arange(0, total_frames, total_frames / down_samp_size).astype(int)
        clip = np.array(clip)
        clip = clip[indices]

        # saving the np array with ped_id as file name
        file_name = cur_ped_id
        np.save(save_dir + file_name, clip)


def flip_img_horizontally(pil_img):
    return pil_img.transpose(Image.FLIP_LEFT_RIGHT)

def add_gaussian_noise(pil_img):
    image_array = np.array(pil_img)
    mean = 0
    std_dev = 20

    # Generate Gaussian noise
    gaussian_noise = np.random.normal(mean, std_dev, image_array.shape)

    # Add the Gaussian noise to the image
    noisy_image_array = image_array + gaussian_noise

    # Clip the values to be in the proper range for image data (0 to 255)
    noisy_image_array = np.clip(noisy_image_array, 0, 255)

    # Convert the noisy image back to a Pillow Image object
    noisy_image = Image.fromarray(noisy_image_array.astype('uint8'))
    return noisy_image

def warp_image(pil_image):
    width, height = pil_image.size

    # original_points = (
    #     0, 0,               # upper left corner
    #     0, height,          # lower left corner
    #     width, height,          # lower right corner
    #     width, 0,           # upper right corner

    # )

    # Define the points for where you want the corners to be after the transformation
    # This will create the perspective effect
    margin = 200
    new_points = (
        margin, margin,         # upper left corner
        0, height,              # lower left corner
        width, height,          # lower right corner
        width - margin, margin,               # upper right corner

    )

    # Perform the perspective transformation
    perspective_transform = ImageTransform.QuadTransform(new_points)
    transformed_image = pil_image.transform(pil_image.size, perspective_transform)
    return transformed_image

if __name__ == '__main__':
    train_clips_dir = "SET THE TRAIN DIR HERE"
    test_clips_dir = "SET THE TEST DIR HERE"
    val_clips_dir = "SET THE VAL DIR HERE"
    SAMPLING_RATE = 16
    save_npy_clips(traj_sec_train, train_clips_dir, SAMPLING_RATE)
    save_npy_clips(traj_sec_test, test_clips_dir, SAMPLING_RATE)
    save_npy_clips(traj_sec_val, val_clips_dir, SAMPLING_RATE)