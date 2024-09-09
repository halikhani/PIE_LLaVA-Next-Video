# generating train, val, and test clips for fine tuning (saving in .npy format for videos)
# Note1: resizing the frames in half size
# Note2: down sampling to 8 for less compute heavy operations 
from tqdm import tqdm
def save_npy_clips(traj_sec_dict, save_dir, down_samp_size):
    
    # keys are: dict_keys(['image', 'bbox', 'occlusion', 'intention_prob', 'intention_binary', 'ped_id'])
    images = traj_sec_dict['image']
    pedestrians = traj_sec_dict['ped_id']

    for i in tqdm(range(len(images))):
        cur_image_file_list = images[i]
        cur_ped_id = pedestrians[i][0]
        
        clip = []
        for f in cur_image_file_list:
            image = Image.open(f)
            # Resize the image
            new_size = (540, 960) 
            resized_image = image.resize(new_size, Image.LANCZOS)
            image_array = np.array(resized_image)
            clip.append(image_array)
        
        # down-sampling 
        total_frames = len(clip)
        indices = np.arange(0, total_frames, total_frames / down_samp_size).astype(int)
        clip = np.array(clip)
        clip = clip[indices]

        # saving the np array with ped_id as file name
        file_name = ped_id
        np.save(save_dir + file_name, clip)


save_npy_clips(traj_sec_train, './hamidreza_files/train_clips/', 10)
save_npy_clips(traj_sec_test, './hamidreza_files/test_clips/', 10)
save_npy_clips(traj_sec_val, './hamidreza_files/val_clips/', 10)