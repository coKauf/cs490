import os
import csv
import cv2
import pandas as pd
from selenium import webdriver
from bs4 import BeautifulSoup
import subprocess
import shutil
from PIL import Image
from moviepy.editor import VideoFileClip
from concurrent.futures import ProcessPoolExecutor
import shutil
import numpy as np


train_folds = [
    ['0030', '4008', '0025', '2008', '0018', '4003', '2006', '5018', '5019', '5034', '1008', '5010', '5006', '5004', '5003', '5002'],
    ['0030', '1006', '4008', '0029', '0025', '0018', '4003', '5018', '2013', '5031', '5019', '5034', '5010', '5006', '5004', '5003'],
    ['0030', '1006', '0029', '0025', '2008', '0018', '2006', '5018', '2013', '5031', '5019', '1008', '5006', '5004', '5003', '5002'],
    ['0030', '1006', '4008', '0029', '2008', '0018', '4003', '2006', '2013', '5031', '5019', '5034', '1008', '5010', '5004', '5002'],
    ['1006', '4008', '0029', '0025', '2008', '4003', '2006', '5018', '2013', '5031', '5034', '1008', '5010', '5006', '5003', '5002']
]

test_folds = [
    ['1006', '0029', '2013', '5031'],
    ['2008', '2006', '1008', '5002'],
    ['4008', '4003', '5034', '5010'],
    ['0025', '5018', '5006', '5003'],
    ['0030', '0018', '5019', '5004']
]


MEAN = [85.66148864825567, 81.4623240158293, 78.21916069613563]
STD = [66.15624651183622, 65.53667122926284, 69.54996902718322]

train_videos = ['0018', '0025', '0030', '1006', '2008', '2013', '4003', '4008', '5002', '5003', '5004', '5006', '5010', '5018', '5019', '5031']
test_videos = ['0029', '1008', '2006', '5034']

def check_image_counts():
    # folders = ['dsav360/test', 'dsav360/train']
    # counts = {
    #     'audio_energy_maps': {},
    #     'saliency_maps': {},
    #     'videos': {}
    # }

    # for folder in folders:
    #     for content in os.listdir(folder):
    #         if content == 'head_data':
    #             continue
    #         content_dir = os.path.join(folder, content)
    #         for video in os.listdir(content_dir):
    #             dir = os.path.join(content_dir, video)
    #             count = max_number = 0
    #             for file in os.listdir(dir):
    #                 path = os.path.join(dir, file)
    #                 if path.endswith('.png'):
    #                     max_number = max(int(file[:4]), max_number)
    #                     count += 1

    #             counts[content][video] = (count, max_number)
        
    # video_counts = {key: [(), (), ()] for key in counts['videos'].keys()}


    # for i, k in enumerate(counts.keys()):
    #     print(k)
    #     for video, count in counts[k].items():
    #         video_counts[video][i] = count
    # 
    # for k, v in video_counts.items():
    #     print(f"video: {k}")
    #     print(f'counts: {v}')

    for video in test_videos + train_videos:
        test_train = 'test' if video in test_videos else 'train'
        results = []
        if not os.path.exists(f'dsav360/{test_train}/audio_energy_maps/{video}'):
            continue
        for content_type in ['videos', 'saliency_maps', 'audio_energy_maps']:
            cnt_cmd = f'ls -1 dsav360/{test_train}/{content_type}/{video} | wc -l'
            result = subprocess.run(cnt_cmd, shell=True, text=True, capture_output=True)
            results.append((content_type, result.stdout.strip()))
        
            # expected_files = {f"{str(i).zfill(4)}.png" for i in range(1800)}
            # actual_files = set([x for x in os.listdir(f'dsav360/{test_train}/{content_type}/{video}') if x.endswith('.png')])
            # missing_files = expected_files - actual_files
            # extra_files = actual_files - expected_files

            # print(f"Count missing files in {video} in {content_type} folder: {len(missing_files)}")
            # print(f"Count extra files in {video} in {content_type} folder: {len(extra_files)}")

        print(f'{video}: {results}')






def get_DSAV360_video_features():
    url = 'https://graphics.unizar.es/projects/D-SAV360/dataset_index.html'

    # Initialize a Selenium WebDriver
    driver = webdriver.Chrome()  # You need to have chromedriver installed in your system

    # Load the webpage
    driver.get(url)

    # Wait for a few seconds to allow dynamic content to load (adjust as needed)
    driver.implicitly_wait(5)

    # Get the page source after dynamic content is loaded
    html_content = driver.page_source

    # Close the Selenium WebDriver
    driver.quit()

    soup = BeautifulSoup(html_content, 'html.parser')


    # Find the video table using its ID
    video_table = soup.find('table', {'id': 'videoTable'})

    # Extract video name, start timestamps and start cube positions from the table
    # formatted as tuple(index, name, timestamp, cube_position)
    video_features = []

    if video_table:
        # Iterate over each row in the table (skipping the header row)
        for row in video_table.find_all('tr')[1:]:
            # Extract the timestamp from the third cell and cube position from the fourth cell
            data = row.find_all('td')
            name_cell = data[0]
            timestamp_cell = data[2]
            cube_pos_cell = data[3]
            monoscopic = data[4]
            stereoscopic = data[5]
            depth = data[6]
            
            # Extracting text content from the cells
            name_text = name_cell.get_text(strip=True)
            timestamp_text = timestamp_cell.get_text(strip=True)
            cube_pos_text = cube_pos_cell.get_text(strip=True)
            monoscopic_text = monoscopic.get_text(strip=True)
            stereoscopic_text = stereoscopic.get_text(strip=True)
            depth_text = depth.get_text(strip=True)
            
            # Extract the timestamp and cube position part (e.g., "u = 0.625")
            index = len(video_features) + 1
            name = name_text.split(":")[-1].strip()
            timestamp = timestamp_text.split(":")[-1].strip()
            cube_position = cube_pos_text.split(":")[-1].strip()
            monoscopic = monoscopic_text.split(":")[-1].strip()
            stereoscopic = stereoscopic_text.split(":")[-1].strip()
            depth = depth_text.split(":")[-1].strip()

            video_features.append([index, name, timestamp, cube_position, monoscopic, stereoscopic, depth])


    # Write the extracted data to a CSV file
    with open('video_table.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(['Index', 'Name', 'Timestamp', 'Cube Position', 'Monoscopic', 'Stereoscopic', 'Depth'])
        # Write data
        csv_writer.writerows(video_features)

def get_frame_rate(video_path):
    # Run ffprobe command to get frame rate
    ffprobe_command = f"ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 {video_path}"
    result = subprocess.run(ffprobe_command, shell=True, stdout=subprocess.PIPE, text=True)
    
    # Extract and return the frame rate
    frame_rate = result.stdout.strip()
    return frame_rate

def process_DSAV360_images_audio():
    folders = ['dsav360/train/videos', 'dsav360/test/videos']
    # folders = ['dsav360/test/videos']

    # video_table = 'dsav360/video_table.csv'
    # if not os.path.exists(video_table):
    #     get_DSAV360_video_features()
    
    # df = pd.read_csv(video_table)
    # for index, row in df.iterrows():
    #     name = str(row['Name']).zfill(4)
    #     start = row['Timestamp']
    #     if name[0] in ['4', '5']:
    #         continue
    for folder in folders:
        for video in os.listdir(folder):
            video_folder = os.path.join(folder, video)
            if os.path.isdir(video_folder):
                input_path = os.path.join(video_folder, f'{video}.mp4')

                # Get all frames
                frame_command = f"ffmpeg -i {input_path} -start_number 0 -vsync 0 {video_folder}/%04d.png" 
                ## Get every 8 frames
                # frame_command = f"ffmpeg -i {input_path} -ss {start} -to {start + 30} -vf \"select=not(mod(n\,8))\" -vsync vfr -start_number 0 {output_dir}/%04d.png"
                subprocess.run(frame_command, shell=True)

                # audio_command = f"cp {input_dir}/{name}.wav {output_dir}"
                # subprocess.run(audio_command, shell=True)

def trim_DSAV360_AV():
    input_folder = 'dsav360/360videos_with_ambisonic/raw'
    output_folder = 'dsav360/360videos_with_ambisonic/processed/mono'

    video_table = 'dsav360/video_table.csv'
    if not os.path.exists(video_table):
        get_DSAV360_video_features()
    
    df = pd.read_csv(video_table)
    for index, row in df.iterrows():
        name = str(row['Name']).zfill(4)
        start = row['Timestamp']
        has_stereoscopic = True if row['Stereoscopic'] == 'Yes' else False
        if not has_stereoscopic:
            continue
        
        output_dir = os.path.join(output_folder, name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # delete_mp4_cmd = f'rm {output_dir}/*.mp4'
        # subprocess.run(delete_mp4_cmd, shell=True)
        # delete_wav_cmd = f'rm {output_dir}/*.wav'
        # subprocess.run(delete_wav_cmd, shell=True)
        
        input_dir = os.path.join(input_folder, name)
        vid_input_path = os.path.join(input_dir, "".join([name, '_mono_60fps.mp4']))
        audio_input_path = os.path.join(input_dir, "".join([name, '.wav']))

        video_cmd = f'ffmpeg -i {vid_input_path} -ss 00:00:{str(start).zfill(2)} -t 30 {output_dir}/{name}.mp4'
        subprocess.run(video_cmd, shell=True)

        audio_cmd = f'ffmpeg -i {audio_input_path} -ss 00:00:{str(start).zfill(2)} -t 30 {output_dir}/{name}.wav'
        subprocess.run(audio_cmd, shell=True)

def remove_unused_DSAV360_frames(base_folder):
    saliency_folder = base_folder + '/saliency_maps_8fps'
    AEM_folder = base_folder + '/audio_energy_maps'
    # video_folder = base_folder + '/videos'

    # get min and max saliency map frame for each video
    frame_dict = {}
    for d in os.listdir(saliency_folder):
        saliency_dir_path = os.path.join(saliency_folder, d)
        if os.path.isdir(saliency_dir_path):
            min_frame, max_frame = '9999', '0000'
            for f in os.listdir(saliency_dir_path):
                frame = f[:4]
                min_frame = min(frame, min_frame)
                max_frame = max(frame, max_frame)
            frame_dict[d] = (min_frame, max_frame)

    for d in os.listdir(AEM_folder):
        AEM_dir_path = os.path.join(AEM_folder, d)
        
        if os.path.isdir(AEM_dir_path):
            for f in os.listdir(AEM_dir_path):
                min_frame, max_frame = frame_dict[d]
                if min_frame > f[:4] or f[:4] > max_frame:
                    del_cmd = f'rm {AEM_dir_path}/{f}'
                    subprocess.run(del_cmd, shell=True)
    
    # for d in os.listdir(video_folder):
    #     video_dir_path = os.path.join(video_folder, d)
        
    #     if os.path.isdir(video_dir_path):
    #         for f in os.listdir(video_dir_path):
    #             min_frame, max_frame = frame_dict[d]
    #             if min_frame > f[:4] or f[:4] > max_frame:
    #                 del_cmd = f'rm {video_dir_path}/{f}'
    #                 subprocess.run(del_cmd, shell=True)
    
    for d in os.listdir(saliency_folder):
        saliency_dir_path = os.path.join(saliency_folder, d)
        if os.path.isdir(saliency_dir_path):
            for f in os.listdir(saliency_dir_path):
                min_frame = frame_dict[d][0]
                new_name = str(int(f[:4]) - int(min_frame)).zfill(4) + '.png'
                old_path = os.path.join(saliency_dir_path, f)
                new_path = os.path.join(saliency_dir_path, new_name)
                os.rename(old_path, new_path)
        
        AEM_dir_path = os.path.join(AEM_folder, d)
        if os.path.isdir(AEM_dir_path):
            for f in os.listdir(AEM_dir_path):
                min_frame = frame_dict[d][0]
                new_name = str(int(f[:4]) - int(min_frame)).zfill(4) + '.png'
                old_path = os.path.join(AEM_dir_path, f)
                new_path = os.path.join(AEM_dir_path, new_name)
                os.rename(old_path, new_path)
        
        # video_dir_path = os.path.join(video_folder, d)
        # if os.path.isdir(video_dir_path):
        #     for f in os.listdir(video_dir_path):
        #         min_frame = frame_dict[d][0]
        #         new_name = str(int(f[:4]) - int(min_frame)).zfill(4) + '.png'
        #         old_path = os.path.join(video_dir_path, f)
        #         new_path = os.path.join(video_dir_path, new_name)
        #         os.rename(old_path, new_path)

def process_DSAV360_saliency():
    input_folders = [f'dsav360/{x}/saliency_maps_8fps' for x in ['test', 'train']]

    for input_folder in input_folders: 
        for d in os.listdir(input_folder):
            dir_path = os.path.join(input_folder, d)
            
            if os.path.isdir(dir_path):
                for f in os.listdir(dir_path):
                    if f.endswith('.png') and '_' in f:
                        old_path = os.path.join(dir_path, f)
                        new_name = f.split('_')[-1]
                        new_path = os.path.join(dir_path, new_name)
                        os.rename(old_path, new_path)

# def sample_8fps():
#     folder = 'dsav360/360videos_with_ambisonic/processed/mono'
    
#     for d in sorted(os.listdir(input_folder)):
#         print(d)
#         input_video_folder = os.path.join(input_folder, d)
#         output_dir = os.path.join(output_folder, d)
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)

#         # Get a sorted list of all frame files
#         frame_files = sorted([f for f in os.listdir(input_video_folder) if f.endswith('.png')])
        
#         # Copy every 8th frame
#         for i, frame_file in enumerate(frame_files):
#             if i % 8 == 0:
#                 source = os.path.join(input_video_folder, frame_file)
#                 destination = os.path.join(output_dir, frame_file)
#                 shutil.copy(source, destination)
#         # copy audio files
#         source = os.path.join(input_video_folder, f'{d}.wav')
#         destination = os.path.join(output_dir, f'{d}.wav')
#         shutil.copy(source, destination)

def oversample_to_8fps():
    input_dir = 'dsav360/saliency_maps_8fps'

    for d in sorted(os.listdir(input_dir)):
        output_dir = os.path.join(input_dir, d)

        # Get a sorted list of all frame files
        frame_files = []
        for f in os.listdir(output_dir):
            if f.endswith('.png'):
                frame_files.append(f)
            else:
                os.remove(os.path.join(output_dir, f))

        frame_files.sort()

        for i in range(len(frame_files) - 1, -1, -1):
            old_name = frame_files[i]
            source = os.path.join(output_dir, old_name)
            img = Image.open(source)

            for j in range(8):
                new_name = f"{i * 8 + j:04d}.png"
                destination = os.path.join(output_dir, new_name)
                
                # Save the image under the new name
                img.save(destination)

def extract_frames_and_audio(video_info):
    approx_fps = {
        'dd39herpgXA.mp4' : 30,
        'ey9J7w98wlI.mp4' : 50,
        'Bvu9m__ZX60.mp4' : 30,
        'kZB3KMhqqyI.mp4' : 30,
        'idLVnagjl_s.mp4' : 30,
        '5h95uTtPeck.mp4' : 24,
        '6QUCaLvQ_3I.mp4' : 30,
        'Ngj6C_RMK1g.mp4' : 30,
        'dpfkpZzZvqw.mp4' : 30,
        'fryDy9YcbI4.mp4' : 30,
        'MzcdEI-tSUc.mp4' : 25,
        'nZJGt3ZVg3g.mp4' : 30,
        'RbgxpagCY_c_2.mp4' : 30,
        '8ESEI0bqrJ4.mp4' : 25,
        '8feS1rNYEbg.mp4' : 30,
        'Oue_XEKHq3g.mp4' : 30,
        'gSueCRQO_5g.mp4' : 30,
        'ByBF08H-wDA.mp4' : 30,
        'OZOaN_5ymrc.mp4' : 30,
        'oegasz59U7I.mp4' : 60,
        '1An41lDIJ6Q.mp4' : 25
    }

    filename, input_folder, output_folder = video_info
    video_folder = os.path.join(output_folder, os.path.splitext(filename)[0])
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    input_path = os.path.join(input_folder, filename)
    frame_rate = get_frame_rate(input_path)  # Ensure this function is correctly implemented

    frame_command = f"ffmpeg -i \"{input_path}\" -r {approx_fps[filename]} -start_number 0  \"{os.path.join(video_folder, '%04d.png')}\""
    subprocess.run(frame_command, shell=True)

    audio_command = f"ffmpeg -i \"{input_path}\" -map 0:a -c:a pcm_s16le \"{os.path.join(video_folder, 'audio.wav')}\"" # doesn't work for videos with silent sound
    subprocess.run(audio_command, shell=True)


def generate_silent_audio():
    """Generate a silent audio file matching the duration of the input video."""
    input_dir = 'avs360_test/raw/videos/none'
    output_dir = 'avs360_test/processed/videos/none'
    list_vid = ['idLVnagjl_s', 'ey9J7w98wlI', 'kZB3KMhqqyI', 'MzcdEI-tSUc', '8ESEI0bqrJ4','1An41lDIJ6Q','6QUCaLvQ_3I', '8feS1rNYEbg','ByBF08H-wDA','fryDy9YcbI4', 'RbgxpagCY_c_2','dd39herpgXA']   
    for filename in os.listdir(input_dir):
        if filename.split('.mp4')[0] not in list_vid:
            continue
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.split('.mp4')[0], 'audio.wav')

        # Use ffprobe to get the duration of the video in seconds
        cmd_duration = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input_path]
        result_duration = subprocess.run(cmd_duration, text=True, capture_output=True)
        duration = result_duration.stdout.strip()
        
        # Generate a silent audio file with the same duration as the video
        silent_audio_command = ['ffmpeg', '-f', 'lavfi', '-i', 'anullsrc=r=48000:cl=mono', '-t', duration, '-acodec', 'pcm_s16le', output_path]
        subprocess.run(silent_audio_command)

def process_chao_dataset():
    # process audio_video
    sound_types = ['none', 'mono', 'ambix']
    list_vid = ['idLVnagjl_s', 'ey9J7w98wlI', 'kZB3KMhqqyI', 'MzcdEI-tSUc', '8ESEI0bqrJ4','1An41lDIJ6Q','6QUCaLvQ_3I', '8feS1rNYEbg','ByBF08H-wDA','fryDy9YcbI4', 'RbgxpagCY_c_2','dd39herpgXA']   

    for st in sound_types:
        input_folder = 'avs360_test/raw/videos/' + st
        output_folder = 'avs360_test/processed/videos/' + st

        videos_to_process = [(filename, input_folder, output_folder) for filename in os.listdir(input_folder) if filename.split('/')[-1].split('.mp4')[0] in list_vid]

        # Using ProcessPoolExecutor to parallelize video processing
        with ProcessPoolExecutor() as executor:
            executor.map(extract_frames_and_audio, videos_to_process)


def upsample_salmaps(dir):
    for video in os.listdir(dir):
        print(video)
        d = os.path.join(dir, video)
        if os.path.isdir(d):
            files = os.listdir(d)
            for f in sorted([f for f in files if f.endswith('.png')])[:-1]:
                old_file_path = os.path.join(d, f)
                for i in range(1, 8):
                    new_name = str(int(f[:4]) + i).zfill(4) + '.png'
                    new_file_path = os.path.join(d, new_name)
                    shutil.copy2(old_file_path, new_file_path)

def convert_uv_to_vector(row):
    u, v = row['u'], row['v']
    longitude = u * 360 - 180
    latitude = v * 180 - 90
    longitude_in_radians = np.deg2rad(longitude)
    latitude_in_radians = np.deg2rad(latitude)
    x = np.cos(latitude_in_radians) * np.cos(longitude_in_radians)
    y = np.cos(latitude_in_radians) * np.sin(longitude_in_radians)
    z = np.sin(latitude_in_radians)
    return [x, y, z]

def add_head_vec():
    directory = 'dsav360/head_data'
    for filename in os.listdir(directory):
        if filename.startswith('head_video_') and filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            
            # Apply the conversion to each row
            df['head_pos_vec'] = df.apply(convert_uv_to_vector, axis=1)
            
            # Save the modified DataFrame back to CSV
            df.to_csv(file_path, index=False)

def resize_images(directory, target_size=(512, 256)):
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.png'):
                file_path = os.path.join(root, filename)
                with Image.open(file_path) as img:
                    resized_img = img.resize(target_size, Image.LANCZOS)
                    resized_img.save(file_path) 

def calculate_video_stats(videos):
    sum_pixels = np.zeros(3)  # To store the sum of RGB values
    sum_squares = np.zeros(3)  # To store the sum of squares of RGB values
    count_pixels = 0  # To count the total number of pixels processed

    # First pass: Calculate means
    for video in videos:
        path = os.path.join('dsav360/videos', f'{video}')
        for filename in os.listdir(path):
            if filename.endswith('.png'):
                file_path = os.path.join(path, filename)
                with Image.open(file_path) as img:
                    img = img.convert('RGB')
                    pixels = np.array(img)
                    sum_pixels += np.sum(pixels, axis=(0, 1))
                    count_pixels += pixels.shape[0] * pixels.shape[1]

    mean_pixels = sum_pixels / count_pixels

    # Second pass: Calculate sum of squares for the standard deviation
    for video in videos:
        path = os.path.join('dsav360/videos', f'{video}')
        for filename in os.listdir(path):
            if filename.endswith('.png'):
                file_path = os.path.join(path, filename)
                with Image.open(file_path) as img:
                    img = img.convert('RGB')
                    pixels = np.array(img)
                    sum_squares += np.sum((pixels - mean_pixels) ** 2, axis=(0, 1))

    var_pixels = sum_squares / count_pixels
    std_dev_pixels = np.sqrt(var_pixels)

    return mean_pixels.tolist(), std_dev_pixels.tolist()

def trim_head_data():
    video_table_path = 'dsav360/video_table.csv'
    head_data_dir = 'dsav360/head_data'

    # Load the main csv file
    main_df = pd.read_csv(video_table_path)

    # Iterate over each row in the main DataFrame
    for _, row in main_df.iterrows():
        video_name = str(row['Name']).zfill(4)
        start_timestamp = row['Timestamp']
        end_timestamp = start_timestamp + 30  # Since each video is 30 seconds long

        # Path to the corresponding head video CSV
        head_csv_path = os.path.join(head_data_dir, f'head_video_{video_name}.csv')
        
        # Check if the head video CSV exists
        if os.path.exists(head_csv_path):
            print(f'Processing head data for video {video_name}')
            # Load the head video data
            head_df = pd.read_csv(head_csv_path)
            
            # Filter rows where the 't' column is within the start and end timestamp
            head_df = head_df[(head_df['t'] >= start_timestamp) & (head_df['t'] <= end_timestamp)]
            
            # Overwrite the existing CSV file with the filtered data
            head_df.to_csv(head_csv_path, index=False)
        else:
            print(f"No data file found for video {video_name}")


if __name__ == "__main__":
    # dir = '../AVS360/360_AV_Perception/code/output/saliency/ambix'

    # for video in os.listdir(dir):
    #     d = os.path.join(dir, video)
    #     if os.path.isdir(d):
    #         for f in os.listdir(d):
    #             # Open the image file
    #             path = os.path.join(d, f)
    #             img = Image.open(path)
    #             width, height = img.size

    #             print(f"Width: {width}, Height: {height}")
    # check_image_counts()
    # process_DSAV360_images_audio()
    # process_DSAV360_saliency()
    # process_chao_dataset()
    # trim_DSAV360_AV()
    # remove_unused_DSAV360_frames('dsav360/test')
    # remove_unused_DSAV360_frames('dsav360/train')
    # sample_8fps()
    # upsample_salmaps('dsav360/train/saliency_maps_8fps') # rename 8 fps to be sequential (0, 1, 2, ... instead of 0, 8, 16, ...)
    # upsample_salmaps('dsav360/test/saliency_maps_8fps') # rename 8 fps to be sequential (0, 1, 2, ... instead of 0, 8, 16, ...)
    # process_chao_dataset()
    # generate_silent_audio()
    # add_head_vec()
    for i, videos in enumerate(train_folds):
        mean, std = calculate_video_stats(videos)
        mean = [mu / 255.0 for mu in mean]
        std = [sigma / 255.0 for sigma in std]

        print(f'Fold: {i + 1}')
        print(f'mean: {mean}')
        print(f'std: {std}')
    # trim_head_data()
    pass
