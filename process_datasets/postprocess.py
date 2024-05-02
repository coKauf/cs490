import cv2
import os

def create_video_from_images(input_folder, output_video_path, original_frame_rate=60):
    temp = [img for img in os.listdir(input_folder) if img.endswith(".png")]
    images = []
    for img in temp:
        for _ in range(8):
            images.append(img)
            
    images.sort()

    frame = cv2.imread(os.path.join(input_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), original_frame_rate, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(input_folder, image)))

    cv2.destroyAllWindows()
    video.release()

# Example usage
input_folder = '../PanoSalNet/saliency/output_saliency_maps/rgb/0018'
output_video_path = '../PanoSalNet/saliency/output_saliency_videos/rgb/0018.mp4'
# input_folder = 'dsav360/saliency_maps_8fps/0018'
# output_video_path = 'dsav360/saliency_videos/0018.mp4'
create_video_from_images(input_folder, output_video_path)