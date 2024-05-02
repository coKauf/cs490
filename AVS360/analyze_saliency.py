
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, wasserstein_distance, entropy
import os
from PIL import Image
import csv
import collections
import pandas as pd
import scipy.stats as stats


indoor_focused_videos = {'4003', '2008', '2006', '0029', '5018', '0030', '0025', '0018', '4008', '1006'}
outdoor_exploratory_videos = {'5002', '5019', '5034', '5031', '5006', '2013', '1008', '5010', '5004', '5003'}

def compute_saliency_metrics(ground_truth, predicted):
    # AUC
    ground_truth_flat = ground_truth.flatten()
    predicted_flat = predicted.flatten()

    threshold = np.max(ground_truth_flat) * 0.5
    auc_gt = (ground_truth_flat >= threshold).astype(int)
    auc = roc_auc_score(auc_gt, predicted_flat)

    # Pearson's CC
    cc = pearsonr(predicted_flat, ground_truth_flat)[0]

    # NSS
    nss = (predicted_flat - np.mean(predicted_flat)) / np.std(predicted_flat)
    if np.sum(auc_gt) == 0:
        nss_score = np.nan  # No positive samples to compute NSS
    else:
        nss_score = nss[auc_gt > 0].mean()

    # EMD - Here we assume both arrays are histograms
    emd = wasserstein_distance(predicted_flat, ground_truth_flat)

    # KL Divergence
    epsilon = 1e-8
    predicted_safe = predicted_flat + epsilon
    ground_truth_safe = ground_truth_flat + epsilon
    # Normalize to make them sum to 1
    predicted_safe /= predicted_safe.sum()
    ground_truth_safe /= ground_truth_safe.sum()
    kl_div = entropy(predicted_safe, ground_truth_safe)

    return auc, cc, nss_score, emd, kl_div


def load_and_preprocess_image(path):
    img = Image.open(path).resize((12, 6), Image.Resampling.LANCZOS).convert('L')
    return np.array(img, dtype=np.float32) / 255.0

    
def analyze_saliency_maps():
    metrics = ['auc', 'cc', 'nss', 'emd', 'kl_div']
    headers = ['video', 'frame', 'sound', 'auc', 'cc', 'nss', 'emd', 'kl_div']

    gt_sound_maps_dir = '../datasets/dsav360/saliency_maps'
    sound_maps_dir = 'sound/predicted_saliency/dsav'
    no_sound_maps_dir = 'no_sound/predicted_saliency/dsav'

    # Prepare to write to CSV
    with open('frame_level_saliency_metrics.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write the header

        for video in os.listdir(gt_sound_maps_dir):
            gt_video_dir = os.path.join(gt_sound_maps_dir, video)
            sound_video_dir = os.path.join(sound_maps_dir, video)
            no_sound_video_dir = os.path.join(no_sound_maps_dir, video)

            # Assume frame ordering is consistent and the lists are of equal length
            gt_maps = sorted([os.path.join(gt_video_dir, f) for f in os.listdir(gt_video_dir) if f.endswith('.png')])
            if len(gt_maps) > 120:
                gt_maps = gt_maps[::15]
            
            # for idx, gt_path in enumerate(gt_maps):
            #     ground_truth = load_and_preprocess_image(gt_path)
            #     print(f"Stats for {video}, Frame {idx}: Min={np.min(ground_truth)}, Max={np.max(ground_truth)}, Mean={np.mean(ground_truth)}")
            # exit()
            sound_maps = sorted([os.path.join(sound_video_dir, f) for f in os.listdir(sound_video_dir) if f.endswith('.png')])
            if len(sound_maps) > 120:
                sound_maps = sound_maps[::15]

            no_sound_maps = sorted([os.path.join(no_sound_video_dir, f) for f in os.listdir(no_sound_video_dir) if f.endswith('.png')])
            if len(no_sound_maps) > 120:
                no_sound_maps = no_sound_maps[::15]

            assert len(gt_maps) == len(sound_maps)
            assert len(gt_maps) == len(no_sound_maps)

            # Iterate over each frame
            for idx, (gt_path, sound_path, no_sound_path) in enumerate(zip(gt_maps, sound_maps, no_sound_maps)):
                # For sound
                ground_truth = load_and_preprocess_image(gt_path)
                predicted = load_and_preprocess_image(sound_path)
                metrics_values = compute_saliency_metrics(ground_truth, predicted)
                writer.writerow([video, idx, True] + list(metrics_values))

                # For no sound
                predicted = load_and_preprocess_image(no_sound_path)
                metrics_values = compute_saliency_metrics(ground_truth, predicted)
                writer.writerow([video, idx, False] + list(metrics_values))


def compute_statistical_significance():
    video_types = pd.read_csv('../datasets/dsav360/video_types.csv')
    video_types['type'] = video_types.apply(
        lambda row: 'indoor_focused' if (row['indoor'] == True and row['focus'] == True)
        else ('outdoor_exploratory' if row['indoor'] == False and row['focus'] == False
        else 'other'), axis=1)
    
    video_types = video_types[['Name', 'type']]
    video_types = video_types.rename(columns={'Name': 'video'})

    df = pd.read_csv('frame_level_saliency_metrics.csv')

    merged_df = pd.merge(df, video_types, left_on='video', right_on='video', how='left')

    results = {}

    for video_type in merged_df['type'].unique():
        results[video_type] = {}
        video_data = merged_df[merged_df['type'] == video_type]
        sound_data = video_data[video_data['sound'] == True]
        no_sound_data = video_data[video_data['sound'] == False]

        for metric in ['auc', 'cc', 'nss']:#, 'emd']:
            sound_metric = sound_data[metric]
            no_sound_metric = no_sound_data[metric]

            # Check if data is sufficient for Shapiro-Wilk test
            if len(sound_metric) < 3 or len(no_sound_metric) < 3:
                print(f"Insufficient data for {metric} in video {video_type}")
                continue

            # Check normality
            if stats.shapiro(sound_metric)[1] > 0.05 and stats.shapiro(no_sound_metric)[1] > 0.05:
                # If both groups are normally distributed
                stat, p = stats.ttest_rel(sound_metric, no_sound_metric)
                print('normally distributed')
            else:
                stat, p = stats.wilcoxon(sound_metric, no_sound_metric)

            results[video_type][metric] = p

    # Adjust for multiple comparisons using Bonferroni correction
    num_tests = len(results) * 3 # Assuming 3 metrics evaluated per video type
    alpha = 0.05
    bonferroni_alpha = alpha / num_tests

    # Determine significance
    significant_results = {}
    for video_type, metrics in results.items():
        significant_results[video_type] = {metric: p < bonferroni_alpha for metric, p in metrics.items()}

    print("Significant Results (Bonferroni corrected):")
    for video_type, metrics in significant_results.items():
        print(f"Video {video_type}: {metrics}")

def analyze_videos():
    data_types = {
        'video': str,
        'frame': int,
        'sound': bool,
        'auc': float,
        'cc': float,
        'nss': float,
        'emd': float,
        'kl_div': float
    }

    df = pd.read_csv('frame_level_saliency_metrics.csv', dtype = data_types)
    df['indoor_focused'] = df['video'].isin(indoor_focused_videos)

    merged_df = df.groupby(['indoor_focused', 'sound']).mean()
    merged_df.drop(columns=['frame', 'emd', 'kl_div'], inplace=True)
    # for row in merged_df.iterrows():
    #     print(row)
    # print(merged_df)

    merged_df.columns = ['_'.join(col) for col in merged_df.columns]

    # Reset index to turn multi-level index into columns
    video_sound_type_stats = merged_df.reset_index()

    # Now you can print or save this DataFrame, and it will show 'indoor_focused' and 'sound' for every row
    print(video_sound_type_stats)


    # # Calculate statistics for each group
    # video_sound_type_stats = merged_df.agg({
    #     'auc': ['mean', 'median', 'std', 'min', 'max'],
    #     'cc': ['mean', 'median', 'std', 'min', 'max'],
    #     'nss': ['mean', 'median', 'std', 'min', 'max'],
    #     # 'emd': ['mean', 'median', 'std', 'min', 'max'],
    #     'kl_div': ['mean', 'median', 'std', 'min', 'max']
    # })

    # video_sound_type_stats.to_csv('sound_type_level_statistics.csv')

    # print(video_sound_type_stats)

    # means_only = video_sound_type_stats.loc[:, (slice(None), 'mean')]
    # print(means_only)

    # return video_sound_type_stats

def analyze_og_avs360():
    df = pd.read_csv('avs360_slim_scores.csv')
    print(df.columns)

if __name__=='__main__':
    # analyze_saliency_maps()
    # analyze_videos()
    # df = pd.read_csv('saliency_metrics.csv')
    # sound_df = df[df['sound'] == True]
    # no_sound_df = df[df['sound'] == False]
    
    # print(sound_df.head())
    # print(no_sound_df.head())

    compute_statistical_significance()
    # analyze_og_avs360()
    pass
