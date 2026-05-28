from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import scipy
import pandas as pd
import scipy.stats as sc
from matplotlib.lines import lineStyles

output_image_folder = './analysis_plots_xray'

open_image_automated = sorted(glob.glob('./ALL_datasets/automated_all_tresholds/P280*.npy'))
open_image_semi_automated = sorted(glob.glob('./ALL_datasets/semi_automated_ALL/semi_automated/P280*.npy'))

filename_list = []
dice_list = []
true_positive_list = []
true_negative_list = []
false_positive_list = []
false_negative_list = []
precision_list = []
recall_list = []
iou_list = []
specificity_list = []
semi_auto_area_list =[]
auto_area_list = []

def dice_coefficient(mask_auto, mask_semi_automated):
    #mask_auto = mask_auto.astype(bool) #boolean array
    #mask_semi_automated = mask_semi_automated.astype(bool)

    intersection = np.sum(mask_auto * mask_semi_automated) #pixels where both masks detect a focus
    if np.sum(mask_auto) + np.sum(mask_semi_automated) == 0:
        return 1
    else:
        return (2.0 * intersection) / (mask_auto.sum() + mask_semi_automated.sum()) #dice equation

for filename in open_image_automated:
    automated_mask = np.load(filename)
    image_filename = os.path.basename(filename)
    if './ALL_datasets/semi_automated_ALL/semi_automated/' + image_filename not in open_image_semi_automated:
        continue
    semi_automated_mask = np.load('./ALL_datasets/semi_automated_ALL/semi_automated/' + image_filename)
    filename_list.append(image_filename)
    dice = dice_coefficient(mask_auto= automated_mask, mask_semi_automated= semi_automated_mask)
    dice_list.append(dice)
    true_positive = np.sum((automated_mask == 1) & (semi_automated_mask == 1))
    true_negative = np.sum((automated_mask == 0) & (semi_automated_mask == 0))
    false_positive = np.sum((automated_mask == 1) & (semi_automated_mask == 0))
    false_negative = np.sum((automated_mask == 0) & (semi_automated_mask == 1))
    true_positive_list.append(true_positive)
    true_negative_list.append(true_negative)
    false_negative_list.append(false_negative)
    false_positive_list.append(false_positive)
    if np.sum(automated_mask) + np.sum(semi_automated_mask) == 0:
        precision = 1
        recall = 1
        iou = 1
        specificity = 1
    elif true_positive + false_positive == 0 or true_positive + false_negative == 0:
        precision = 0
        recall = 0
        iou = 0
        specificity = true_negative / (true_negative + false_positive)
    else:
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        union = true_positive + false_positive + false_negative
        iou = true_positive / union
        specificity = true_negative / (true_negative + false_positive)
    precision_list.append(precision)
    recall_list.append(recall)
    iou_list.append(iou)
    specificity_list.append(specificity)
    semi_auto_area_list.append(np.sum(semi_automated_mask))
    auto_area_list.append(np.sum(automated_mask))

pixel_level_df = pd.DataFrame()
pixel_level_df['Filename'] = filename_list
pixel_level_df['TP'] = true_positive_list
pixel_level_df['FP'] = false_positive_list
pixel_level_df['FN'] = false_negative_list
pixel_level_df['TN'] = true_negative_list
pixel_level_df['Precision'] = precision_list
pixel_level_df['Recall'] = recall_list
pixel_level_df['Dice'] = dice_list
pixel_level_df['IoU'] = iou_list
pixel_level_df['Specificity'] = specificity_list

pixel_level_df.to_excel(output_image_folder + '/pixel_level_analysis_person1.xlsx')

average_pixel_level_df = pd.DataFrame()
average_pixel_level_df['Category'] = ['Precision', 'Recall', 'Dice', 'IoU', 'Specificity']
average_pixel_level_df['Mean'] = [np.mean(precision_list), np.mean(recall_list), np.mean(dice_list), np.mean(iou_list),
                                  np.mean(specificity_list)]
average_pixel_level_df['Median'] = [np.median(precision_list), np.median(recall_list), np.median(dice_list), np.median(iou_list),
                                  np.median(specificity_list)]
average_pixel_level_df['Variance'] = [np.var(precision_list), np.var(recall_list), np.var(dice_list), np.var(iou_list),
                                  np.var(specificity_list)]
average_pixel_level_df['Stdev'] = [np.std(precision_list), np.std(recall_list), np.std(dice_list), np.std(iou_list),
                                  np.std(specificity_list)]
average_pixel_level_df.to_excel( output_image_folder + '/pixel_level_averages_person1.xlsx')

# dice stat test
dice_array = np.array(dice_list)
_, pvalue_shapiro = sc.shapiro(dice_array)

if pvalue_shapiro <0.05:
    _, pvalue_dice = sc.wilcoxon(dice_array - 1)

else:
    _, pvalue_dice = sc.ttest_1samp(dice_array, 1)
    

# recall stat test
recall_array = np.array(recall_list)
_, pvalue_shapiro = sc.shapiro(recall_array)

if pvalue_shapiro < 0.05:
    _, pvalue_recall = sc.wilcoxon(recall_array - 1)

else:
    _, pvalue_recall = sc.ttest_1samp(recall_array, 1)

# precision stat test
precision_array = np.array(precision_list)
_, pvalue_shapiro = sc.shapiro(precision_array)

if pvalue_shapiro < 0.05:
    _, pvalue_precision = sc.wilcoxon(precision_array - 1)

else:
    _, pvalue_precision = sc.ttest_1samp(precision_array, 1)

# iou stat test
iou_array = np.array(iou_list)
_, pvalue_shapiro = sc.shapiro(iou_array)

if pvalue_shapiro < 0.05:
    _, pvalue_iou = sc.wilcoxon(iou_array - 1)

else:
    _, pvalue_iou = sc.ttest_1samp(iou_array, 1)

#segmentation area comparision
diff_area = np.abs(np.array(semi_auto_area_list) - np.array(auto_area_list))
_, pvalue_shapiro = sc.shapiro(diff_area)

if pvalue_shapiro < 0.05:
    _, pvalue_area = sc.wilcoxon(diff_area)

else:
    _, pvalue_area = sc.ttest_rel(np.array(semi_auto_area_list), np.array(auto_area_list))

#correlation
pearson_r, pvalue_pearson = sc.pearsonr(np.array(semi_auto_area_list), np.array(auto_area_list))
spearman_r, pvalue_spearman = sc.spearmanr(np.array(semi_auto_area_list), np.array(auto_area_list))

#plot for dice score (histogram)
plt.figure()
plt.hist(dice_list, bins=10)
plt.xlabel('Dice score')
plt.ylabel('Frequency')
plt.title('Pixel Level Dice Score Distribution')

plt.savefig(output_image_folder + '/pixel_level_dice_histogram.png', dpi = 300)
plt.close()

#dice, precison, recall, iou boxplot
plt.figure()
plt.boxplot([precision_list, recall_list, dice_list, iou_list], labels=['Precision', 'Recall', 'Dice','IoU'], whis=1.5)
plt.title('Pixel Level Metrics')

plt.savefig(output_image_folder + '/pixel_level_metrics_boxplot.png', dpi = 300)
plt.close()

#diff area plot
plt.figure()
plt.hist(diff_area, bins=15)

plt.xlabel('Absolute Segmentation Area Difference (pixels)')
plt.ylabel('Frequency')
plt.title('Segmentation Area Difference Distribution')

plt.savefig(output_image_folder + '/area_difference_histogram.png', dpi = 300)
plt.close()

#correlation
plt.figure()

plt.scatter(np.array(semi_auto_area_list), np.array(auto_area_list))
gradient, y_intercept = np.polyfit(np.array(semi_auto_area_list), np.array(auto_area_list), deg = 1)
plt.plot(np.array(semi_auto_area_list), (gradient * np.array(semi_auto_area_list)) + y_intercept)
correlation_text = 'Pearson r = ' + str(np.round(pearson_r, 2)) + '\nSpearman ρ = ' + str(np.round(spearman_r, 2))
plt.text(0.8 * np.max(np.array(semi_auto_area_list)),
         0.95 * np.max((gradient * np.array(semi_auto_area_list)) + y_intercept),
         correlation_text, fontsize = 8, bbox = dict(facecolor = 'bisque', edgecolor = 'black', boxstyle = 'round', alpha = 0.5))

plt.xlabel('Semi-automated area')
plt.ylabel('Automated area')
plt.title('Segmented area correlation (automated vs semi-automated)')

plt.savefig(output_image_folder + '/area_correlation_plot.png', dpi = 300)
plt.close()

#output of p values
pval_df = pd.DataFrame()
pval_df['p value_category'] = ['Precision', 'Recall', 'Dice', 'IoU', 'Segmented Area']
pval_df['p value'] = [pvalue_precision, pvalue_recall, pvalue_dice, pvalue_iou, pvalue_area]
pval_df.to_excel(output_image_folder + '/pixel_level_pvalue_output_person1.xlsx')

#segmentation area bland altman plot
auto_area = []
semi_auto_area = []
for i,x in enumerate(auto_area_list):
    if x == 0 and semi_auto_area_list[i] == 0:
        continue
    else:
        auto_area.append(x)
        semi_auto_area.append(semi_auto_area_list[i])

mean_images = (np.array(auto_area) + np.array(semi_auto_area)) / 2
diff_images = (np.array(semi_auto_area) - np.array(auto_area))
bias = np.mean(diff_images)
stdev = np.std(diff_images, ddof= 1)
upper_limit = bias + (1.96*stdev)
lower_limit = bias - (1.96*stdev)


plt.title('Pixel Level Segmentation Area Bland Altman Plot')
plt.scatter(mean_images, diff_images, alpha= 0.3)
plt.axhline(bias, linestyle = '--', color = 'red', alpha=0.5)
plt.axhline(upper_limit, linestyle = '--', color = 'red', alpha=0.5)
plt.axhline(lower_limit, linestyle = '--', color = 'red', alpha=0.5)
plt.annotate('Bias=' + str(np.round(bias, 2)),  (200, bias + 1), size= 8)
plt.annotate('Upper LoA=' + str(np.round(upper_limit, 2)),  (200, upper_limit + 1), size= 8)
plt.annotate('Lower LoA=' + str(np.round(lower_limit, 2)),  (200, lower_limit + 1), size= 8)
plt.xlabel('Tool Mean')
plt.ylabel('Tool Difference (semi auto - auto)')
plt.savefig(output_image_folder + '/Segmentation_area_Bland_Altman.png', dpi=300)

