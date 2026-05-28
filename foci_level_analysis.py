from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import scipy
import pandas as pd
import scipy.stats as sc


output_image_folder = './analysis_plots_xray'

open_image_automated = sorted(glob.glob('./ALL_datasets/automated_all_tresholds/P280*.npy'))
open_image_semi_automated = sorted(glob.glob('./ALL_datasets/semi_automated_ALL/semi_automated/P280*.npy'))

filename_list = []
no_of_foci_auto_list = []
no_of_foci_semi_auto_list =[]
precision_list = []
recall_list = []
dice_list = []
true_positive_list = []
false_positive_list =[]
false_negative_list = []
automated_mean_focus_size = []
semi_automated_mean_focus_size = []


for filename in open_image_automated:
    auto_size_list = []
    semi_auto_size_list = []
    automated_mask = np.load(filename)
    image_filename = os.path.basename(filename)
    if './ALL_datasets/semi_automated_ALL/semi_automated/' + image_filename not in open_image_semi_automated:
        continue
    semi_automated_mask = np.load('./ALL_datasets/semi_automated_ALL/semi_automated/' + image_filename)
    filename_list.append(image_filename)
    automated_mask_labeled, no_foci_automated = scipy.ndimage.label(automated_mask, structure= np.ones((3,3))) #3x3 array with 1s (to detect edges)
    semi_automated_mask_labeled, no_foci_semi_automated = scipy.ndimage.label(semi_automated_mask,  structure= np.ones((3,3)))

    auto_sizes = scipy.ndimage.sum(automated_mask, automated_mask_labeled, index=range(1, no_foci_automated + 1))
    semi_auto_sizes = scipy.ndimage.sum(semi_automated_mask, semi_automated_mask_labeled,
                                        index=range(1, no_foci_semi_automated + 1))
    auto_size_list.append(auto_sizes)
    semi_auto_size_list.append(semi_auto_sizes)

    no_of_foci_semi_auto_list.append(no_foci_semi_automated)
    auto_semi_focus_overlap = 0
    for label in range(1, no_foci_automated +1):
        automated_focus_mask = (automated_mask_labeled == label)
        if np.sum(semi_automated_mask_labeled[automated_focus_mask]) > 0:
            auto_semi_focus_overlap += 1

    semi_auto_focus_overlap = 0
    for label in range(1, no_foci_semi_automated + 1):
        semi_automated_focus_mask = (semi_automated_mask_labeled == label)
        if np.sum(automated_mask_labeled[semi_automated_focus_mask]) > 0:
            semi_auto_focus_overlap += 1
    print(no_foci_automated, no_foci_semi_automated, auto_semi_focus_overlap, semi_auto_focus_overlap)

    if no_foci_automated == no_foci_semi_automated == auto_semi_focus_overlap and 0< auto_semi_focus_overlap :
        automated_mean_focus_size.append(np.mean(auto_size_list))
        semi_automated_mean_focus_size.append(np.mean(semi_auto_size_list))


    true_positive = auto_semi_focus_overlap
    false_positive = no_foci_automated - true_positive
    false_negative = no_foci_semi_automated - true_positive

    if false_negative < 0:
        false_negative = 0

    true_positive_list.append(true_positive)
    false_negative_list.append(false_negative)
    false_positive_list.append(false_positive)

    no_of_foci_auto_list.append(auto_semi_focus_overlap)

    if no_foci_automated + no_foci_semi_automated == 0:
        precision = 1
        recall = 1
        dice = 1
    else:
        if no_foci_automated == 0:
            precision = 0
        else:
            precision = true_positive / (true_positive + false_positive)

        if no_foci_semi_automated == 0:
            recall = 0
        else:
            recall = true_positive / (true_positive + false_negative)

        if no_foci_automated == 0 or no_foci_semi_automated == 0:
            dice = 0
        else:
            dice = (2 * true_positive) / ((2 * true_positive) + false_positive + false_negative)
            if dice > 1:
                print(true_positive, false_positive, false_negative)
                exit()


    precision_list.append(precision)
    recall_list.append(recall)
    dice_list.append(dice)

foci_level_df = pd.DataFrame()
foci_level_df ['Filename'] = filename_list
foci_level_df['TP'] = true_positive_list
foci_level_df['FP'] = false_positive_list
foci_level_df['FN'] = false_negative_list
foci_level_df['Precision'] = precision_list
foci_level_df['Recall'] = recall_list
foci_level_df['Dice'] = dice_list

foci_level_df.to_excel(output_image_folder + '/foci_level_analysis_person1.xlsx')

average_foci_level_df = pd.DataFrame()
average_foci_level_df['Category'] = ['Precision', 'Recall', 'Dice']
average_foci_level_df['Mean'] = [np.mean(precision_list), np.mean(recall_list), np.mean(dice_list)]
average_foci_level_df['Median'] = [np.median(precision_list), np.median(recall_list), np.median(dice_list)]
average_foci_level_df['Variance'] = [np.var(precision_list), np.var(recall_list), np.var(dice_list)]
average_foci_level_df['Stdev'] = [np.std(precision_list), np.std(recall_list), np.std(dice_list)]
average_foci_level_df.to_excel(output_image_folder + '/foci_level_averages_person1.xlsx')

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

# mean focus size stat test
semi_automated_mean_focus_size_array = np.array(semi_automated_mean_focus_size)
automated_mean_focus_size_array = np.array(automated_mean_focus_size)
_, pvalue_shapiro = sc.shapiro(semi_automated_mean_focus_size_array - automated_mean_focus_size_array)

if pvalue_shapiro < 0.05:
    _, pvalue_mean_focus_size = sc.wilcoxon(semi_automated_mean_focus_size_array, automated_mean_focus_size_array)

else:
    _, pvalue_mean_focus_size = sc.ttest_rel(semi_automated_mean_focus_size_array, automated_mean_focus_size_array)

# focus number stat test
no_of_foci_semi_auto_array = np.array(no_of_foci_semi_auto_list)
no_of_foci_auto_array = np.array(no_of_foci_auto_list)
_, pvalue_shapiro = sc.shapiro(no_of_foci_semi_auto_array - no_of_foci_auto_array)

if pvalue_shapiro < 0.05:
    _, pvalue_no_of_foci = sc.wilcoxon(no_of_foci_semi_auto_array, no_of_foci_auto_array)

else:
    _, pvalue_no_of_foci = sc.ttest_rel(no_of_foci_semi_auto_array, no_of_foci_auto_array)


#correlation
pearson_r, pvalue_pearson = sc.pearsonr(np.array(no_of_foci_semi_auto_list), np.array(no_of_foci_auto_list))
spearman_r, pvalue_spearman = sc.spearmanr(np.array(no_of_foci_semi_auto_list), np.array(no_of_foci_auto_list))

#correlation mean focus size
pearson_r_focus_size, pvalue_pearson_focus_size = sc.pearsonr(np.array(semi_automated_mean_focus_size), np.array(automated_mean_focus_size))
spearman_r_focus_size, pvalue_spearman_focus_size = sc.spearmanr(np.array(semi_automated_mean_focus_size), np.array(automated_mean_focus_size))

#plot for dice score (histogram)
plt.figure()
plt.hist(dice_list, bins=10)
plt.xlabel('Dice score')
plt.ylabel('Frequency')
plt.title('Focus Level Dice Score Distribution')

plt.savefig(output_image_folder + '/focus_level_dice_histogram.png', dpi = 300)
plt.close()

#dice, precison, recall, iou boxplot
plt.figure()
plt.boxplot([precision_list, recall_list, dice_list], labels=['Precision', 'Recall', 'Dice'])
plt.title('Focus Level Metrics')

plt.savefig(output_image_folder + '/focus_level_metrics_boxplot.png', dpi = 300)
plt.close()

# correlation of no_foci
plt.figure()

plt.scatter(np.array(no_of_foci_semi_auto_list), np.array(no_of_foci_auto_list))
gradient, y_intercept = np.polyfit(np.array(no_of_foci_semi_auto_list), np.array(no_of_foci_auto_list), deg = 1)
plt.plot(np.array(no_of_foci_semi_auto_list), (gradient * np.array(no_of_foci_semi_auto_list)) + y_intercept)
correlation_text = 'Pearson r = ' + str(np.round(pearson_r, 2)) + '\nSpearman ρ = ' + str(np.round(spearman_r, 2))
plt.text(0.8 * np.max(np.array(no_of_foci_semi_auto_list)),
         0.95 * np.max((gradient * np.array(no_of_foci_semi_auto_list)) + y_intercept),
         correlation_text, fontsize = 8, bbox = dict(facecolor = 'bisque', edgecolor = 'black', boxstyle = 'round', alpha = 0.5))

plt.xlabel('Semi-automated foci no')
plt.ylabel('Automated foci no')
plt.title('Foci Number Correlation (automated vs semi-automated)')

plt.savefig(output_image_folder + '/foci_correlation_plot.png', dpi = 300)
plt.close()

# correlation of focus_size
plt.figure()

plt.scatter(np.array(semi_automated_mean_focus_size), np.array(automated_mean_focus_size))
gradient, y_intercept = np.polyfit(np.array(semi_automated_mean_focus_size), np.array(automated_mean_focus_size), deg = 1)
plt.plot(np.array(semi_automated_mean_focus_size), (gradient * np.array(semi_automated_mean_focus_size)) + y_intercept)
correlation_text = 'Pearson r = ' + str(np.round(pearson_r_focus_size, 2)) + '\nSpearman ρ = ' + str(np.round(spearman_r_focus_size, 2))
plt.text(0.8 * np.max(np.array(semi_automated_mean_focus_size)),
         0.95 * np.max((gradient * np.array(semi_automated_mean_focus_size)) + y_intercept),
         correlation_text, fontsize = 8, bbox = dict(facecolor = 'bisque', edgecolor = 'black', boxstyle = 'round', alpha = 0.5))

plt.xlabel('Mean semi-automated foci area')
plt.ylabel('Mean automated foci area (matched)')
plt.title('Mean Focus Area Correlation (automated vs semi-automated)')

plt.savefig(output_image_folder + '/foci_area_correlation_plot.png', dpi = 300)
plt.close()

print(automated_mean_focus_size, semi_automated_mean_focus_size, len(automated_mean_focus_size))


#output of p values
pval_df = pd.DataFrame()
pval_df['p value_category'] = ['Precision', 'Recall', 'Dice', 'Mean Focus Size']
pval_df['p value'] = [pvalue_precision, pvalue_recall, pvalue_dice, pvalue_mean_focus_size]
pval_df.to_excel(output_image_folder + '/foci_level_pvalue_output_person1.xlsx')

#bland altman plot
no_of_foci_auto = []
no_of_foci_semi_auto = []
for i,x in enumerate(no_of_foci_auto_list):
    if x == 0 and no_of_foci_semi_auto_list[i] == 0:
        continue
    else:
        no_of_foci_auto.append(x)
        no_of_foci_semi_auto.append(no_of_foci_semi_auto_list[i])

mean_images = (np.array(no_of_foci_auto) + np.array(no_of_foci_semi_auto)) / 2
diff_images = (np.array(no_of_foci_semi_auto) - np.array(no_of_foci_auto))
bias = np.mean(diff_images)
stdev = np.std(diff_images, ddof= 1)
upper_limit = bias + (1.96*stdev)
lower_limit = bias - (1.96*stdev)


plt.title('Focus Level Bland Altman Plot')
plt.scatter(mean_images, diff_images, alpha= 0.3)
plt.axhline(bias, linestyle = '--', color = 'red', alpha=0.5)
plt.axhline(upper_limit, linestyle = '--', color = 'red', alpha=0.5)
plt.axhline(lower_limit, linestyle = '--', color = 'red', alpha=0.5)
plt.annotate('Bias=' + str(np.round(bias, 2)),  (20, bias + 0.1), size= 8)
plt.annotate('Upper LoA=' + str(np.round(upper_limit, 2)),  (20, upper_limit + 0.1), size= 8)
plt.annotate('Lower LoA=' + str(np.round(lower_limit, 2)),  (20, lower_limit + 0.1), size= 8)
plt.xlabel('Tool Mean')
plt.ylabel('Tool Difference (semi auto - auto)')
plt.ylim((-5,12))
plt.savefig(output_image_folder +'/Foci_Bland_Altman.png', dpi=300)
plt.close()

#mean focus area bland altman plot
auto_area = []
semi_auto_area = []
for i,x in enumerate(automated_mean_focus_size):
    if x == 0 and semi_automated_mean_focus_size[i] == 0:
        continue
    else:
        auto_area.append(x)
        semi_auto_area.append(semi_automated_mean_focus_size[i])

mean_images = (np.array(auto_area) + np.array(semi_auto_area)) / 2
diff_images = (np.array(semi_auto_area) - np.array(auto_area))
bias = np.mean(diff_images)
stdev = np.std(diff_images, ddof= 1)
upper_limit = bias + (1.96*stdev)
lower_limit = bias - (1.96*stdev)


plt.title('Mean Focus Area Bland Altman Plot')
plt.scatter(mean_images, diff_images, alpha= 0.3)
plt.axhline(bias, linestyle = '--', color = 'red', alpha=0.5)
plt.axhline(upper_limit, linestyle = '--', color = 'red', alpha=0.5)
plt.axhline(lower_limit, linestyle = '--', color = 'red', alpha=0.5)
plt.annotate('Bias=' + str(np.round(bias, 2)),  (70, bias + 1), size= 8)
plt.annotate('Upper LoA=' + str(np.round(upper_limit, 2)),  (70, upper_limit + 1), size= 8)
plt.annotate('Lower LoA=' + str(np.round(lower_limit, 2)),  (70, lower_limit + 1), size= 8)
plt.xlabel('Tool Mean')
plt.ylabel('Tool Difference (semi auto - auto)')
plt.savefig(output_image_folder + '/Mean_focus_area_Bland_Altman.png', dpi=300)