from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import scipy
import pandas as pd
import scipy.stats as sc

person1_data = pd.read_excel('./pixel_level_analysis_person1.xlsx')
person2_data = pd.read_excel('./pixel_level_analysis_person2.xlsx')

person1_data = person1_data[person1_data['Filename'].isin(person2_data['Filename'])]
person1_data.sort_values('Filename', inplace= True)
person2_data.sort_values('Filename', inplace= True)

# dice stat test
dice_array_1 = np.array(list(person1_data['Dice']))
dice_array_2 = np.array(list(person2_data['Dice']))
limit_data = dice_array_2.shape[0]

_, pvalue_shapiro_1 = sc.shapiro(dice_array_1[:limit_data])
_, pvalue_shapiro_2 = sc.shapiro(dice_array_2)

if pvalue_shapiro_1 <0.05 or pvalue_shapiro_2 <0.05:
    _, pvalue_dice = sc.wilcoxon(dice_array_1[:limit_data], dice_array_2)

else:
    _, pvalue_dice = sc.ttest_1samp(dice_array_1[:limit_data], dice_array_2)


# recall stat test
# recall_array_1 = np.array(list(person1_data['Recall']))
# recall_array_2 = np.array(list(person2_data['Recall']))
# _, pvalue_shapiro_1 = sc.shapiro(recall_array_1[:limit_data])
# _, pvalue_shapiro_2 = sc.shapiro(recall_array_2)
#
# if pvalue_shapiro_1 <0.05 or pvalue_shapiro_2 <0.05:
#     _, pvalue_recall = sc.wilcoxon(recall_array_1[:limit_data], recall_array_2)
#
# else:
#     _, pvalue_recall = sc.ttest_1samp(recall_array_1[:limit_data], recall_array_2)

# precision stat test
# precision_array_1 = np.array(list(person1_data['Precision']))
# precision_array_2 = np.array(list(person2_data['Precision']))
# _, pvalue_shapiro_1 = sc.shapiro(precision_array_1[:limit_data])
# _, pvalue_shapiro_2 = sc.shapiro(precision_array_2)
#
# if pvalue_shapiro_1 <0.05 or pvalue_shapiro_2 <0.05:
#     _, pvalue_precision = sc.wilcoxon(precision_array_1[:limit_data], precision_array_2)
#
# else:
#     _, pvalue_precision = sc.ttest_1samp(precision_array_1[:limit_data], precision_array_2)

# iou stat test
# iou_array_1 = np.array(list(person1_data['IoU']))
# iou_array_2 = np.array(list(person2_data['IoU']))
# _, pvalue_shapiro_1 = sc.shapiro(iou_array_1[:limit_data])
# _, pvalue_shapiro_2 = sc.shapiro(iou_array_2)
#
# if pvalue_shapiro_1 <0.05 or pvalue_shapiro_2 <0.05:
#     _, pvalue_iou = sc.wilcoxon(iou_array_1[:limit_data], iou_array_2)
#
# else:
#     _, pvalue_iou = sc.ttest_1samp(iou_array_1[:limit_data], iou_array_2)


print(pvalue_dice)

#correlation
person1_semi_auto_area_list = np.array(list(person1_data['TP'])) + np.array(list(person1_data['FN']))
person2_semi_auto_area_list = np.array(list(person2_data['TP'])) + np.array(list(person2_data['FN']))

#extract most different filenames
delta_semi_auto_area_list = np.abs(np.array(person1_semi_auto_area_list) - np.array(person2_semi_auto_area_list))
delta_semi_auto_area_index = np.argsort(-delta_semi_auto_area_list)
filenames_ordered = np.array(list(person2_data['Filename']))[delta_semi_auto_area_index]
output_max_delta_df = pd.DataFrame()
output_max_delta_df['Filename'] = filenames_ordered
output_max_delta_df['Delta'] = delta_semi_auto_area_list[delta_semi_auto_area_index]
output_max_delta_df.to_excel('./delta_segmented_area_semi_auto_filenames.xlsx')

#calculate correlations and plot
pearson_r, pvalue_pearson = sc.pearsonr(np.array(person1_semi_auto_area_list), np.array(person2_semi_auto_area_list))
spearman_r, pvalue_spearman = sc.spearmanr(np.array(person1_semi_auto_area_list), np.array(person2_semi_auto_area_list))

plt.figure()

plt.scatter(np.array(person1_semi_auto_area_list), np.array(person2_semi_auto_area_list))
gradient, y_intercept = np.polyfit(np.array(person1_semi_auto_area_list), np.array(person2_semi_auto_area_list), deg = 1)
plt.plot(np.array(person1_semi_auto_area_list), (gradient * np.array(person1_semi_auto_area_list)) + y_intercept)
correlation_text = 'Pearson r = ' + str(np.round(pearson_r, 2)) + '\nSpearman ρ = ' + str(np.round(spearman_r, 2))
plt.text(0.8 * np.max(np.array(person1_semi_auto_area_list)),
         0.95 * np.max((gradient * np.array(person1_semi_auto_area_list)) + y_intercept),
         correlation_text, fontsize = 8, bbox = dict(facecolor = 'bisque', edgecolor = 'black', boxstyle = 'round', alpha = 0.5))

plt.xlabel('Semi-automated area P1')
plt.ylabel('Semi-automated area P2')
plt.title('Segmented area correlation (semi_automated_P1 vs semi-automated_P2)')

plt.savefig('./semi_automated_comparison_area_correlation_plot.png', dpi = 300)
plt.close()