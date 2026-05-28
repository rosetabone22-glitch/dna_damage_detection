import pandas as pd
import numpy as np
import scipy.stats as sc



xray_foci = pd.read_excel('./analysis_plots_xray/foci_level_analysis_person1.xlsx')
xray_pixel = pd.read_excel('./analysis_plots_xray/pixel_level_analysis_person1.xlsx')
Fe_foci = pd.read_excel('./analysis_plots_Fe/foci_level_analysis_person1.xlsx')
Fe_pixel = pd.read_excel('./analysis_plots_Fe/pixel_level_analysis_person1.xlsx')

dice_list = xray_foci['Dice']
dice_list2 = Fe_foci['Dice']
precision_list = xray_foci['Precision']
precision_list2 = Fe_foci['Precision']
recall_list = xray_foci['Recall']
recall_list2 = Fe_foci['Recall']


# dice stat test
dice_array = np.array(dice_list)
dice_array2 = np.array(dice_list2)
_, pvalue_ks = sc.kstest(dice_array, 'norm')
_, pvalue_ks2 = sc.kstest(dice_array2, 'norm')

if pvalue_ks < 0.05 or pvalue_ks2 < 0.05:
    _, pvalue_dice_foci = sc.ranksums(dice_array, dice_array2)

else:
    _, pvalue_dice_foci = sc.ttest_ind(dice_array, dice_array2)

# recall stat test
recall_array = np.array(recall_list)
recall_array2 = np.array(recall_list2)
_, pvalue_ks = sc.kstest(recall_array, 'norm')
_, pvalue_ks2 = sc.kstest(recall_array2, 'norm')

if pvalue_ks < 0.05 or pvalue_ks2 < 0.05:
    _, pvalue_recall_foci = sc.ranksums(recall_array, recall_array2)

else:
    _, pvalue_recall_foci = sc.ttest_ind(recall_array, recall_array2)

# precision stat test
precision_array = np.array(precision_list)
precision_array2 = np.array(precision_list2)
_, pvalue_ks = sc.kstest(precision_array, 'norm')
_, pvalue_ks2 = sc.kstest(precision_array2, 'norm')

if pvalue_ks < 0.05 or pvalue_ks2 < 0.05:
    _, pvalue_precision_foci = sc.ranksums(precision_array, precision_array2)

else:
    _, pvalue_precision_foci = sc.ttest_ind(precision_array, precision_array2)

#pixel level stats

dice_list = xray_pixel['Dice']
dice_list2 = Fe_pixel['Dice']
precision_list = xray_pixel['Precision']
precision_list2 = Fe_pixel['Precision']
recall_list = xray_pixel['Recall']
recall_list2 = Fe_pixel['Recall']


# dice stat test
dice_array = np.array(dice_list)
dice_array2 = np.array(dice_list2)
_, pvalue_ks = sc.kstest(dice_array, 'norm')
_, pvalue_ks2 = sc.kstest(dice_array2, 'norm')

if pvalue_ks < 0.05 or pvalue_ks2 < 0.05:
    _, pvalue_dice_pixel = sc.ranksums(dice_array, dice_array2)

else:
    _, pvalue_dice_pixel = sc.ttest_ind(dice_array, dice_array2)

# recall stat test
recall_array = np.array(recall_list)
recall_array2 = np.array(recall_list2)
_, pvalue_ks = sc.kstest(recall_array, 'norm')
_, pvalue_ks2 = sc.kstest(recall_array2, 'norm')

if pvalue_ks < 0.05 or pvalue_ks2 < 0.05:
    _, pvalue_recall_pixel = sc.ranksums(recall_array, recall_array2)

else:
    _, pvalue_recall_pixel = sc.ttest_ind(recall_array, recall_array2)

# precision stat test
precision_array = np.array(precision_list)
precision_array2 = np.array(precision_list2)
_, pvalue_ks = sc.kstest(precision_array, 'norm')
_, pvalue_ks2 = sc.kstest(precision_array2, 'norm')

if pvalue_ks < 0.05 or pvalue_ks2 < 0.05:
    _, pvalue_precision_pixel = sc.ranksums(precision_array, precision_array2)

else:
    _, pvalue_precision_pixel = sc.ttest_ind(precision_array, precision_array2)

print(pvalue_precision_foci, pvalue_recall_foci, pvalue_dice_foci)
print(pvalue_precision_pixel, pvalue_recall_pixel, pvalue_dice_pixel)