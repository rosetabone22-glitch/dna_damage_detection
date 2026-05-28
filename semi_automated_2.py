import numpy as np
from PIL import Image, ImageFilter
import glob
from scipy import ndimage
from Utilities import *
# from find_maxima import grayscale_image
import copy
import configparser
from matplotlib.backend_bases import MouseButton
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os


class AutomatedTool:
    def __init__(self):

        self.config = configparser.ConfigParser()
        self.config.read('./config.ini')
        self.input_folder = self.config.get('Paths', 'image_folder')

        self.full_width_fraction_max = self.config.getfloat('Parameters', 'full_width_fraction_max')
        self.stdev_x = self.config.getfloat('Parameters', 'stdev_x')
        self.gaussian_filter_sigma = self.config.getfloat('Parameters', 'gaussian_filter_sigma')
        self.run_type = self.config.get('Parameters', 'run_type')

        self.proximity_pixel_number = self.config.getfloat('Focus_check', 'proximity_pixel_number')
        self.direction_width = self.config.getfloat('Focus_check', 'direction_width')
        self.elongation_check = self.config.getfloat('Focus_check', 'elongation_check')
        self.area_check = self.config.getfloat('Focus_check', 'area_check')
        self.focus_max_check = self.config.getfloat('Focus_check', 'focus_max_check')

        self.file_name = None
        self.y_limit = None
        self.x_limit = None
        self.global_maximum = None
        self.global_treshold = None
        self.image_original = None
        self.image_grayscale = None
        self.focus_check = []
        self.x = None
        self.y = None

        # filename list obtained from the folder of images
        self.filenames = sorted(glob.glob(self.input_folder))
        self.foci_pixels = list()

        for filename in self.filenames:
            if self.run_type == 'semi_automated':
                plot_folder = self.config.get('Paths', 'plot_folder_semi_automated')
                if os.path.exists(plot_folder + os.path.basename(filename)[:-4] + '_mask.npy'):
                    continue
            self._image_open(filename)
            self._image_adjustment()

            self.filename_foci = list()

            if self.run_type == 'automated':
                self.plot_folder = self.config.get('Paths', 'plot_folder_automated')
                while True:
                    y, x = get_image_argmax(self.image_grayscale)  # y, x value of brightest pixel
                    self._focus_search(pix_y = y, pix_x = x)
                    if np.amax(self.image_grayscale) < self.global_treshold:  # stop looking when next brightest pixel is less than the treshold (noise)
                        break
                self.foci_pixels.append(self.filename_foci)  # save what there's in each filename foci before for loop ends
                self._focus_check()

            elif self.run_type == 'semi_automated':
                self.plot_folder = self.config.get('Paths', 'plot_folder_semi_automated')
                self.check_image_finished = False
                self._run_semi_automated()

            self._focus_plot()

            self._focus_outputs()

    def _image_open(self, filename):
        self.image_opened = Image.open(filename)  # function to open image
        file_name = os.path.basename(filename)
        self.file_name = file_name.split('.')[0]

    def _image_adjustment(self):
        self.image_grayscale = np.array(self.image_opened)  # image is converted into a numpy array
        self.image_grayscale = ndimage.gaussian_filter(input=self.image_grayscale, sigma=self.gaussian_filter_sigma)
        image_dimensions = self.image_grayscale.shape  # get the image dimensions
        self.y_limit = image_dimensions[0] - 1  # y limit of the current image (number of pixels in y axis)
        self.x_limit = image_dimensions[1] - 1  # x limit of the current image (number of pixels in x axis)
        self.global_maximum = np.amax(self.image_grayscale)  # to get the max value of the brightest pixel
        global_mean = np.mean(self.image_grayscale)
        global_stdev = np.std(self.image_grayscale)
        self.global_treshold = global_mean + (self.stdev_x * global_stdev)  # global treshold for background noise (5stdev away from the mean)
        self.image_original = copy.deepcopy(self.image_grayscale)  # to save the original image
        self.image_grayscale[self.image_grayscale < self.global_treshold] = 0

    def _focus_search(self, pix_y, pix_x):

        y = pix_y
        x = pix_x
        self.focus_specific_pixels = list()
        self.focus_specific_pixels.append([y, x])  # brightest pixels saved in list
        pixel_value = self.image_grayscale[y, x]  # pixel value
        threshold = pixel_value * self.full_width_fraction_max  # half the max brightness of the focus
        checked_pixels = list()
        j = 0  # first pixel
        self.image_grayscale[y, x] = 0  # changes original image

        while True:
            j = j + 1  # once while loop is done, pixel is updated

            if 0 <= y + 1 <= self.y_limit and 0 <= x <= self.x_limit:  # within the border
                pixel_1 = self.image_grayscale[y + 1, x]
                if pixel_1 >= threshold and pixel_1 >= self.global_treshold:
                    self.focus_specific_pixels.append([y + 1, x])  # position saved
                    self.image_grayscale[y + 1, x] = 0  # to not find the same pixel again once entering the loop
                else:
                    self.image_grayscale[y + 1, x] = 0  # avoid detection in subsequent highest brightness pixel searches

            if 0 <= y + 1 <= self.y_limit and 0 <= x + 1 <= self.x_limit:
                pixel_2 = self.image_grayscale[y + 1, x + 1]
                if pixel_2 >= threshold and pixel_2 >= self.global_treshold:
                    self.focus_specific_pixels.append([y + 1, x + 1])
                    self.image_grayscale[y + 1, x + 1] = 0
                else:
                    self.image_grayscale[y + 1, x + 1] = 0

            if 0 <= y <= self.y_limit and 0 <= x + 1 <= self.x_limit:
                pixel_3 = self.image_grayscale[y, x + 1]
                if pixel_3 >= threshold and pixel_3 >= self.global_treshold:
                    self.focus_specific_pixels.append([y, x + 1])
                    self.image_grayscale[y, x + 1] = 0
                else:
                    self.image_grayscale[y, x + 1] = 0

            if 0 <= y - 1 <= self.y_limit and 0 <= x + 1 <= self.x_limit:
                pixel_4 = self.image_grayscale[y - 1, x + 1]
                if pixel_4 >= threshold and pixel_4 >= self.global_treshold:
                    self.focus_specific_pixels.append([y - 1, x + 1])
                    self.image_grayscale[y - 1, x + 1] = 0
                else:
                    self.image_grayscale[y - 1, x + 1] = 0

            if 0 <= y - 1 <= self.y_limit and 0 <= x <= self.x_limit:
                pixel_5 = self.image_grayscale[y - 1, x]
                if pixel_5 >= threshold and pixel_5 >= self.global_treshold:
                    self.focus_specific_pixels.append([y - 1, x])
                    self.image_grayscale[y - 1, x] = 0
                else:
                    self.image_grayscale[y - 1, x] = 0

            if 0 <= y - 1 <= self.y_limit and 0 <= x - 1 <= self.x_limit:
                pixel_6 = self.image_grayscale[y - 1, x - 1]
                if pixel_6 >= threshold and pixel_6 >= self.global_treshold:
                    self.focus_specific_pixels.append([y - 1, x - 1])
                    self.image_grayscale[y - 1, x - 1] = 0
                else:
                    self.image_grayscale[y - 1, x - 1] = 0

            if 0 <= y <= self.y_limit and 0 <= x - 1 <= self.x_limit:
                pixel_7 = self.image_grayscale[y, x - 1]
                if pixel_7 >= threshold and pixel_7 >= self.global_treshold:
                    self.focus_specific_pixels.append([y, x - 1])
                    self.image_grayscale[y, x - 1] = 0
                else:
                    self.image_grayscale[y, x - 1] = 0

            if 0 <= y + 1 <= self.y_limit and 0 <= x - 1 <= self.x_limit:
                pixel_8 = self.image_grayscale[y + 1, x - 1]
                if pixel_8 >= threshold and pixel_8 >= self.global_treshold:
                    self.focus_specific_pixels.append([y + 1, x - 1])
                    self.image_grayscale[y + 1, x - 1] = 0
                else:
                    self.image_grayscale[y + 1, x - 1] = 0

            checked_pixels.append([y, x])  # y,x pixels for which we have looked around the central pixel
            if len(checked_pixels) == len(self.focus_specific_pixels):  # once every pixel is checked that passed the treshold
                self.filename_foci.append(self.focus_specific_pixels)  # saving all foci for an image
                break
            else:
                y, x = self.focus_specific_pixels[j]  # y,x for the next pixel

    def _onclick(self, event):
        # Single-click foci selection
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            self.check_image_finished = True
            return
        self.user_click = True
        self.x, self.y = int(event.xdata), int(event.ydata)
        self._focus_search(pix_y=self.y, pix_x=self.x)
        plt.close()

    def _on_press(self, event):
        if event.inaxes is None:
            return
        self.dragging = True
        self.drag_button = event.button

    def _on_release(self, event):
        self.dragging = False

    def _on_motion(self, event):
        if not getattr(self, 'dragging', False) or event.inaxes is None:
            return
        x, y = int(event.xdata), int(event.ydata)
        if self.drag_button == MouseButton.LEFT:
            for grp in self.filename_foci:
                if [y, x] in grp:
                    grp.remove([y, x])
                    self.adjusted = True
        elif self.drag_button == MouseButton.RIGHT:
            if not any([y, x] in g for g in self.filename_foci):
                self.filename_foci.append([[y, x]])
                self.adjusted = True
        ax = event.inaxes
        ax.clear()
        ax.imshow(self.image_original, cmap='gray', vmin=400, vmax=4000)
        for grp in self.filename_foci:
            ys, xs = zip(*grp) if grp else ([], [])
            ax.scatter(xs, ys, c='red', s=0.6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.figure.canvas.draw_idle()

    def _focus_check(self):
        self.focus_check = []

        for i,focus in enumerate(self.filename_foci): #give each foci a number
            focus = np.array(focus)
            y_values = focus[:,0] #first column of y values
            x_values = focus[:,1] #second column of x values
            x_max = np.amax(x_values) #edge in the x direction
            y_max = np.amax(y_values) #edge in the y direction
            x_min = np.amin(x_values)
            y_min = np.amin(y_values)
            focus_max = np.amax(self.image_original[y_min:y_max + 1, x_min:x_max + 1]) #brightest point within the boundaries
            y_diff = np.abs(y_max - y_min) + 1 #to get an idea of the shape of the foci in the y direction
            x_diff = np.abs(x_max - x_min) + 1 #" " in the x direction
            area = y_diff * x_diff
            no_of_pixels = len(focus)

            #check 1
            focuscheck = True
            if x_diff  == self.direction_width or y_diff == self.direction_width: #if x,y direction is less than 2 pixels (artefacts like straight lines or dots)
                focuscheck = False
            print(focuscheck)
            #check 2
            if abs(y_diff - x_diff) > self.elongation_check:  # to check that the diff in x and y is not greater than 4 (to check if foci is elongated)
                focuscheck = False
            print(focuscheck)
            #check 3
            if no_of_pixels <= area * self.area_check:
                focuscheck = False
            print(focuscheck)
            #check 4
            for focus_ in self.filename_foci:  # to check each other focus for proximity
                focus_ = np.array(focus_)
                y_values_ = focus_[:, 0]
                x_values_ = focus_[:, 1]
                x_max_ = np.amax(x_values_)
                y_max_ = np.amax(y_values_)
                x_min_ = np.amin(x_values_)
                y_min_ = np.amin(y_values_)
                if np.abs(x_max_ - x_max) < self.proximity_pixel_number or np.abs(x_max_ - x_min) < self.proximity_pixel_number:  # if <3 pixels away from each other  in x direction
                    if np.abs(y_max_ - y_max) < self.proximity_pixel_number or np.abs(y_max_ - y_min) < self.proximity_pixel_number:  # in y direction
                        focus_max_ = np.amax(self.image_original[y_min_:y_max_ + 1, x_min_:x_max_ + 1])  # check brightness to ensure that they are true foci
                        if focus_max < focus_max_ * self.focus_max_check:  # if close to each other and less than half of the brightness than the other one
                            focuscheck = False
                    if np.abs(y_min_ - y_max) < self.proximity_pixel_number or np.abs(y_min_ - y_min) < self.proximity_pixel_number:  # check other directions
                        focus_max_ = np.amax(self.image_original[y_min_:y_max_ + 1, x_min_:x_max_ + 1])
                        if focus_max < focus_max_ * self.focus_max_check:
                            focuscheck = False
                if np.abs(x_min_ - x_max) < self.proximity_pixel_number or np.abs(x_min_ - x_min) < self.proximity_pixel_number:
                    if np.abs(y_max_ - y_max) < self.proximity_pixel_number or np.abs(y_max_ - y_min) < self.proximity_pixel_number:
                        focus_max_ = np.amax(self.image_original[y_min_:y_max_ + 1, x_min_:x_max_ + 1])
                        if focus_max < focus_max_ * self.focus_max_check:
                            focuscheck = False
                    if np.abs(y_min_ - y_max) < self.proximity_pixel_number or np.abs(y_min_ - y_min) < self.proximity_pixel_number:
                        focus_max_ = np.amax(self.image_original[y_min_:y_max_ + 1, x_min_:x_max_ + 1])
                        if focus_max < focus_max_ * self.focus_max_check:
                            focuscheck = False
            print(focuscheck)

            if not focuscheck:
                self.focus_check.append(False)

            else:
                self.focus_check.append(True)

    def _focus_plot(self):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row and 2 columns
        fig.suptitle(self.file_name)

        # left subplot
        axes[0].imshow(self.image_original, cmap='gray')
        axes[0].set_title('Original')

        # right subplot
        axes[1].imshow(self.image_original, cmap='gray')
        for i, focus in enumerate(self.filename_foci):
            if self.run_type == 'automated':
                if self.focus_check[i]:  # if focus check is true
                    for pixel in focus:
                        axes[1].scatter(pixel[1], pixel[0], c='red', s=1)  # 1 = y value, 0 = x value
            elif self.run_type == 'semi_automated':
                for pixel in focus:
                    axes[1].scatter(pixel[1], pixel[0], c='red', s=1)  # 1 = y value, 0 = x value


        axes[1].set_title('Labeled')

        plt.tight_layout()  # spacing
        plt.savefig(self.plot_folder + self.file_name + '.png', dpi=300)
        plt.close()

    def _focus_outputs(self):
        mask = np.zeros((self.y_limit + 1, self.x_limit +1))
        for i, focus in enumerate(self.filename_foci):  # give each foci a number
            if self.run_type == 'automated':
                if self.focus_check[i]:
                    for pixel in focus:
                        mask[pixel[0], pixel[1]] = 1
            if self.run_type == 'semi_automated':
                for pixel in focus:
                    mask[pixel[0], pixel[1]] = 1

        fig, axes = plt.subplots(figsize=(10, 5))
        fig.suptitle(self.file_name)
        axes.imshow(mask, cmap='gray')
        axes.set_title('Mask')

        plt.tight_layout()  # spacing
        plt.savefig(self.plot_folder + self.file_name + '_mask.png', dpi=300)
        plt.close()

        np.save(self.plot_folder + self.file_name + '_mask.npy', mask)

    def _run_semi_automated(self):
        iter_ = 0

        while True:
            self.user_click = False

            if iter_ > 0:

                self.fig, self.axs = plt.subplots(1, 2, figsize=(16, 9))

                self.fig.suptitle('Foci Selection - ' + self.file_name)
                cid = self.fig.canvas.mpl_connect('button_press_event', self._onclick)
                img_scaled = self.image_original
                if self.config.getboolean('Parameters', 'image_scaling'):
                    img_scaled = (10 * np.log10(np.log10((self.image_original - self.image_original.min()) *
                                                         (1 / (self.image_original.max() - self.image_original.min()) * 255)))
                                  .astype('float32'))

                self.axs[0].imshow(img_scaled, cmap='gray', vmin=400, vmax=4000)
                self.axs[0].title.set_text('Original Image')
                self.axs[1].imshow(img_scaled, cmap='gray', vmin=400, vmax=4000)
                self.axs[1].title.set_text('Current Selection (0Gy Adjusted)')
                for x in range(len(self.filename_foci)):
                    for i, val in enumerate(self.filename_foci[x]):
                        self.axs[1].scatter(val[1], val[0], c='red', s=0.6)

                plt.show()
                iter_ += 1

            if iter_ == 0:

                self.fig, self.axs = plt.subplots(1, 2, figsize=(16, 9))

                self.fig.suptitle('Foci Selection - ' + self.file_name)
                cid = self.fig.canvas.mpl_connect('button_press_event', self._onclick)
                img_scaled = self.image_original
                if self.config.getboolean('Parameters', 'image_scaling'):
                    img_scaled = (10 * np.log10(np.log10((self.image_original - self.image_original.min()) *
                                                         (1 / (self.image_original.max() - self.image_original.min()) * 255)))
                                  .astype('float32'))
                self.axs[0].imshow(img_scaled, cmap='gray', vmin=400, vmax=4000)
                self.axs[0].title.set_text('Original Image')
                self.axs[1].imshow(img_scaled, cmap='gray', vmin=400, vmax=4000)
                self.axs[1].title.set_text('Current Selection (0Gy Adjusted)')


                plt.show()
                iter_ +=1

            if self.check_image_finished or not self.user_click:
                break

        # Adjustment Screen
        self.adjusted = False
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.canvas.manager.window.state('zoomed')
        ax.imshow(self.image_original, cmap='gray', vmin=400, vmax=4000)
        all_xy = [pt for grp in self.filename_foci for pt in grp]
        if all_xy:
            ys, xs = zip(*all_xy)
        else:
            ys, xs = [], []
        ax.scatter(xs, ys, c='red', s=0.6)
        fig.suptitle(f"Adjust Pixels - {self.file_name}")
        fig.canvas.mpl_connect('button_press_event', self._on_press)
        fig.canvas.mpl_connect('button_release_event', self._on_release)
        fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        plt.show()


