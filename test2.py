import os, sys
import Tkinter
import tkFileDialog
import PIL
from PIL import ImageTk, Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import cv2
from scipy.cluster.vq import kmeans
from skimage import data, img_as_float
#from skimage.measure import compare_ssim as ssim
from skimage.measure import structural_similarity as ssim


LETTERS = ["a","b","c","d","e","f","g","h","i","j","k","l","m",
           "n","o","p","q","r","s","t","u","v","w","x","y","z"]

class Rectangle:

    def __init__(self, x_param=0, y_param=0, w_param=0, h_param=0):
        self.x = x_param
        self.y = y_param
        self.w = w_param
        self.h = h_param

    def __str__(self):
        return "Width = "+str(self.w)+", Height = "+str(self.h)



class MainWindow:

    def __init__(self, master):

        self.video = None
        self.frame_rate = 0
        self.video_length = 0

        # The scaled image used for display. Needs to persist for display
        self.display_image = None
        self.display_ratio = 0

        self.awaiting_corners = False
        self.corners = []

        #Tkinter related fields
        self.master = master
        self.master.title("Auto Kifu Test2")
        self.window_width = root.winfo_screenwidth()
        self.window_height = root.winfo_screenheight() - 100
        self.master.geometry("%dx%d+0+0" % (self.window_width, self.window_height))
        self.master.configure(background='grey')

        self.canvas = Tkinter.Canvas(self.master)
        self.canvas.place(x=0,
                     y=0,
                     width=self.window_width,
                     height=self.window_height)
        self.canvas.bind("<Button-1>", self.mouse_clicked)

        self.menubar = Tkinter.Menu(root)
        root.config(menu=self.menubar)
        self.fileMenu = Tkinter.Menu(self.menubar)
        self.fileMenu.add_command(label="Load Image", command=self.load())
        self.menubar.add_cascade(label="File", menu=self.fileMenu)

    def mouse_clicked(self, event):
        if self.awaiting_corners:
            self.draw_x(event.x, event.y)
            self.corners += [(event.x/self.display_ratio, event.y/self.display_ratio)]
            if len(self.corners) == 4:
                self.awaiting_corners = False
                self.main()

    def main(self):
        board_positions, crop_window = self.find_grid(self.corners)
        frames = self.parse_video(crop_window)
        for x in range(len(frames)):
            frames[x] = cv2.cvtColor(frames[x], cv2.COLOR_BGR2GRAY)
            frames[x] = cv2.GaussianBlur(frames[x], (51, 51), 0)
        thresholds = self.determine_thresholds(frames[-1], board_positions)
        for x in range(len(frames)):
            cv2.imwrite('output/2/frames'+str(x)+'.png', frames[x])


        for x in range(len(frames)):
            frames[x] = self.parse_frames(frames[x], board_positions, thresholds)

        for x in range(1, len(frames)):
            print "Board: "+str(x)
            self.print_board(frames[x])

        output = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]SZ[19]"

        for i in range(1, len(frames)):
            moves = self.frame_difference(frames[i-1], frames[i])
            for move in moves:
                color = move["color"]
                x = LETTERS[move["position"][0]]
                y = LETTERS[move["position"][1]]
                output += ";"+color+"["+x+y+"]"

        output += ")"

        file = open("output.txt", "w")
        file.write(output)
        file.close()

    def find_grid(self, corners):
        top_left = corners[0]
        bottom_right = corners[2]
        board_width = bottom_right[0] - top_left[0]
        board_height = bottom_right[1] - top_left[1]
        horizontal_spacing = board_width / 18
        vertical_spacing = board_height / 18

        crop_window = Rectangle()
        crop_window.x = int(top_left[0] - horizontal_spacing)
        crop_window.y = int(top_left[1] - vertical_spacing)
        crop_window.w = int(board_width + (2 * horizontal_spacing))
        crop_window.h = int(board_height + (2 * vertical_spacing))

        board_positions = []
        for x in range(0, 19):
            board_positions += [[]]
            for y in range(0, 19):
                x_coord = int(top_left[0] + horizontal_spacing * x)
                y_coord = int(top_left[1] + vertical_spacing * y)
                x_coord -= crop_window.x
                y_coord -= crop_window.y
                board_positions[x] += [(y_coord, x_coord)]

        return board_positions, crop_window


    def print_board(self, frame):
        print "-------------------"
        for y in range(19):
            string = ""
            for x in range(19):
                string += frame[x][y]
            print string
        print "-------------------"

    def parse_video(self, crop_window):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out1 = cv2.VideoWriter('output.avi', fourcc, 30.0, (crop_window.w, crop_window.h))

        success, current_frame = self.video.read()
        current_frame = current_frame[crop_window.y:crop_window.y + crop_window.h,
                        crop_window.x:crop_window.x + crop_window.w]
        differences = []
        final_video = [current_frame]
        while (self.video.isOpened() and success):

            last_frame = current_frame
            success, current_frame = self.video.read()
            if not success: break
            current_frame = current_frame[crop_window.y:crop_window.y+crop_window.h,
                            crop_window.x:crop_window.x+crop_window.w]
            out1.write(current_frame)
            s = self.mse_total(last_frame, current_frame)

            #s = ssim(last_frame, current_frame) # Doesn't Work
            differences += [s]
            recently_still = True
            still_duration = 15
            for x in range(still_duration):
                if x<len(differences) and differences[-x]>4:
                    recently_still = False
            if recently_still:
                #out1.write(current_frame)
                s = self.mse_total(current_frame, final_video[-1])
                if s>20:
                    final_video += [current_frame]

        #plt.hist(differences, bins=400)
        plt.title("Frame Difference Historgram")
        plt.xlabel("Difference (mean squared error)")
        plt.ylabel("Number of Frames")
        #plt.show()

        time = np.arange(0, self.video_length/self.frame_rate, 1.0/self.frame_rate)
        time = time[:len(differences)]
        #plt.plot(time, differences)
        plt.xlabel('time (s)')
        plt.ylabel('Difference')
        plt.title('MSE over Time')
        plt.grid(True)
        #plt.show()

        out1.release()

        '''
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out2 = cv2.VideoWriter('output2.avi', fourcc, 30.0,
                              (self.crop_w, self.crop_h))
        for x in final_video:
            for y in range(30):
                out2.write(x)

        out2.release()
        '''

        return final_video

    def mse_total(self, imageA, imageB):
        err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])

        return err

    def mse_image(self, imageA, imageB):
        return (imageA - imageB) ** 2

    def determine_thresholds(self, image, board_positions):
        samples = []
        for x in range(0, 19):
            for y in range(0, 19):
                position = board_positions[x][y]
                samples += [float(image[position[0]][position[1]])]
        plt.hist(samples, bins=255)
        plt.title("Intersection Intensity Historgram")
        plt.xlabel("Intensity (Greyscale)")
        plt.ylabel("Number of Intersections")
        # plt.show()

        centroids, _ = kmeans(samples, 3)
        plt.axvline(x=centroids[0], color="red")
        plt.axvline(x=centroids[1], color="red")
        plt.axvline(x=centroids[2], color="red")
        plt.show()

        min = 0
        mid = 0
        max = 0
        for x in range(0, 3):
            if centroids[x] < centroids[min]:
                min = x
            if centroids[x] > centroids[max]:
                max = x
        for x in range(0, 3):
            if x != min and x != max:
                mid = x
        min = centroids[min]
        mid = centroids[mid]
        max = centroids[max]
        threshold1 = (min + mid) / 2
        threshold2 = (max + mid) / 2
        print "threshold 1 = "+str(threshold1)
        print "threshold 2 = "+str(threshold2)
        #return [threshold1, threshold2]
        return [120,185]

    def parse_frames(self, image, board_positions, thresholds):
        return_array = []
        for x in range(0, 19):
            return_array += [[]]
            for y in range(0, 19):
                position = board_positions[x][y]
                intensity = image[position[0]][position[1]]
                if intensity < thresholds[0]:
                    return_array[x] += ["B"]
                elif intensity > thresholds[1]:
                    return_array[x] += ["W"]
                else:
                    return_array[x] += ["+"]
        return return_array

    def frame_difference(self, former_frame, later_frame):
        moves = []
        for x in range(19):
            for y in range(19):
                if (later_frame[x][y] != former_frame[x][y]
                    and former_frame[x][y] == "+"):
                    moves += [{"color":later_frame[x][y],
                              "position":(x,y)}]
        return moves

    def display_grid(self, board_positions):
        for x in range(0, 19):
            for y in range(0, 19):
                self.draw_x(board_positions[x][y][1],
                            board_positions[x][y][0],
                            transform=self.display_ratio)

    def draw_x(self, x, y, radius=10, width=3, color = "red", transform = 1):
        self.canvas.create_line((x-radius)*transform,
                                (y-radius)*transform,
                                (x+radius)*transform,
                                (y+radius)*transform,
                                width=width,
                                fill=color)
        self.canvas.create_line((x-radius)*transform,
                                (y+radius)*transform,
                                (x+radius)*transform,
                                (y-radius)*transform,
                                width=width,
                                fill=color)

    def load(self):

        # Load Video
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = tkFileDialog.askopenfilename(initialdir=dir_path,
                                            title="Select file",
                                            filetypes=(
                                                ("mp4 files", "*.mp4"),
                                                ("jpeg files", "*.jpg"),
                                                ("png files", "*.png")))
        self.video = cv2.VideoCapture(path)
        self.frame_rate = self.video.get(cv2.CAP_PROP_FPS)
        self.video_length = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        success, first_frame = self.video.read()
        image_height, image_width = first_frame.shape[:2]

        # Display Image
        self.display_ratio = float(self.window_height - 200)/image_height
        resize_dimentions = (int(image_width*self.display_ratio), int(image_height*self.display_ratio))
        resized_image = cv2.resize(first_frame, resize_dimentions, interpolation=cv2.INTER_CUBIC)
        tk_image = self.convert_cv2_to_PIL(resized_image)
        self.display_image = PIL.ImageTk.PhotoImage(tk_image)
        self.canvas.create_image(0, 0, anchor ="nw", image = self.display_image)

        # cue corner collection
        self.awaiting_corners = True

    def convert_cv2_to_PIL(self, cv2image):
        cv2_im = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
        return PIL.Image.fromarray(cv2_im)

root = Tkinter.Tk()
main_window = MainWindow(root)
root.mainloop()
