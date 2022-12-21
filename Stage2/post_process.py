from cmath import inf
import os
import json
from xmlrpc.client import boolean
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from sklearn.neighbors import KernelDensity
from matplotlib.gridspec import GridSpec
import math
import argparse
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
import cv2

Bit2FloatTable = [0.00000, 0.03125, 0.06250, 0.09375, 0.12500, 0.15625, 0.18750, 0.21875,
                  0.25000, 0.28125, 0.31250, 0.34375, 0.37500, 0.40625, 0.43750, 0.46875,
                  0.50000, 0.53125, 0.56250, 0.59375, 0.62500, 0.65625, 0.68750, 0.71875,
                  0.75000, 0.78125, 0.81250, 0.84375, 0.87500, 0.90625, 0.93750, 0.96875]

def preprocess(disfiles):
    disp = []
    with open(disfiles, 'rb') as f:
        while True:
            byte = f.read(2)
            if b'' == byte:
                break
            else:
                intbytes = int.from_bytes(byte, byteorder='little', signed=False)
                dis = (intbytes >> 5) + Bit2FloatTable[intbytes & 0x1F]
                disp.append(dis)
    disparity = np.array(disp).reshape(720, 1280)
    return disparity

def IOU(boxA, boxB):  # [x0,y0,x1,y1]
    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def compute_pred_success(pred_bbox_path, gt_bbox_path):
    with open(pred_bbox_path, 'r') as f_A:
        data_A = json.load(f_A)
        points_A = data_A["shapes"][0]["points"]  # [left_upper + w + h]
        x0_A = points_A[0][0]
        y0_A = points_A[0][1]
        w_A = points_A[1][0]
        h_A = points_A[1][1]
        boxA = [x0_A, y0_A, x0_A + w_A, y0_A + h_A]
    with open(gt_bbox_path, 'r') as f_B:
        data_B = json.load(f_B)
        points_B = data_B["shapes"][0]["points"]  # [left_upper, right_bottom]
        x0_B = points_B[0][0]
        y0_B = points_B[0][1]
        x1_B = points_B[1][0]
        y1_B = points_B[1][1]
        boxB = [x0_B, y0_B, x1_B, y1_B]

    iou = IOU(boxA, boxB)
    row_dis = abs(boxA[2] - boxB[0])
    col_dis = abs(boxA[1] - boxB[1])
    if iou > 0.0 or (row_dis < 200 and col_dis < 20):
        flag = 1
    else:
        flag = 0

    return boxA, flag

def depth(disparity):
    bfValue = 240  # baseline = 120mm
    z = bfValue / disparity
    return z

def disparity_to_depth(disp_selected):
    depth_array = np.zeros_like(disp_selected)
    for i in range(disp_selected.shape[0]):
        for j in range(disp_selected.shape[1]):
            z = depth(disp_selected[i, j])
            depth_array[i, j] = z  # unit: m
    return depth_array

def readCamera(path_CAM):
    baseline = 0.12
    bf_value = 235.0
    mm2m = 1000.0
    file = []
    base_opt_x = 0
    opticalY = 0
    opticalX = 0
    for dirpath, dirnames, filenames in os.walk(path_CAM):
        filenames.sort(key=str.lower)
        for filename in filenames:
            img = os.path.join(dirpath, filename)
            file.append(img)
            if filename == "calibData.json":
                with open(file[len(file) - 1], 'r', encoding='utf-8') as load_f:
                    strF = load_f.read()
                    if len(strF) > 0:
                        datas = json.loads(strF)
                    else:
                        datas = {}
                    base_opt_x = (float(datas["data"][39]) + float(datas["data"][41])) / 2.0   # Cx
                    opticalY = (float(datas["data"][40]) + float(datas["data"][42])) / 2.0   # final Cy
                    fx = float(datas["data"][4])
                    fy = float(datas["data"][12])
                    # baseline = abs(float(datas["data"][22] / mm2m))
            if filename == 'DepthCalib.json':
                with open(file[len(file) - 1], 'r', encoding='utf-8') as load_f:
                    strF = load_f.read()
                    if len(strF) > 0:
                        datas = json.loads(strF)
                    else:
                        datas = {}
                    # bf_value = float(datas["data"][1])
                    delta = float(datas["data"][4]) + float(datas["data"][0])
                    opticalX = base_opt_x - math.floor(delta)    # final Cx  this part has some problem
                    # baseline = abs(float(datas["data"][2]))
    return opticalX, opticalY, fx, fy

def depth2xyz(depth_map, depth_cam_matrix, box, extend_num, depth_scale=1, extend=True):
    if extend == True:
        box[1] = box[1] - extend_num
        box[3] = box[3] + extend_num
        if box[0] < 0:
            box[0] = 0
        if box[1] < 0:
            box[1] = 0
        if box[2] > 1280:
            box[2] = 1280
        if box[3] > 720:
            box[3] = 720
    cameraHeight = 1.450  # unit: m
    fx, fy = depth_cam_matrix[0, 0], depth_cam_matrix[1, 1]
    cx, cy = depth_cam_matrix[0, 2], depth_cam_matrix[1, 2]

    h, w = np.mgrid[int(box[1]): int(box[3]), int(box[0]): int(box[2])]

    z = depth_map[int(box[1]): int(box[3]), int(box[0]): int(box[2])] / depth_scale # unit: mm
    assert z.shape[0] > 0

    x = (w - cx) * z / fx
    y = ((h - cy) * z / fy) - cameraHeight
    xyz = np.dstack((x, y, z)).reshape(-1, 3)
    return xyz

def delete_point(point_cloud, M):
    df = pd.DataFrame(point_cloud, columns=['x','y','z'])
    df1 = df[df['y'] < -M]    # y axis is pointing down
    df2 = df1[df1['y'] != -math.inf]
    del_point = df2.values
    return del_point

def Gaussian(del_point, sigma, N, no_Gaussian, no_depth_filter, bandwidth=2.5):

    df = pd.DataFrame(del_point, columns=['x', 'y', 'z'])
    if not no_Gaussian:
        # sklearn
        z_axis = del_point[:, 2].reshape(-1,1)
        X_plot = np.linspace(0, int(z_axis.max()), int(z_axis.max())+1)[:, np.newaxis]
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(z_axis)
        log_dens = kde.score_samples(X_plot)
        max_value_idx = np.argmax(log_dens)
    else:
        max_value_idx = df['z'].mean()

    if not no_depth_filter:
        center_substrct_sigma = max_value_idx - sigma
        center_plus_sigma = max_value_idx + sigma
        df1 = df[center_substrct_sigma < df['z']]
        df2 = df1[df1['z'] < center_plus_sigma]
    else:
        df2 = df

    df3 = df2.sort_values(by=['y'], ascending=False)
    df4 = df3['y'][:N]
    predict_height = np.abs(np.mean(df4))
    return predict_height

def kalman(value, all_img_names, ground_truth, save_filter_path, Q=1e-5, R=1e-2):
    if not os.path.exists(save_filter_path):
        os.makedirs(save_filter_path)
    n_iter = len(value) + 1   # the number of images
    sz = (n_iter,)  # size of array
    x = 6  # truth value (typo in example at top of p. 13 calls this z)
    z = value  # observations (normal about x, sigma=0.1)
    Q = Q  # process variance
    # allocate space for arrays
    xhat = np.zeros(sz)  # a posteri estimate of x
    P = np.zeros(sz)  # a posteri error estimate
    xhatminus = np.zeros(sz)  # a priori estimate of x
    Pminus = np.zeros(sz)  # a priori error estimate
    K = np.zeros(sz)  # gain or blending factor
    R = R  # estimate of measurement variance, change to see effect
    # intial guesses
    xhat[0] = value[0]
    P[0] = x
    for j in range(1, n_iter):  # 6
        for k in range(1, j):
            # time update
            xhatminus[k] = xhat[k - 1]  # X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0
            Pminus[k] = P[k - 1] + Q  # P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1
            # measurement update
            K[k] = Pminus[k] / (Pminus[k] + R)  # Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1
            xhat[k] = xhatminus[k] + K[k] * (z[k] - xhatminus[k])  # X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1
            P[k] = (1 - K[k]) * Pminus[k]  # P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1

        font = {'family': 'serif',
                'color': 'darkred',
                'weight': 'normal',
                'size': 16,
                }

        plt.figure() # 1280/200； 720/200
        plt.plot([i for i in range(1, j)], z[1:j], 'cx', label='noisy measurements', zorder=2)
        plt.plot([i for i in range(1, j)], [float(ground_truth[i]) for i in range(1, j)], color='tomato', label='ground truth')
        plt.plot([i for i in range(1, j)], xhat[1:j], color='teal', label='a posteri estimate', zorder=1)
        plt.legend(('noisy measurement', 'ground-truth', 'filtered-value'), loc='upper right', shadow=True)
        plt.ylabel('height (m)', fontdict=font)
        plt.xlabel('time (f)', fontdict=font)
        plt.title('Ground-truth: %.2fm, filtered-value: %.2fm' % (ground_truth[0], round(xhat[j - 1], 4)), fontdict=font)
        plt.subplots_adjust(left=0.15)
        plt.savefig(os.path.join(save_filter_path, all_img_names[j-1] + '_filter.jpg'), dpi=150)
        plt.close()

    return xhat[1:]

if __name__ == '__main__':

    data_storage = []
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=float, default=1.5)
    parser.add_argument('--sigma', type=float, default=0.6)
    parser.add_argument('--bandwidth', type=float, default=2.5)
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--extend_num', type=int, default=20)
    parser.add_argument('--tracker', type=str, default='mil')
    parser.add_argument('--record_file', type=str, default='debug')
    parser.add_argument('--no_kalman', action='store_true')
    parser.add_argument('--kalman_Q', type=float, default='1e-3')
    parser.add_argument('--kalman_R', type=float, default='1e-2')
    parser.add_argument('--no_Gaussian', action='store_true')
    parser.add_argument('--no_depth_filter', action='store_true')  
    parser.add_argument('--data_num', type=int, default=14)  
    args = parser.parse_args()
    record_file = open(args.record_file+'.txt', 'a')
    print('M:{} bandwidth:{} sigma:{} N:{} extend_num:{} tracker:{} record_file:{} Gaussian:{} Depth Filter:{} Kalman:{} Kalman_Q:{} Kalman_R:{}\n'.format(args.M, args.bandwidth, args.sigma, args.N, args.extend_num, args.tracker, args.record_file, not args.no_Gaussian, not args.no_depth_filter, not args.no_kalman, args.kalman_Q, args.kalman_R))
    record_file.write('M:{} bandwidth:{} sigma:{} N:{} extend_num:{} tracker:{} record_file:{} Gaussian:{} Depth Filter:{} Kalman:{} Kalman_Q:{} Kalman_R:{}\n'.format(args.M, args.bandwidth, args.sigma, args.N, args.extend_num, args.tracker, args.record_file, not args.no_Gaussian, not args.no_depth_filter, not args.no_kalman, args.kalman_Q, args.kalman_R))

    num = args.data_num
    dat_path = "../data/demo_data/val/images_{:0>3d}/disparity_images".format(int(num))
    path_CAM = '../data/demo_data/adas_params'
    gt_bbox_path = '../data/demo_data/val/images_{:0>3d}/left_colorimages'.format(int(num))
    pred_bbox_path = '../Stage1/faster_rcnn_output/tracking_{}_json/images_{:0>3d}'.format(args.tracker, int(num))
    image_path = '../Stage1/faster_rcnn_output/tracking_{}_image/images_{:0>3d}'.format(args.tracker, int(num))
    save_xyz_path = '../Stage1/faster_rcnn_output/tracking_{}_xyz/images_{:0>3d}'.format(args.tracker, int(num))
    save_filter_path = '../Stage1/faster_rcnn_output/tracking_{}_filter/images_{:0>3d}'.format(args.tracker, int(num))
    M = args.M   # point below M will be removed 实验二
    sigma = args.sigma   # kde center width 实验三
    N = args.N  # select the 50 lowest points, compute mean 实验四
    extend_num = args.extend_num  # bounding box extends 5 pixels up and down 实验一
    all_height, all_disparity, all_rgb, all_img_name= [], [], [], []
    filenames = os.listdir(pred_bbox_path)
    filenames.sort()

    for i in filenames:
        aa = i.split('.')[0].split('_')
        bb = i.split('.')[0] + '.txt'
        cc = i.split('.')[0]
        dat_name = 'disparity_5_' + aa[1] + '_' + aa[2] + '_' + aa[3] + '_' + aa[4] + '.dat'
        box, flag = compute_pred_success(os.path.join(pred_bbox_path, i), os.path.join(gt_bbox_path, i))

        if flag == 1:

            rgb_image = cv2.imread(os.path.join(image_path, i).replace('json', 'jpg'))
            all_rgb.append(rgb_image)
            disparity = preprocess(os.path.join(dat_path, dat_name))
            colored_disparity = cv2.applyColorMap(np.uint8(disparity / np.amax(disparity) * 255), cv2.COLORMAP_JET)
            all_disparity.append(colored_disparity)

            depth_map = disparity_to_depth(disparity)
            cx, cy, fx, fy = readCamera(path_CAM)  # unit: pixel
            depth_cam_matrix = np.array([[fx, 0, cx],
                                        [0, fy, cy],
                                        [0, 0, 1]])
            xyz = depth2xyz(depth_map, depth_cam_matrix, box, extend_num, depth_scale=1, extend=True)
            last_point_cloud = delete_point(xyz, M)
            predict_height = Gaussian(last_point_cloud, sigma, N, args.no_Gaussian, args.no_depth_filter, args.bandwidth)

            all_height.append(predict_height)
            all_img_name.append(cc)

    img_num = pred_bbox_path.split('/')[-1]
 
    all_heights, all_disparitys, all_rgbs, all_img_names= [], [], [], []
    for x, y, z, w in zip(all_height, all_img_name, all_rgb, all_disparity):
        if np.isnan(x) == False:
            all_heights.append(x)
            all_img_names.append(y)
            all_rgbs.append(z)
            all_disparitys.append(w)


    # if img_num == 'images_014':
    #     ground_truth = 3.7
    # elif img_num == 'images_015':
    #     ground_truth = 2.85
    # elif img_num == 'images_001':
    #     ground_truth = 3.2
    # elif img_num == 'images_002':
    #     ground_truth = 3.3
    # elif img_num == 'images_003':
    #     ground_truth = 6.0
    # elif img_num == 'images_004':
    #     ground_truth = 4.7
    # elif img_num == 'images_005':
    #     ground_truth = 5.3
    # elif img_num == 'images_006':
    #     ground_truth = 2.89
    # elif img_num == 'images_007':
    #     ground_truth = 3.86
    # elif img_num == 'images_008':
    #     ground_truth = 2.2
    # elif img_num == 'images_011':
    #     ground_truth = 3.7
    # elif img_num == 'images_012':
    #     ground_truth = 3.3
    # elif img_num == 'images_013':
    #     ground_truth = 3.3
    if img_num == 'images_017':
        ground_truth = 2.85
    elif img_num == 'images_018':
        ground_truth = 2.2
    elif img_num == 'images_019':
        ground_truth = 5.3

    # images_001 3.2m 桥洞
    # images_002 3.3m 桥洞
    # images_003 6.0m 监控杆
    # images_004 4.7m 电线
    # images_005 5.3m 立交桥
    # images_006 2.89m 限高杆 
    # images_007 3.86m 桥洞 
    # images_008 2.2m 限高杆 

    # images_011 3.7m 限高杆
    # images_012 3.3m 限高杆
    # images_013 3.3m 限高杆 同上
    # images_014 3.7m 限高杆
    # images_015 2.85m 限高杆
    # images_016 3.1m 入口

    if not args.no_kalman:
        all_heights = kalman(all_heights, all_img_names, ground_truth * np.ones_like(all_heights), save_filter_path, Q=args.kalman_Q, R=args.kalman_R)
    errors = all_heights - ground_truth * np.ones_like(all_heights)
    error = np.mean(errors)
    accuracy = error / ground_truth


    data_storage.append({'len':len(all_heights),'accuracy':accuracy,'error':error})
    print('average error : {:.2f}m'.format(error))
    print('{} : {:.2%}'.format(img_num, accuracy))
    record_file.write('average error : {:.2f}m\n'.format(error))
    record_file.write('{} : {:.2%}\n'.format(img_num, accuracy))

    os.makedirs('./output/images{:0>3d}'.format(int(num)), exist_ok=True)

    for i in range(0, len(all_heights)):
        # all_heights: 预测高度  error: 与GT误差 all_disparitys: 视差图 all_rgbs: 左视图(带预测框)
        disparity, rgb, height, error = all_disparitys[i], all_rgbs[i], all_heights[i], errors[i]

        fig = plt.figure(figsize=(36, 9))
        # bg = plt.imread("tmp.png")
        # fig.figimage(bg)

        plt.suptitle('SHLE Demo in Scene Image{:0>3d}'.format(int(num)),  fontsize = 80, y=0.99)
        gs = GridSpec(1, 9)
        ax1 = plt.subplot(gs[0, 0:4])
        ax1.axis('off')
        plt.text(550, 800, 'Disparity Map', fontsize=30)

        ax2 = plt.subplot(gs[0, 4:8])
        plt.text(550, 800, 'RGB Image', fontsize=30)
        ax2.axis('off')
        ax3 = plt.subplot(gs[0:, 8])
        ax3.axis('off')
        plt.text(0, 0.5, 'Height: {:.2f}m\nError   : {:.2f}m'.format(height, error), fontsize=50)

        ax1.imshow(disparity)
        ax2.imshow(rgb)
        plt.savefig('./output/images{:0>3d}/frame_{:0>4d}.jpg'.format(int(num), i))
