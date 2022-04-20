import numpy as np
from matplotlib.path import Path
import matplotlib.patches as patches
from math import sin, cos
from scipy.spatial.transform import Rotation


class detectionInfo(object):
    def __init__(self, line):
        self.name = line[0]

        self.truncation = float(line[1])
        self.occlusion = int(line[2])

        # local orientation = alpha + pi/2
        self.alpha = float(line[3])

        # in pixel coordinate
        self.xmin = float(line[4])
        self.ymin = float(line[5])
        self.xmax = float(line[6])
        self.ymax = float(line[7])

        # height, weigh, length in object coordinate, meter
        self.h = float(line[8])
        self.w = float(line[9])
        self.l = float(line[10])

        # x, y, z in camera coordinate, meter
        self.tx = float(line[11])
        self.ty = float(line[12])
        self.tz = float(line[13])

        # global orientation [-pi, pi]
        self.rot_global = float(line[14])

    def member_to_list(self):
        output_line = []
        for name, value in vars(self).items():
            output_line.append(value)
        return output_line

    def box3d_candidate(self, rot_local, soft_range):
        x_corners = [self.l, self.l, self.l, self.l, 0, 0, 0, 0]
        y_corners = [self.h, 0, self.h, 0, self.h, 0, self.h, 0]
        z_corners = [0, 0, self.w, self.w, self.w, self.w, 0, 0]

        x_corners = [i - self.l / 2 for i in x_corners]
        y_corners = [i - self.h for i in y_corners]
        z_corners = [i - self.w / 2 for i in z_corners]

        corners_3d = np.transpose(np.array([x_corners, y_corners, z_corners]))
        point1 = corners_3d[0, :]
        point2 = corners_3d[1, :]
        point3 = corners_3d[2, :]
        point4 = corners_3d[3, :]
        point5 = corners_3d[6, :]
        point6 = corners_3d[7, :]
        point7 = corners_3d[4, :]
        point8 = corners_3d[5, :]

        # set up projection relation based on local orientation
        xmin_candi = xmax_candi = ymin_candi = ymax_candi = 0

        if 0 < rot_local < np.pi / 2:
            xmin_candi = point8
            xmax_candi = point2
            ymin_candi = point2
            ymax_candi = point5

        if np.pi / 2 <= rot_local <= np.pi:
            xmin_candi = point6
            xmax_candi = point4
            ymin_candi = point4
            ymax_candi = point1

        if np.pi < rot_local <= 3 / 2 * np.pi:
            xmin_candi = point2
            xmax_candi = point8
            ymin_candi = point8
            ymax_candi = point1

        if 3 * np.pi / 2 <= rot_local <= 2 * np.pi:
            xmin_candi = point4
            xmax_candi = point6
            ymin_candi = point6
            ymax_candi = point5

        # soft constraint
        div = soft_range * np.pi / 180
        if 0 < rot_local < div or 2*np.pi-div < rot_local < 2*np.pi:
            xmin_candi = point8
            xmax_candi = point6
            ymin_candi = point6
            ymax_candi = point5

        if np.pi - div < rot_local < np.pi + div:
            xmin_candi = point2
            xmax_candi = point4
            ymin_candi = point8
            ymax_candi = point1

        return xmin_candi, xmax_candi, ymin_candi, ymax_candi


def get_xzgrid(zx_dim=(128, 101), zrange=30.0):
    """
    BEV grids when transfer RF images to cart coordinates
    :param zx_dim: dimension of output BEV image
    :param zrange: largest range value in z axis
    """
    origin = np.array([0, int(zx_dim[1] / 2)])
    zline, zreso = np.linspace(
        0, zrange, num=zx_dim[0], endpoint=False, retstep=True)
    xmax = zreso * (origin[1] + 1)
    xline = np.linspace(0, xmax, num=origin[1] + 1, endpoint=False)
    xline = np.concatenate([np.flip(-xline[1:]), xline])
    return xline, zline


def xz2idx_interpolate(x, z, x_grid, z_grid):
    """get interpolated XZ indices in float"""
    xids = np.arange(x_grid.shape[0])
    zids = np.arange(z_grid.shape[0])
    x_id = np.interp(x, x_grid, xids)
    z_id = np.interp(z, z_grid, zids)
    return x_id, z_id


def compute_birdviewbox(annotation, shape, scale):
    # annotation: ((x, y, z)), (h, w, l)), rot_y)
    position, dimension, rot_y = annotation
    [h, w, l, x, y, z, rot_y] = np.array(
        [*dimension, *position, rot_y], dtype=np.float64) * scale

    R = np.array([[-np.cos(rot_y), np.sin(rot_y)],
                  [np.sin(rot_y), np.cos(rot_y)]])
    t = np.array([x, z]).reshape(1, 2).T

    x_corners = [0, l, l, 0]  # -l/2
    z_corners = [w, w, 0, 0]  # -w/2

    x_corners += -l / 2
    z_corners += -w / 2

    # bounding box in object coordinate
    corners_2D = np.array([x_corners, z_corners])
    # rotate
    corners_2D = R.dot(corners_2D)
    # translation
    corners_2D = t - corners_2D
    # in camera coordinate
    corners_2D[0] += int(shape/2)
    corners_2D = (corners_2D).astype(np.int16)
    corners_2D = corners_2D.T

    # (9, 2): includes the first corner at the end
    return np.vstack((corners_2D, corners_2D[0, :]))


def draw_birdeyes(ax2, gt, prediction, shape=900, scale=15):
    # gt and prediction: ((x, y, z)), (h, w, l)), rot_y), scale: rotio of pixel / meter
    # shape = 900

    pred_corners_2d = compute_birdviewbox(gt, shape, scale)
    codes = [Path.LINETO] * pred_corners_2d.shape[0]
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    pth = Path(pred_corners_2d, codes)
    p = patches.PathPatch(pth, fill=False, color='green', label='prediction')
    ax2.add_patch(p)

    if prediction is not None:
        gt_corners_2d = compute_birdviewbox(prediction, shape, scale)
        codes = [Path.LINETO] * gt_corners_2d.shape[0]
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY
        pth = Path(gt_corners_2d, codes)
        p = patches.PathPatch(
            pth, fill=False, color='orange', label='ground truth')
        ax2.add_patch(p)


def draw_on_chirp(ax3, line_p, x_grid, z_grid, color, scale_xyz):
    # shape = 900
    obj = detectionInfo(line_p)

    R = np.array([[np.cos(obj.rot_global), np.sin(obj.rot_global)],
                  [-np.sin(obj.rot_global), np.cos(obj.rot_global)]])
    t = np.array([obj.tx * scale_xyz[0], obj.tz *
                 scale_xyz[2]]).reshape(1, 2).T

    x_corners = [0, obj.l, obj.l, 0]  # -l/2
    z_corners = [obj.w, obj.w, 0, 0]  # -w/2

    x_corners += -np.float64(obj.l) / 2
    z_corners += -np.float64(obj.w) / 2

    # bounding box in object coordinate
    corners_2D = np.array([x_corners, z_corners])
    # rotate
    corners_2D = R.dot(corners_2D)
    # translation
    corners_2D = corners_2D + t

    x_coors, z_coors = xz2idx_interpolate(
        corners_2D[0, :], corners_2D[1, :], x_grid, z_grid)
    corners_2D = np.concatenate(
        (x_coors[:, None], z_coors[:, None]), axis=1).astype(np.int16)
    corners_2D = np.vstack((corners_2D, corners_2D[0, :]))

    codes = [Path.LINETO] * corners_2D.shape[0]
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    pth = Path(corners_2D, codes)
    p = patches.PathPatch(pth, fill=False, color=color, label='auxiliary')
    ax3.add_patch(p)

    # draw center of cars
    x = np.float64(obj.tx) * scale_xyz[0]
    z = np.float64(obj.tz) * scale_xyz[2]
    x_centers, z_centers = xz2idx_interpolate(x, z, x_grid, z_grid)
    ax3.scatter(x=x_centers, y=z_centers, c=color, alpha=.5)


def GIOU(gt_bb, pred_bb):
    # gt_bb, pred_bb: (4, ), (4, )
    x_g1, x_g2, y_g1, y_g2 = gt_bb[0], gt_bb[2], gt_bb[1], gt_bb[3]
    x_p1, x_p2, y_p1, y_p2 = pred_bb[0], pred_bb[2], pred_bb[1], pred_bb[3]

    A_g = (x_g2 - x_g1) * (y_g2 - y_g1)
    A_p = (x_p2 - x_p1) * (y_p2 - y_p1)
    x_I1, x_I2, y_I1, y_I2 = np.maximum(x_p1, x_g1), np.minimum(
        x_p2, x_g2), np.maximum(y_p1, y_g1), np.minimum(y_p2, y_g2)  # intersection
    A_I = np.clip((x_I2 - x_I1) * (y_I2 - y_I1), 0, None)
    A_U = A_g + A_p - A_I

    # area of the smallest enclosing box
    min_box = np.minimum(gt_bb, pred_bb)
    max_box = np.maximum(gt_bb, pred_bb)
    A_C = (max_box[2] - min_box[0]) * (max_box[3] - min_box[1])

    iou = A_I / A_U
    giou = iou - (A_C - A_U) / A_C
    return float(giou)


# def draw_2Dbox(ax, P2, line, color, draw_gt, pitch, scale_xyz, w_h):
#     K = P2[:, :3]
#     obj = detectionInfo(line)
#     # todo: incoporate pitch to encode_label()
#     _, pred_box, _ = encode_label(K, obj.rot_global, (obj.l, obj.h, obj.w), locs=(obj.tx * scale_xyz[0], obj.ty * scale_xyz[1], obj.tz * scale_xyz[2]), rx=pitch)
#     xmin = int(np.clip(pred_box[0], 0., float(w_h[0])))
#     ymin = int(np.clip(pred_box[1], 0., float(w_h[1])))
#     xmax = int(np.clip(pred_box[2], 0., float(w_h[0])))
#     ymax = int(np.clip(pred_box[3], 0., float(w_h[1])))
#     width = xmax - xmin
#     height = ymax - ymin
#     box_2d = patches.Rectangle((xmin, ymin), width, height, fill=False, color=color, linewidth='1')
#     ax.add_patch(box_2d)

#     if draw_gt:
#         xmin_gt = int(float(line[16]))
#         xmax_gt = int(float(line[18]))
#         ymin_gt = int(float(line[17]))
#         ymax_gt = int(float(line[19]))
#         width = xmax_gt - xmin_gt
#         height = ymax_gt - ymin_gt
#         box_2d = patches.Rectangle((xmin_gt, ymin_gt), width, height, fill=False, color='r', linewidth='1')
#         ax.add_patch(box_2d)
#         # add GIOU text
#         gt, pred = np.array([xmin_gt, ymin_gt, xmax_gt, ymax_gt]), np.array([xmin, ymin, xmax, ymax])
#         giou = GIOU(gt, pred)
#         bb_min = np.minimum(gt, pred)
#         ax.text(int((xmin_gt + xmax_gt)/2), int(bb_min[1]) -5, f'GIoU:{giou:.2f}', color='lime', size='x-small')


def euler_to_Rot(yaw, pitch, roll, order='yxz'):

    Y = np.array([[cos(yaw), 0, sin(yaw)],
                  [0, 1, 0],
                  [-sin(yaw), 0, cos(yaw)]])
    P = np.array([[1, 0, 0],
                  [0, cos(pitch), -sin(pitch)],
                  [0, sin(pitch), cos(pitch)]])
    R = np.array([[cos(roll), -sin(roll), 0],
                  [sin(roll), cos(roll), 0],
                  [0, 0, 1]])

    rot_mat = {
        'y': Y,
        'x': P,
        'z': R
    }

    rot = {
        'x': pitch,
        'y': yaw,
        'z': roll
    }
    # return np.dot(rot_mat[order[2]], np.dot(rot_mat[order[1]], rot_mat[order[0]]))
    return Rotation.from_euler(order, [rot[order[0]], rot[order[1]], rot[order[2]]]).as_matrix()


def compute_3Dbox(P2, obj, is_kitti, scale_xyz=[1., 1., 1.]):
    rotation = euler_to_Rot(obj['yaw'], obj['pitch'], obj['roll']
                            ) if is_kitti else Rotation.from_quat(obj['quat'])
    # euler = [obj['pitch'], obj['yaw'], obj['roll']]
    R = rotation  # .as_matrix()
    # R = euler_to_Rot(obj['yaw'], obj['pitch'], obj['roll'])
    # R = euler_to_Rot(euler2[1], euler2[0], euler2[2])
    # print(quat)
    # print(obj['w'], obj['h'], obj['l'])
    x_corners = [0, obj['w'], obj['w'], obj['w'], obj['w'], 0, 0, 0]
    y_corners = [0, 0, obj['h'], obj['h'], 0, 0, obj['h'], obj['h']]
    z_corners = [0, 0, 0, obj['l'], obj['l'], obj['l'], obj['l'], 0]

    x_corners = [i - obj['w'] / 2 for i in x_corners]
    y_corners = [i - obj['h'] / 2 for i in y_corners]
    z_corners = [i - obj['l'] / 2 for i in z_corners]

    corners_3D = np.array([x_corners, y_corners, z_corners])
    corners_3D = R.dot(corners_3D)
    corners_3D += np.array([obj['tx'] * scale_xyz[0], obj['ty']
                           * scale_xyz[1], obj['tz'] * scale_xyz[2]]).reshape((3, 1))

    corners_3D_1 = np.vstack((corners_3D, np.ones((corners_3D.shape[-1]))))
    corners_2D = P2.dot(corners_3D_1)
    corners_2D = corners_2D / corners_2D[2]
    corners_2D = corners_2D[:2]

    return corners_2D


def direct_compute_3Dbox(P2, P1, psr):
    R = euler_to_Rot(psr['rotation']['y'], psr['rotation']
                     ['x'], psr['rotation']['z'], order='zyx')
    scale, t = psr['scale'], psr['position']
    x_corners = [0, scale['x'], scale['x'], scale['x'], scale['x'], 0, 0, 0]
    y_corners = [0, 0, scale['y'], scale['y'], 0, 0, scale['y'], scale['y']]
    z_corners = [0, 0, 0, scale['z'], scale['z'], scale['z'], scale['z'], 0]

    x_corners = [i - scale['x'] / 2 for i in x_corners]
    y_corners = [i - scale['y'] / 2 for i in y_corners]
    z_corners = [i - scale['z'] / 2 for i in z_corners]

    corners_3D = np.array([x_corners, y_corners, z_corners])
    corners_3D = R.dot(corners_3D)
    corners_3D += np.array([t['x'], t['y'], t['z']]).reshape((3, 1))

    corners_3D_1 = np.vstack((corners_3D, np.ones((corners_3D.shape[-1]))))
    corners_2D = P2.dot(P1.dot(corners_3D_1))
    corners_2D = corners_2D / corners_2D[2]
    corners_2D = corners_2D[:2]

    return corners_2D


def draw_3Dbox(ax, P2, obj, color, is_kitti=True):
    # Use this to draw 3d boudning box

    corners_2D = compute_3Dbox(P2, obj, is_kitti)
    # draw all lines through path
    # https://matplotlib.org/users/path_tutorial.html
    bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]
    bb3d_on_2d_lines_verts = corners_2D[:, bb3d_lines_verts_idx]
    verts = bb3d_on_2d_lines_verts.T
    codes = [Path.LINETO] * verts.shape[0]
    codes[0] = Path.MOVETO
    # codes[-1] = Path.CLOSEPOLYq
    pth = Path(verts, codes)
    p = patches.PathPatch(pth, fill=False, color=color, linewidth=1)

    # put a mask on the front
    ax.plot([corners_2D[0, 3], corners_2D[0, 1]], [
            corners_2D[1, 3], corners_2D[1, 1]], color='r', lw=1)
    ax.plot([corners_2D[0, 4], corners_2D[0, 2]], [
            corners_2D[1, 4], corners_2D[1, 2]], color='r', lw=1)
    # for i in range(corners_2D.shape[1]):
    #     ax.text(corners_2D[0, i], corners_2D[1, i], i, c='g', fontsize='xx-large')
    ax.add_patch(p)


def draw_box_center(ax, P2, obj, color):
    center_3D_1 = np.array([obj['tx'], obj['ty'], obj['tz'], 1]).reshape(4, 1)
    center_2D = P2.dot(center_3D_1)
    center_2D = center_2D / center_2D[2]
    center_2D = center_2D[:2]
    ax.scatter(center_2D[0], center_2D[1], s=50, c=color)


# Debug purpose
# # l13, l23(# of sampled points, 3(x, y1 or y2, y3))
# def draw_sampled_points(ax, l13, l23):
#     # todo draw sampled points
#     l12_point_idx = (l13[:, 2] < 0).nonzero(as_tuple=True)[0]
#     l12_line_idx = (l13[:, 2] >= 0).nonzero(as_tuple=True)[0]
#     # draw sampled points on projected 3d bbox with no vertically matched hull mask
#     ax.scatter(l13[l12_point_idx, 0], l13[l12_point_idx, 1], s=1, c='green')
#     # draw line in matched pair
#     for i in l12_line_idx:
#         ax.plot([l13[i, 0], l13[i, 0]], [l13[i, 1], l13[i, 2]], '-go', linewidth=1, markersize=2)

#     l13_point_idx = (l23[:, 2] < 0).nonzero(as_tuple=True)[0]
#     l13_line_idx = (l23[:, 2] >= 0).nonzero(as_tuple=True)[0]

#     # draw sampled points on projected 3d bbox with no vertically matched hull mask
#     ax.scatter(l23[l13_point_idx, 0], l23[l13_point_idx, 1], s=1, c='red')
#     # draw line in matched pair
#     for i in l13_line_idx:
#         ax.plot([l23[i, 0], l23[i, 0]], [l23[i, 1], l23[i, 2]], '-ro', linewidth=1, markersize=2)


# def draw_hull_mask(line, im):
#     encoded_rle = {
#             'size': [float(line[20]), float(line[21])],
#             'counts': line[22]
#     }
#     decoded_mask = mask.decode(encoded_rle) # (mask_h, mask_w) np array
#     color = [0, 0, 255]
#     mask_blue = decoded_mask[:, :, None].repeat(3, axis=2) * np.array(color, dtype=np.uint8).reshape((1, 1, 3))
#     im =  im * ~decoded_mask.astype(bool)[:, :, None].repeat(3, axis=2) + mask_blue
#     return im


if __name__ == '__main__':
    # use this to draw 3d bounding box
    # ax is plt axis, P2 is a projection matrix or intrinsics (np array 3x4), obj: dictionary of keys {'tx', 'ty', 'tz', 'h', 'w', 'l', 'yaw', 'pitch', 'roll'}, color: bounding box color
    # draw_3Dbox(ax, P2, obj, color, is_kitti=True)
    # use this to draw bird eye view
    # gt and prediction: ((x, y, z)), (h, w, l)), rot_y)
    gt = None
    # draw_birdeyes(ax2, gt, prediction)
    pass
