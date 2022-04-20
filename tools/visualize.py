import os
import pickle
import numpy as np

from pathlib import Path
from skimage import io
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import gridspec

from pcdet.utils.calibration_kitti import get_calib_from_file
from visual_utils.camera_utils import draw_3Dbox, draw_birdeyes


def load_image(root_split_path, frame_id):
    """
    Loads image for a sample
    Args:
        idx [int]: Index of the image sample
    Returns:
        image [np.ndarray(H, W, 3)]: RGB Image
    """
    img_file = root_split_path / 'image_2' / ('%s.png' % frame_id)
    assert img_file.exists()
    image = io.imread(img_file)
    image = image[:, :, :3]  # Remove alpha channel
    image = image.astype(np.float32)
    image /= 255.0
    return image


def load_calib(root_split_path, frame_id):
    calib_file_path = root_split_path / 'calib' / ('%s.txt' % frame_id)
    calib_dict = get_calib_from_file(calib_file_path)
    return calib_dict


def draw_3d_bboxes_on_image(im, res_dict, calib_dict, save_fig_path):
    fig = plt.figure()
    fig.set_size_inches(16, 5)
    gs = gridspec.GridSpec(1, 2)

    # rgb image visualization
    ax1 = plt.subplot(gs[0])
    ax1.axis('off')
    plt.imshow(im, origin='upper')
    plt.xlim(0, im.shape[1])
    plt.ylim(im.shape[0], 0)

    # bev visualization
    ax2 = plt.subplot(gs[1])
    shape = 900
    scale = 15
    birdimage = 250 * np.ones((shape, shape), dtype=np.uint8)
    x1 = np.linspace(0, shape / 2)
    x2 = np.linspace(shape / 2, shape)
    ax2.plot(x1, shape / 2 - x1, ls='--', color='grey', linewidth=1, alpha=0.5)
    ax2.plot(x2, x2 - shape / 2, ls='--', color='grey', linewidth=1, alpha=0.5)
    ax2.plot(shape / 2, 0, marker='+', markersize=16, markeredgecolor='red')

    tick_count = 10
    tick_step = int(shape/tick_count)
    ax2.imshow(birdimage, origin='lower', cmap='gray', vmin=0, vmax=255)
    ax2.set_xticks([i for i in range(tick_step, shape, tick_step)])
    ax2.set_yticks([i for i in range(tick_step, shape, tick_step)])
    ax2.set_xticklabels([round((i - shape/2) / scale, 1)
                        for i in range(tick_step, shape, tick_step)])
    ax2.set_yticklabels([round(i / scale, 1)
                        for i in range(tick_step, shape, tick_step)])
    ax2.grid(alpha=0.2)
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('z (m)')

    for obj_id in range(res_dict['bbox'].shape[0]):
        obj = {
            'tx': res_dict['location'][obj_id, 0],
            'ty': res_dict['location'][obj_id, 1] - res_dict['dimensions'][obj_id, 1] / 2,
            'tz': res_dict['location'][obj_id, 2],
            'h': res_dict['dimensions'][obj_id, 1],
            'w': res_dict['dimensions'][obj_id, 2],
            'l': res_dict['dimensions'][obj_id, 0],
            'yaw': res_dict['rotation_y'][obj_id],
            'pitch': 0,
            'roll': 0
        }
        draw_3Dbox(ax1, calib_dict['P2'], obj, color=(1, 0, 0), is_kitti=True)

        pred = ((obj['tx'], obj['ty'], obj['tz']),
                (obj['h'], obj['w'], obj['l']), obj['yaw'])
        draw_birdeyes(ax2, None, pred, shape, scale)

    plt.savefig(save_fig_path)
    plt.close()


if __name__ == '__main__':
    data_split_root = Path('data/kitti/training')
    result_pkl_path = Path(
        'output/kitti_models/CaDDN/default/eval/epoch_no_number/val/default/result.pkl')
    vis_dir = Path(
        'output/kitti_models/CaDDN/default/eval/epoch_no_number/val/default/vis')
    os.makedirs(vis_dir, exist_ok=True)

    with open(result_pkl_path, "rb") as f:
        results = pickle.load(f)
    for res_dict in tqdm(results):
        im = load_image(data_split_root, res_dict['frame_id'])
        calib_dict = load_calib(data_split_root, res_dict['frame_id'])
        vis_path = vis_dir / ('%s.png' % res_dict['frame_id'])
        draw_3d_bboxes_on_image(
            im, res_dict, calib_dict, save_fig_path=vis_path)
