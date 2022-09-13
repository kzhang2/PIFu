from torch.utils.data import Dataset
import numpy as np
import os
import random
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import cv2
import torch
from PIL.ImageFilter import GaussianBlur
import trimesh
import logging
import pickle

log = logging.getLogger('trimesh')
log.setLevel(40)

def pload(fn, mode='rb'):
    with open(fn, mode) as handle:
        return pickle.load(handle)
homo = lambda v: np.concatenate([
    v, np.ones((len(v),1)),
], axis=1)

def load_trimesh(root_dir):
    folders = os.listdir(root_dir)
    meshs = {}
    for i, f in enumerate(folders):
        sub_name = f
        meshs[sub_name] = trimesh.load(os.path.join(root_dir, f, '%s_100k.obj' % sub_name))

    return meshs

def save_samples_truncted_prob(fname, points, prob):
    '''
    Save the visualization of sampling to a ply file.
    Red points represent positive predictions.
    Green points represent negative predictions.
    :param fname: File name to save
    :param points: [N, 3] array of points
    :param prob: [N, 1] array of predictions in the range [0~1]
    :return:
    '''
    r = (prob > 0.5).reshape([-1, 1]) * 255
    g = (prob < 0.5).reshape([-1, 1]) * 255
    b = np.zeros(r.shape)

    to_save = np.concatenate([points, r, g, b], axis=-1)
    return np.savetxt(fname,
                      to_save,
                      fmt='%.6f %.6f %.6f %d %d %d',
                      comments='',
                      header=(
                          'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                          points.shape[0])
                      )


class TrainDatasetVRM(Dataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt=None, phase='train'):
        self.opt = opt
        self.projection_mode = 'orthogonal'

        # Path setup
        # self.root = self.opt.dataroot
        #self.root = "../aechmea/aechmea"
        self.root = "../aechmeaD"
        self.RENDER = os.path.join(self.root, 'RENDER')
        self.MASK = os.path.join(self.root, 'MASK')
        self.PARAM = os.path.join(self.root, 'PARAM')
        self.UV_MASK = os.path.join(self.root, 'UV_MASK')
        self.UV_NORMAL = os.path.join(self.root, 'UV_NORMAL')
        self.UV_RENDER = os.path.join(self.root, 'UV_RENDER')
        self.UV_POS = os.path.join(self.root, 'UV_POS')
        self.OBJ = os.path.join(self.root, 'GEO', 'OBJ')

        self.B_MIN = np.array([-128, -28, -128])
        self.B_MAX = np.array([128, 228, 128])

        self.is_train = (phase == 'train')
        # self.load_size = self.opt.loadSize
        self.load_size = 512

        # self.num_views = self.opt.num_views
        self.num_views = 1

        # self.num_sample_inout = self.opt.num_sample_inout
        self.num_sample_inout = 5000
        # self.num_sample_color = self.opt.num_sample_color
        self.num_sample_color = 0

        # self.yaw_list = list(range(0,360,1))
        # self.pitch_list = [0]
        
        # hacky
        bn = '6152365338188306398'
        data = pload(f'{self.root}/fixed84_points/{bn[-1]}/{bn}.pkl')
        self.num_cams = data["cameras_extrinsic"].shape[0]
        self.subjects = self.get_subjects()

        # PIL to tensor
        self.to_tensor = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # augmentation
        # self.aug_trans = transforms.Compose([
        #     transforms.ColorJitter(brightness=opt.aug_bri, contrast=opt.aug_con, saturation=opt.aug_sat,
        #                            hue=opt.aug_hue)
        # ])
        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=0, contrast=0, saturation=0,
                                   hue=0)
        ])

        # self.mesh_dic = load_trimesh(self.OBJ)
        
        self.sigma = 5.0
        self.random_multiview = False

    def get_subjects(self):
        all_subjects = os.listdir(f'{self.root}/fixed84_512/8')[3:4]
        var_subjects = []
        # var_subjects = np.loadtxt(os.path.join(self.root, 'val.txt'), dtype=str)
        # if len(var_subjects) == 0:
        #     return all_subjects

        if self.is_train:
            return sorted(list(set(all_subjects) - set(var_subjects)))
        else:
            return sorted(list(var_subjects))

    def __len__(self):
        return len(self.subjects) * self.num_cams

    def get_render(self, subject, num_views, yid=0, pid=0, random_sample=False):
        '''
        Return the render data
        :param subject: subject name
        :param num_views: how many views to return
        :param view_id: the first view_id. If None, select a random one.
        :return:
            'img': [num_views, C, W, H] images
            'calib': [num_views, 4, 4] calibration matrix
            'extrinsic': [num_views, 4, 4] extrinsic matrix
            'mask': [num_views, 1, W, H] masks
        '''
        # pitch = self.pitch_list[pid] # basically unused

        # The ids are an even distribution of num_views around view_id
        # view_ids = [self.yaw_list[(yid + len(self.yaw_list) // num_views * offset) % len(self.yaw_list)]
        #             for offset in range(num_views)]
        
        # we only use 1 view in practice
        view_ids = [yid]
        
        # unused 
        if random_sample:
            view_ids = np.random.choice(self.yaw_list, num_views, replace=False)

        calib_list = []
        render_list = []
        mask_list = []
        extrinsic_list = []

        for vid in view_ids:
            # param_path = os.path.join(self.PARAM, subject, '%d_%d_%02d.npy' % (vid, pitch, 0))
            # render_path = os.path.join(self.RENDER, subject, '%d_%d_%02d.jpg' % (vid, pitch, 0))
            # mask_path = os.path.join(self.MASK, subject, '%d_%d_%02d.png' % (vid, pitch, 0))
            
            param_path = f'{self.root}/fixed84_points/{subject[-1]}/{subject}.pkl'
            render_path = f'{self.root}/fixed84_512/{subject[-1]}/{subject}/{vid:04d}.png'

            # loading calibration data
            # preload data into array? 
            data = pload(param_path)
            
            # param = np.load(param_path, allow_pickle=True)
            # # pixel unit / world unit
            # ortho_ratio = param.item().get('ortho_ratio')
            # # world unit / model unit
            # scale = param.item().get('scale')
            # # camera center world coordinate
            # center = param.item().get('center')
            # # model rotation
            # R = param.item().get('R')
            
            extrinsic = data["cameras_extrinsic"][vid]

            # translate = -np.matmul(R, center).reshape(3, 1)
            # extrinsic = np.concatenate([R, translate], axis=1)
            # extrinsic = np.concatenate([extrinsic, np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
            # Match camera space to image pixel space
            # scale_intrinsic = np.identity(4)
            # scale_intrinsic[0, 0] = scale / ortho_ratio
            # scale_intrinsic[1, 1] = -scale / ortho_ratio
            # scale_intrinsic[2, 2] = scale / ortho_ratio
            # # Match image pixel space to image uv space
            # uv_intrinsic = np.identity(4)
            # uv_intrinsic[0, 0] = 1.0 / float(self.load_size // 2)
            # uv_intrinsic[1, 1] = 1.0 / float(self.load_size // 2)
            # uv_intrinsic[2, 2] = 1.0 / float(self.load_size // 2)
            
            orig_intrinsic = data["cameras_intrinsic"][vid]
            # Transform under image pixel space
            trans_intrinsic = np.identity(4)

            # mask = Image.open(mask_path).convert('L')
            render = Image.open(render_path).convert('RGB')
            mask = Image.fromarray(np.uint8((np.array(render) > 0).astype(float)) * 255)

            # hack to temporarily turn off data augmentation 
            self.is_train = False
            if self.is_train:
                # Pad images
                pad_size = int(0.1 * self.load_size)
                render = ImageOps.expand(render, pad_size, fill=0)
                mask = ImageOps.expand(mask, pad_size, fill=0)

                w, h = render.size
                th, tw = self.load_size, self.load_size

                # random flip
                if np.random.rand() > 0.5:
                    scale_intrinsic[0, 0] *= -1
                    render = transforms.RandomHorizontalFlip(p=1.0)(render)
                    mask = transforms.RandomHorizontalFlip(p=1.0)(mask)

                # random scale
                if True:
                    rand_scale = random.uniform(0.9, 1.1)
                    w = int(rand_scale * w)
                    h = int(rand_scale * h)
                    render = render.resize((w, h), Image.BILINEAR)
                    mask = mask.resize((w, h), Image.NEAREST)
                    scale_intrinsic *= rand_scale
                    scale_intrinsic[3, 3] = 1

                # random translate in the pixel space
                if True:
                    dx = random.randint(-int(round((w - tw) / 10.)),
                                        int(round((w - tw) / 10.)))
                    dy = random.randint(-int(round((h - th) / 10.)),
                                        int(round((h - th) / 10.)))
                else:
                    dx = 0
                    dy = 0

                trans_intrinsic[0, 3] = -dx / float(self.load_size // 2)
                trans_intrinsic[1, 3] = -dy / float(self.load_size // 2)

                x1 = int(round((w - tw) / 2.)) + dx
                y1 = int(round((h - th) / 2.)) + dy

                render = render.crop((x1, y1, x1 + tw, y1 + th))
                mask = mask.crop((x1, y1, x1 + tw, y1 + th))

                render = self.aug_trans(render)

                # random blur
                # if self.opt.aug_blur > 0.00001:
                #     blur = GaussianBlur(np.random.uniform(0, self.opt.aug_blur))
                #     render = render.filter(blur)

            #intrinsic = np.matmul(trans_intrinsic, orig_intrinsic)

            # very hacky
            intrinsic = orig_intrinsic
            calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()
            perm = torch.tensor([[0.0, 1.0, 0.0],
                                 [1.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0]])
            calib[:3, :3] = perm @ calib[:3, :3] * 2
            calib[:3, 3:4] = perm @ calib[:3, 3:4] * 2
            calib[0, 3:4] -= 1
            calib[1, 3:4] += 1    
            
            # hacky manual scaling due to pifu depth normalization + centimeter unit usage
            # vs vrm data meter unit usage
            calib = calib / 100
            calib[3, 3] = 1.0

            # just extrinsic is actually unused in the code?
            extrinsic = torch.Tensor(extrinsic).float() 
            extrinsic[:3, :3] = perm @ extrinsic[:3, :3] * 2
            extrinsic[:3, 3:4] = perm @ extrinsic[:3, 3:4] * 2
            extrinsic[0, 3:4] -= 1
            extrinsic[1, 3:4] += 1    

            mask = transforms.Resize(self.load_size)(mask)
            mask = transforms.ToTensor()(mask).float()
            mask_list.append(mask)

            render = self.to_tensor(render)
            render = mask.expand_as(render) * render

            render_list.append(render)
            calib_list.append(calib)
            extrinsic_list.append(extrinsic)

        return {
            'img': torch.stack(render_list, dim=0),
            'calib': torch.stack(calib_list, dim=0),
            'extrinsic': torch.stack(extrinsic_list, dim=0),
            'mask': torch.stack(mask_list, dim=0)
        }

    def select_sampling_method(self, subject):
        if not self.is_train:
            random.seed(1991)
            np.random.seed(1991)
            torch.manual_seed(1991)
        # mesh = self.mesh_dic[subject]
        
        # fold both below into one function? 
        param_path = f'{self.root}/fixed84_points/{subject[-1]}/{subject}.pkl'
        data = pload(param_path)

        # surface points
        surf_inds = np.argwhere(data["sampling_strategy"] == 1).ravel()
        surf_inds_sampled = np.random.choice(surf_inds, 4 * self.num_sample_inout, replace=False)
        surf_labels = data["winding_number"][surf_inds_sampled]
        surf_labels[surf_labels > 0.5] = 1
        surf_labels[surf_labels <= 0.5] = 0
        sample_points = data["xyz"][surf_inds_sampled]

        # add random points within image space
        rand_inds = np.argwhere(data["sampling_strategy"] == 0).ravel()
        rand_inds_sampled = np.random.choice(rand_inds, self.num_sample_inout // 4, replace=False)
        rand_labels = data["winding_number"][rand_inds_sampled]
        rand_labels[rand_labels > 0.5] = 1
        rand_labels[rand_labels <= 0.5] = 0
        random_points = data["xyz"][rand_inds_sampled]
        
        samples = np.concatenate([sample_points, random_points], 0).T
        labels = np.concatenate([surf_labels, rand_labels], 0)[None, ...] # hack
        # np.random.shuffle(sample_points)

        # save_samples_truncted_prob('out.ply', samples.T, labels.T)
        # exit()

        # hacky scaling; see note about calibration matrix in get_render
        samples = samples * 100 

        samples = torch.Tensor(samples).float()
        labels = torch.Tensor(labels).float()

        return {
            'samples': samples,
            'labels': labels
        }


    def get_color_sampling(self, subject, yid, pid=0):
        yaw = self.yaw_list[yid]
        pitch = self.pitch_list[pid]
        uv_render_path = os.path.join(self.UV_RENDER, subject, '%d_%d_%02d.jpg' % (yaw, pitch, 0))
        uv_mask_path = os.path.join(self.UV_MASK, subject, '%02d.png' % (0))
        uv_pos_path = os.path.join(self.UV_POS, subject, '%02d.exr' % (0))
        uv_normal_path = os.path.join(self.UV_NORMAL, subject, '%02d.png' % (0))

        # Segmentation mask for the uv render.
        # [H, W] bool
        uv_mask = cv2.imread(uv_mask_path)
        uv_mask = uv_mask[:, :, 0] != 0
        # UV render. each pixel is the color of the point.
        # [H, W, 3] 0 ~ 1 float
        uv_render = cv2.imread(uv_render_path)
        uv_render = cv2.cvtColor(uv_render, cv2.COLOR_BGR2RGB) / 255.0

        # Normal render. each pixel is the surface normal of the point.
        # [H, W, 3] -1 ~ 1 float
        uv_normal = cv2.imread(uv_normal_path)
        uv_normal = cv2.cvtColor(uv_normal, cv2.COLOR_BGR2RGB) / 255.0
        uv_normal = 2.0 * uv_normal - 1.0
        # Position render. each pixel is the xyz coordinates of the point
        uv_pos = cv2.imread(uv_pos_path, 2 | 4)[:, :, ::-1]

        ### In these few lines we flattern the masks, positions, and normals
        uv_mask = uv_mask.reshape((-1))
        uv_pos = uv_pos.reshape((-1, 3))
        uv_render = uv_render.reshape((-1, 3))
        uv_normal = uv_normal.reshape((-1, 3))

        surface_points = uv_pos[uv_mask]
        surface_colors = uv_render[uv_mask]
        surface_normal = uv_normal[uv_mask]

        if self.num_sample_color:
            sample_list = random.sample(range(0, surface_points.shape[0] - 1), self.num_sample_color)
            surface_points = surface_points[sample_list].T
            surface_colors = surface_colors[sample_list].T
            surface_normal = surface_normal[sample_list].T

        # Samples are around the true surface with an offset
        normal = torch.Tensor(surface_normal).float()
        samples = torch.Tensor(surface_points).float() \
                  + torch.normal(mean=torch.zeros((1, normal.size(1))), std=self.sigma).expand_as(normal) * normal

        # Normalized to [-1, 1]
        rgbs_color = 2.0 * torch.Tensor(surface_colors).float() - 1.0

        return {
            'color_samples': samples,
            'rgbs': rgbs_color
        }

    def get_item(self, index):
        # In case of a missing file or IO error, switch to a random sample instead
        # try:
        sid = index % len(self.subjects) # subject ID
        tmp = index // len(self.subjects) 
        yid = tmp 
        pid = 0

        # name of the subject 'rp_xxxx_xxx'
        subject = self.subjects[sid]
        res = {
            'name': subject,
            'mesh_path': os.path.join(self.OBJ, subject + '.obj'),
            'sid': sid,
            'yid': yid,
            'pid': pid,
            'b_min': self.B_MIN,
            'b_max': self.B_MAX,
        }
        render_data = self.get_render(subject, num_views=self.num_views, yid=yid, pid=pid,
                                        random_sample=self.random_multiview)
        res.update(render_data)

        if self.num_sample_inout:
            sample_data = self.select_sampling_method(subject)
            res.update(sample_data)
        
        img = np.uint8((np.transpose(render_data['img'][0].numpy(), (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0)
        #print(len(render_data['calib']))
        rot = render_data['calib'][0,:3, :3] 
        trans = render_data['calib'][0,:3, 3:4] 
        #perm = torch.tensor([[0.0, 1.0, 0.0],
        #                     [1.0, 0.0, 0.0],
        #                     [0.0, 0.0, 1.0]])
        #rot = perm @ rot * 2
        #trans = perm @ trans * 2
        #trans[0, :] -= 1
        #trans[1, :] += 1    
        pts = torch.addmm(trans, rot, sample_data['samples'][:, sample_data['labels'][0] > 0.5])  # [3, N], inside points
        pts = 0.5 * (pts.numpy().T + 1.0) * render_data['img'].size(2)
        for p in pts:
            #print(p[0], p[1])
            # just for display purposes: in cv2, 0 is top of image, whereas for coords 0 is bottom
            img = cv2.circle(img.copy(), (int(p[0]), int(p[1])), 2, (0,255,0), -1)
        res['vis'] = img
        # cv2.imshow('test', img)
        # cv2.waitKey(1)

        if self.num_sample_color:
            color_data = self.get_color_sampling(subject, yid=yid, pid=pid)
            res.update(color_data)
        return res
        # except Exception as e:
        #     print(e)
        #     return self.get_item(index=random.randint(0, self.__len__() - 1))

    def __getitem__(self, index):
        return self.get_item(index)
