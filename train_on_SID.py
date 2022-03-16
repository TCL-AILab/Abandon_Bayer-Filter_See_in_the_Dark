import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM

import numpy as np
import os
import tqdm
import random
import imageio
import rawpy
import glob

from network import our_Net


def image_read(short_expo_files, long_expo_files):
    """
    load image data to CPU ram

    input: (short exposure images' path list, long exposure images' path list)
    output: datalist
    """
    short_list = []
    long_list = []

    for i in tqdm.tqdm(range(len(short_expo_files))):

        raw = rawpy.imread(short_expo_files[i])
        img = raw.raw_image_visible.astype(np.float32).copy()
        raw.close()

        img_short = (np.maximum(img - 512, 0) / (16383 - 512))

        if long_expo_files[i][-7] == '3':
            ap = 300
        else:
            ap = 100

        img_short = (img_short * ap)
        short_list.append(img_short)

        raw = rawpy.imread(long_expo_files[i])
        img = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16).copy()
        raw.close()
        img_long = np.float32(img / 65535.0)
        long_list.append(img_long)

    return short_list, long_list

class load_data(Dataset):
    """Loads the Data."""

    def __init__(self, short_expo_files, long_expo_files, training=True):

        self.training = training
        if self.training:
            print('\n...... Train files loading\n')
            self.short_list, self.long_list = image_read(short_expo_files, long_expo_files)
            print('\nTrain files loaded ......\n')
        else:
            print('\n...... Test files loading\n')
            self.short_list, self.long_list = image_read(short_expo_files, long_expo_files)
            print('\nTest files loaded ......\n')

    def __len__(self):
        return len(self.short_list)

    def __getitem__(self, idx):

        img_short = self.short_list[idx]
        img_long = self.long_list[idx]

        H, W = img_short.shape

        # if training: crop image to 512*512
        # if testing: use whole image
        if self.training:
            i = random.randint(0, (H - 512 - 2) // 2) * 2
            j = random.randint(0, (W - 512 - 2) // 2) * 2

            img_short_crop = img_short[i:i + 512, j:j + 512]
            img_long_crop = img_long[i:i + 512, j:j + 512, :]

            if random.randint(0, 100) > 50:
                img_short_crop = np.fliplr(img_short_crop).copy()
                img_long_crop = np.fliplr(img_long_crop).copy()

            if random.randint(0, 100) < 20:
                img_short_crop = np.flipud(img_short_crop).copy()
                img_long_crop = np.flipud(img_long_crop).copy()

        else:
            img_short_crop = img_short
            img_long_crop = img_long

        img_short = torch.from_numpy(img_short_crop).float().unsqueeze(0)
        img_long = torch.from_numpy((np.transpose(img_long_crop, [2, 0, 1]))).float()

        return img_short, img_long

def run_test(model, dataloader_test, iteration, save_images_file, save_csv_file, metric_average_filename):
    psnr1 = ['PSNR_1']
    ssim1 = ['SSIM_1']
    psnr2 = ['PSNR_2']
    ssim2 = ['SSIM_2']

    with torch.no_grad():
        model.eval()
        for image_num, img in tqdm.tqdm(enumerate(dataloader_test)):
            input_raw = img[0].to(next(model.parameters()).device)
            gt_rgb = img[1][0].detach().cpu().numpy().transpose(1,2,0) * 255
            gt_rgb = np.clip(gt_rgb.astype(np.uint8),0,255)
            gt_mono = 0.2989 * gt_rgb[:, :, 0] + 0.5870 * gt_rgb[:, :, 1] + 0.1140 * gt_rgb[:, :, 2]
            gt_mono = np.expand_dims(gt_mono,-1)
            gt_mono = np.clip(gt_mono.astype(np.uint8),0,255)

            pred_mono, pred_rgb = model(input_raw)

            pred_mono = (np.clip(pred_mono[0].detach().cpu().numpy().transpose(1,2,0), 0, 1) * 255).astype(np.uint8)
            pred_rgb = (np.clip(pred_rgb[0].detach().cpu().numpy().transpose(1,2,0), 0, 1) * 255).astype(np.uint8)

            psnr_mono_img = PSNR(pred_mono, gt_mono)
            ssim_mono_img = SSIM(pred_mono, gt_mono, multichannel=True)
            psnr_rgb_img = PSNR(pred_rgb, gt_rgb)
            ssim_rgb_img = SSIM(pred_rgb, gt_rgb, multichannel=True)

            # imageio.imwrite(os.path.join(save_images, '{}_{}_gt.jpg'.format(image_num, iteration)), gt_rgb)
            # imageio.imwrite(os.path.join(save_images,'{}_{}_psnr_{:.4f}_ssim_{:.4f}.jpg'.format(image_num, iteration, psnr_rgb_img,ssim_rgb_img)), pred_rgb)
            # imageio.imwrite(os.path.join(save_images,'{}_{}_psnr_{:.4f}_ssim_{:.4f}_mono.jpg'.format(image_num, iteration, psnr_rgb_img,ssim_rgb_img)), pred_mono)

            psnr1.append(psnr_mono_img)
            ssim1.append(ssim_mono_img)
            psnr2.append(psnr_rgb_img)
            ssim2.append(ssim_rgb_img)

    np.savetxt(os.path.join(save_csv_file, 'Metrics_iter_{}.csv'.format(iteration)),
               [p for p in zip(psnr1, ssim1, psnr2, ssim2)], delimiter=',', fmt='%s')

    psnr1_avg = sum(psnr1[1:]) / len(psnr1[1:])
    ssim1_avg = sum(ssim1[1:]) / len(ssim1[1:])
    psnr2_avg = sum(psnr2[1:]) / len(psnr2[1:])
    ssim2_avg = sum(ssim2[1:]) / len(ssim2[1:])

    f = open(metric_average_filename,  'a')
    f.write('-- psnr1_avg:{}, ssim1_avg:{}, psnr2_avg:{}, ssim2_avg:{},, iter:{}\n'.format(psnr1_avg, ssim1_avg, psnr2_avg, ssim2_avg, iteration))
    print('metric average printed.')
    f.close()

    return

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    opt={'base_lr':1e-5}
    opt['batch_size'] = 24
    opt['iterations'] = 500001

    metric_average_file = 'result_on_SID/metric_average.txt'
    # These are folders
    save_weights_file = 'result_on_SID/weights'
    save_images_file = 'result_on_SID/images'
    save_csv_file = 'result_on_SID/csv_files'

    if not os.path.exists(save_weights_file):
        os.makedirs(save_weights_file)
    if not os.path.exists(save_images_file):
        os.makedirs(save_images_file)
    if not os.path.exists(save_csv_file):
        os.makedirs(save_csv_file)

    train_input_paths = glob.glob('./Sony/short/0*_00_0.1s.ARW') + glob.glob('./Sony/short/2*_00_0.1s.ARW')
    train_gt_paths = []
    for x in train_input_paths:
        train_gt_paths += glob.glob('./Sony/long/*' + x[-17:-12] + '*.ARW')

    test_input_paths = glob.glob('./Sony/short/1*_00_0.1s.ARW')
    test_gt_paths = []
    for x in test_input_paths:
        test_gt_paths += glob.glob('./Sony/long/*' + x[-17:-12] + '*.ARW')


    print('train data: %d pairs'%len(train_input_paths))
    print('test data: %d pairs'%len(test_input_paths))

    dataloader_train = DataLoader(load_data(train_input_paths, train_gt_paths, training=True),
                                  batch_size=opt['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    # dataloader_test = DataLoader(load_data(test_input_paths, test_gt_paths,training=False),
    #                              batch_size=1, shuffle=False, num_workers=0, pin_memory=True)


    device = torch.device("cuda")
    model = our_Net()
    # print(model)
    # checkpoint = torch.load(save_weights_file + '/weights_160000.pth')
    # model.load_state_dict(checkpoint['model'], strict=False)
    print('\nTrainable parameters : {}\n'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    print('\nTotal parameters : {}\n'.format(sum(p.numel() for p in model.parameters())))
    model = model.to(device)
    print('Device on cuda: {}'.format(next(model.parameters()).is_cuda))

    iter_num = 0
    l1_loss1 = torch.nn.L1Loss()
    l1_loss2 = torch.nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt['base_lr'])
    optimizer.zero_grad()
    loss_list = ['loss_mono, loss_rgb, loss_total']
    metrics = ['PSNR_mono, PSNR_rgb, SSIM_mono, SSIM_rgb']
    iter_list = ['Iteration']
    iter_LR = ['Iter_LR']

    while iter_num < opt['iterations']:
        for _, img in tqdm.tqdm(enumerate(dataloader_train)):
            input_raw = img[0].to(device)
            gt_rgb = img[1].to(device)
            gt_mono = 0.2989 * gt_rgb[:,0,:,:] + 0.5870 * gt_rgb[:,1,:,:] + 0.1140 * gt_rgb[:,2,:,:]
            gt_mono = gt_mono.unsqueeze(1)

            model.train()
            pred_mono,  pred_rgb = model(input_raw)
            iter_num += 1
            loss1 = l1_loss1(pred_mono, gt_mono)
            loss2 = l1_loss2(pred_rgb, gt_rgb)

            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if iter_num > opt['iterations']:
                break

            if iter_num % 100 == 0:
                psnr_mono = PSNR(gt_mono.detach().cpu().numpy().transpose(0, 2, 3, 1),
                             np.clip(pred_mono.detach().cpu().numpy().transpose(0, 2, 3, 1), 0, 1))
                psnr_rgb = PSNR(gt_rgb.detach().cpu().numpy().transpose(0, 2, 3, 1),
                             np.clip(pred_rgb.detach().cpu().numpy().transpose(0, 2, 3, 1), 0, 1))
                ssim_mono = SSIM(gt_mono.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255,
                                 np.clip(pred_mono.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255, 0, 255), multichannel=True)
                ssim_rgb = SSIM(gt_rgb.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255,
                                np.clip(pred_rgb.detach().cpu().numpy().transpose(0, 2, 3, 1) * 255, 0, 255), multichannel=True)

                print('\niter_num:%.0f, loss_mono:%.4f, loss_rgb:%.4f, PSNR_mono:%.4f, PSNR_rgb:%.4f, SSIM_mono:%.4f, SSIM_rgb:%.4f' % (
                iter_num, np.mean(loss1.detach().cpu().numpy()), np.mean(loss2.detach().cpu().numpy()), np.mean(psnr_mono), np.mean(psnr_rgb), np.mean(ssim_mono), np.mean(ssim_rgb)))

                loss_list.append('{:.5f},{:.5f},{:.5f}'.format(loss1.item(), loss2.item(), loss1.item()+loss2.item()))
                metrics.append('{:.5f},{:.5f},{:.5f},{:.5f}'.format(np.mean(psnr_mono), np.mean(psnr_rgb), np.mean(ssim_mono), np.mean(ssim_rgb)))
                iter_list.append(iter_num)
                iter_LR.append(optimizer.param_groups[0]['lr'])
                np.savetxt(os.path.join(save_csv_file, 'train_curve.csv'), [p for p in zip(iter_list, iter_LR, loss_list, metrics)], delimiter=',', fmt='%s')

            # save checkpoint every 20000 times, make adjustments accordingly
            if iter_num % 20000 == 0:
                torch.save({'model': model.state_dict()}, os.path.join(save_weights_file, 'weights_{}.pth'.format(iter_num)))
                print('model saved......')

            # if iter_num % 40000 == 0:
            #     run_test(model, dataloader_test, iter_num, save_images_file, save_csv_file, metric_average_file)
