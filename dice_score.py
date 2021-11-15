from PIL import Image
import numpy as np

def compute_dice(pred_img, label_img):
    p = pred_img.astype(np.float32)
    l = label_img.astype(np.float32)
    if p.max() > 127.5:
        p /= 255.
    if l.max() > 127.5:
        l /= 255.

    p = np.clip(p, 0, 1.0)
    l = np.clip(l, 0, 1.0)
    p[p > 0.5] = 1.0
    p[p < 0.5] = 0.0
    l[l > 0.5] = 1.0
    l[l < 0.5] = 0.0
    product = np.dot(l.flatten(), p.flatten())
    dice_num = 2 * product + 1
    pred_sum = p.sum()
    label_sum = l.sum()
    dice_den = pred_sum + label_sum + 1
    dice_val = dice_num / dice_den
    return dice_val

if __name__ == '__main__':
    input_path = 'experiments/train_l1_big_real_fix_200/imgs/fixed_samples-100-12100_train.png'
    target_path = 'fixed_set/t_fixed_target.png'
    a = Image.open(input_path).convert('L')
    a = np.asarray(a)

    b = Image.open(target_path).convert('L')
    b = np.asarray(b)

    score = compute_dice(a, b)
    print(score)