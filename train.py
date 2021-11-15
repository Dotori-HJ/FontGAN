import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image, make_grid

from dataset import DataProvider
from torch.utils.data import DataLoader
from model import Encoder, Decoder, Discriminator
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dice_score import compute_dice


class Trainer:
    def __init__(self, data_dir, fixed_dir, fonts_num, batch_size, name, img_size=128, gan=False, ratio=1.0):
        self.data_dir = data_dir
        self.fixed_dir = fixed_dir
        self.fonts_num = fonts_num
        self.batch_size = batch_size
        self.img_size = img_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log_dir = os.path.join('experiments', name)
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.gan = gan
        self.ratio = ratio

        self.encoder = Encoder(1, 32).to(self.device)
        self.decoder = Decoder(fonts_num + 1, 1, 32).to(self.device)
        if gan:
            self.discriminator = Discriminator(2, 32).to(self.device)

        self.t_fixed_source = torch.load(os.path.join(fixed_dir, 't_fixed_source.pkl'), map_location=self.device)
        self.t_fixed_target = torch.load(os.path.join(fixed_dir, 't_fixed_target.pkl'), map_location=self.device)
        self.t_fixed_label = torch.LongTensor(torch.load(os.path.join(fixed_dir, 't_fixed_label.pkl'))).to(self.device)
        self.t_fixed_label[self.t_fixed_label == -1] = fonts_num

        self.fixed_source = torch.load(os.path.join(fixed_dir, 'fixed_source.pkl'), map_location=self.device)
        self.fixed_target = torch.load(os.path.join(fixed_dir, 'fixed_target.pkl'), map_location=self.device)
        self.fixed_label = torch.LongTensor(torch.load(os.path.join(fixed_dir, 'fixed_label.pkl'))).to(self.device)
        self.fixed_label[self.fixed_label == -1] = fonts_num

        self.train_provider = DataProvider(self.data_dir, obj_name='train.obj', augment=False)
        self.val_provider = DataProvider(self.data_dir, obj_name='val.obj', augment=False)
        
        self.train_loader = DataLoader(self.train_provider, batch_size=batch_size, shuffle=True, num_workers=8)
        self.val_loader = DataLoader(self.val_provider, batch_size=batch_size, shuffle=False, num_workers=8)

    def train(self, max_epoch, save_path, to_model_path, sample_step=100, restore=None, from_model_path=False, save_nrow=8, model_save_step=None):
        if restore:
            encoder_path, decoder_path = restore
            self.encoder.load_state_dict(torch.load(encoder_path))
            self.decoder.load_state_dict(torch.load(decoder_path))

        criterion = nn.L1Loss()
        # criterion = nn.MSELoss()
        
        update_parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = optim.Adam(update_parameters, lr=1e-4, betas=(0.9, 0.999))
        if self.gan:
            optimizer_D = optim.Adam(self.discriminator.parameters(), lr=1e-4, betas=(0.9, 0.999))

        iter = 0

        for epoch in range(max_epoch):
            self.encoder.train()
            self.decoder.train()
            pbar = tqdm(self.train_loader)
            for i, batch in enumerate(pbar):
                optimizer.zero_grad()
                font_ids, batch_images = batch
                font_ids[font_ids == -1] = self.fonts_num
                
                embedding_ids = font_ids.to(self.device)
                batch_images = batch_images.to(self.device)

                real_target = batch_images[:, [0], :, :]
                real_source = batch_images[:, [1], :, :]

                encoded = self.encoder(real_source)
                decoded = self.decoder(encoded, embedding_ids)
                
                if self.gan == 'onlygan':
                    fake_logit = self.discriminator(torch.cat((real_source, decoded), dim=1))

                    loss_G = F.softplus(-fake_logit).mean()
                    loss = loss_G
                elif self.gan:
                    fake_logit = self.discriminator(torch.cat((real_source, decoded), dim=1))

                    loss_G = F.softplus(-fake_logit).mean()
                    loss_L1 = criterion(decoded, real_target)
                    loss = loss_L1 + self.ratio * loss_G
                else:
                    loss_L1 = criterion(decoded, real_target)
                    loss = loss_L1
                loss.backward()

                optimizer.step()
                
                if self.gan:
                    optimizer_D.zero_grad()
                    
                    fake_logit = self.discriminator(torch.cat((real_source, decoded.detach()), dim=1))
                    real_logit = self.discriminator(torch.cat((real_source, real_target), dim=1))
                    
                    loss_D = F.softplus(fake_logit).mean() + F.softplus(-real_logit).mean()
                    
                    real_input = torch.cat((real_source, real_target), dim=1).requires_grad_(True)
                    real_logit = self.discriminator(real_input)
                    real_grads = torch.autograd.grad(outputs=real_logit, inputs=real_input,
                                                    grad_outputs=torch.ones(real_logit.size()).to(real_input.device),
                                                    create_graph=True, retain_graph=True)[0].view(real_input.size(0), -1)
                    r1_penalty = torch.mul(real_grads, real_grads).mean()
                    
                    loss_D = loss_D + 10 * r1_penalty
                    loss_D.backward()
                    
                    optimizer_D.step()

                pbar.set_description(f'Train {epoch+1}/{max_epoch}, loss={loss.item():.4f}')
                self.writer.add_scalar('loss/total', loss.item(), global_step=iter)
                if self.gan != 'onlygan':
                    self.writer.add_scalar('loss/L1', loss_L1.item(), global_step=iter)
                if self.gan:
                    self.writer.add_scalar('loss/G', loss_G.item(), global_step=iter)
                    self.writer.add_scalar('loss/D', loss_D.item(), global_step=iter)

                # save image
                if (iter + 1) % sample_step == 0:
                    save_image(
                        (real_target + 1.) * 0.5,
                        os.path.join(save_path, f'real_samples-{epoch+1}-{iter+1}_train.png'),
                        nrow=save_nrow,
                        pad_value=1.0,
                        normalize=False,
                    )
                    save_image(
                        (decoded + 1.) * 0.5,
                        os.path.join(save_path, f'fake_samples-{epoch+1}-{iter+1}_train.png'),
                        nrow=save_nrow,
                        pad_value=1.0,
                        normalize=False,
                    )
                    encoded = self.encoder(self.t_fixed_source)
                    decoded = self.decoder(encoded, self.t_fixed_label)

                    save_image(
                        (decoded + 1.) * 0.5,
                        os.path.join(save_path, f'fixed_samples-{epoch+1}-{iter+1}_train.png'),
                        nrow=save_nrow,
                        pad_value=1.0,
                        normalize=False,
                    )
                iter += 1
            # ---------------- #
            #    Validation    #
            # ---------------- #
            self.encoder.eval()
            self.decoder.eval()
            val_losses = []
            val_scores = []
            pbar = tqdm(self.val_loader)
            
            for i, batch in enumerate(pbar):
                font_ids, batch_images = batch
                font_ids[font_ids == -1] = self.fonts_num
                # embedding_ids = torch.LongTensor(font_ids).to(self.device)
                embedding_ids = font_ids.to(self.device)
                batch_images = batch_images.to(self.device)

                # target / source images
                real_target = batch_images[:, [0], :, :]
                real_source = batch_images[:, [1], :, :]
                
                with torch.no_grad():
                    encoded = self.encoder(real_source)
                    decoded = self.decoder(encoded, embedding_ids)
                
                val_image = make_grid(
                    ((real_target + 1.) * 0.5).cpu(),
                    nrow=save_nrow,
                    pad_value=1.0,
                    normalize=False,
                ).numpy()
                
                pred_image = make_grid(
                    ((decoded + 1.) * 0.5).cpu(),
                    nrow=save_nrow,
                    pad_value=1.0,
                    normalize=False,
                ).numpy()
                
                score = compute_dice(val_image, pred_image)
                val_scores.append(score)

                loss = criterion(decoded, real_target)
                val_losses.append(loss.item())

                pbar.set_description(f'Val, dice score={score}, l1={loss.item():.4f}')

                # save image
                if i == 0:
                    encoded = self.encoder(self.fixed_source)
                    decoded = self.decoder(encoded, self.fixed_label)
                    save_image(
                        (decoded + 1.) * 0.5,
                        os.path.join(save_path, f'fake_samples-{epoch+1}-{i+1}_val.png'),
                        nrow=save_nrow,
                        pad_value=1.0,
                        normalize=False,
                    )
            val_scores = np.array(val_scores)
            val_losses = np.array(val_losses)
            self.writer.add_scalar('validation/dice', np.mean(val_scores), global_step=iter)
            self.writer.add_scalar('validation/l1', np.mean(val_losses), global_step=iter)

            if not model_save_step:
                model_save_step = 5

            if (epoch + 1) % model_save_step == 0:
                torch.save(
                    self.encoder.state_dict(),
                    os.path.join(
                        to_model_path,
                        f'{epoch+1}-Encoder.pkl',
                    )
                )
                torch.save(
                    self.decoder.state_dict(),
                    os.path.join(
                        to_model_path,
                        f'{epoch+1}-Decoder.pkl',
                    )
                )
                if self.gan:
                    torch.save(
                        self.discriminator.state_dict(),
                        os.path.join(
                            to_model_path,
                            f'{epoch+1}-Discriminator.pkl',
                        )
                    )


        # save model
        total_epoch = int(max_epoch)
        torch.save(
            self.encoder.state_dict(),
            os.path.join(
                to_model_path,
                f'{total_epoch}-Encoder.pkl',
            )
        )
        torch.save(
            self.decoder.state_dict(),
            os.path.join(
                to_model_path,
                f'{total_epoch}-Decoder.pkl',
            )
        )
        if self.gan:
            torch.save(
                self.discriminator.state_dict(),
                os.path.join(
                    to_model_path,
                    f'{total_epoch}-Discriminator.pkl',
                )
            )

name = 'train'
save_path = os.path.join('experiments', name, 'imgs')
to_model_path = os.path.join('experiments', name, 'save')
data_path = 'dataset'
fixed_dir = 'fixed_set'
restore = None

os.makedirs(save_path, exist_ok=True)
os.makedirs(to_model_path, exist_ok=True)

if __name__ == '__main__':
    a = Trainer(data_dir=data_path, fixed_dir=fixed_dir, fonts_num=7, batch_size=64, img_size=128, name=name, gan='onlygan', ratio=0)
    a.train(max_epoch=200, save_path=save_path, to_model_path=to_model_path, restore=restore)