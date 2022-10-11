import torch.nn as nn
import torch

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes=2, matcher='hungarian', weight_dict=0.001, losses=0.0, focal_alpha=0.25, mask_out_stride=4, num_frames=1):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        # self.num_classes = num_classes
        # self.matcher = matcher
        # self.weight_dict = weight_dict
        # self.losses = losses
        # self.focal_alpha = focal_alpha
        # self.mask_out_stride = mask_out_stride
        # self.num_frames = num_frames
        # self.valid_ratios = None

        ######  Loss for GAN Idea   #######
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.gan_loss_fn = lambda y_hat, y: self.loss_fn(y_hat, torch.ones_like(y_hat) * y)
        self.l1_loss_fn = nn.L1Loss()
        self.lambda_d = 0.5
        self.lambda_l1 = 100.0
        ######  Loss for expi 1   #######
        #self.lambda_g_adjust = 0.1
        #self.lambda_l1_adust = 0.2
        ######  Loss for expi 2   #######
        self.lambda_g_adjust = 0.1
        self.lambda_l1_adust = 0.2


    def forward(self, fake_ytvis, real_ytvis):
        """
            Write GAN loss logic here
        """

        ############# ytvis #############
        loss_d_real = self.gan_loss_fn(real_ytvis, 1) * self.lambda_d
        loss_d_fake = self.gan_loss_fn(fake_ytvis, 0) * self.lambda_d
        loss_d_ytvis = loss_d_real + loss_d_fake

        loss_g_gan = self.gan_loss_fn(fake_ytvis, 1)
        loss_g_l1 = self.l1_loss_fn(fake_ytvis, real_ytvis) * self.lambda_l1
        #print(" self.lambda_g_adjust is ", self.lambda_g_adjust)
        #print(" self.lambda_l1_adust is ", self.lambda_l1_adust)
        loss_g_ytvis = self.lambda_g_adjust*loss_g_gan + self.lambda_l1_adust*loss_g_l1
        
        
        #print("loss_g_gan should go to 0 and is :", loss_g_gan)
        #print("loss_g_l1 should go to 0.5 and is :", loss_g_l1)

        #loss_g_l1 should go to 0.5
        # loss_g_gan should go to 0
        ############# coco #############
       # loss_d_real = self.gan_loss_fn(real_coco, 1) * self.lambda_d
       # loss_d_fake = self.gan_loss_fn(fake_coco, 0) * self.lambda_d
       # loss_d_coco = loss_d_real + loss_d_fake

       # loss_g_gan = self.gan_loss_fn(fake_coco, 1)
       # loss_g_l1 = self.l1_loss_fn(fake_coco, real_coco) * self.lambda_l1
       # loss_g_coco = loss_g_gan + loss_g_l1

        #loss_d = loss_d_ytvis + loss_d_coco
        #loss_g = loss_g_ytvis + loss_g_coco

       # l_dict = {'loss_g':loss_g, 'loss_d':loss_d}

        #loss_dict.update(l_dict)
        ####### return separate loss_d to avoid it from mixing with other loss in engine train_epoch 
        return loss_g_ytvis