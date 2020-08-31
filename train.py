import datetime
import os
import sys
import time
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

# from options.train_options import TrainOptions
sys.path.append(os.path.join(os.path.dirname(__file__), "options"))
from train_options import TrainOptions

sys.path.append(os.path.join(os.path.dirname(__file__), "dataloader"))
from Sense2StopSync_loader import S2S_dataset
from S2S_settings import settings as S2S_settings
# from ECM_loader import ECM_dataset

sys.path.append(os.path.join(os.path.dirname(__file__), "models"))
# import loss_functions
# import GAN

sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
from utils import load_checkpoint
# from visualizer import Visualizer


writer = SummaryWriter(comment=str(datetime.datetime.now()))

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

cudnn.benchmark = True


if __name__ == '__main__':
    opt = TrainOptions().parse()
    print(opt)

    # train_dataset = S2S_dataset(settings=S2S_settings, usage="train")
    # val_dataset = S2S_dataset(settings=S2S_settings, usage="val")
    # test_dataset = S2S_dataset(settings=S2S_settings, usage="test")

    # # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # dataset_size = len(train_dataset)    # get the size of training dataset
    # print('The size of training dataset = %d' % dataset_size)
    
    # # model = GAN.cGAN(opt)
    # model = create_model(opt)      # create a model given opt.model and other options
    # model.setup(opt)               # regular setup: load and print networks; create schedulers
    # visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    # total_step_cnt = 0             # the total number of training step counts (iterations)

    # if opt.resume:
    # 	# could be moved to Model class
    #     model, total_step_cnt, opt.start_epoch = load_checkpoint(opt.resume_path, model)

    # for epoch in range(opt.start_epoch, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    #     epoch_start_time = time.time()  # timer for entire epoch
    #     iter_data_time = time.time()    # timer for data loading per iteration
    #     epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
    #     visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
    #     # model.update_learning_rate()    # update learning rates in the beginning of every epoch. UNSURE HOW EPOCH IS PASSED IN

    #     for i, data in enumerate(train_dataset):
    #         iter_start_time = time.time()
    #         if total_iters % opt.print_freq == 0:
    #             t_data = iter_start_time - iter_data_time

    #         total_step_cnt += opt.batch_size
    #         epoch_iter += opt.batch_size
    #         model.set_input(data)         # unpack data from dataset and apply preprocessing
    #         model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
    #         model.TfWriter(writer, total_step_cnt)

    #         if total_step_cnt % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
    #         	save_result = total_iters % opt.update_html_freq == 0
    #             model.get_visual_path()
    #             visualizer.display_current_results(model.get_current_visuals(), save_result)

    #         if total_step_cnt % opt.print_freq == 0:    # print training losses and save logging information to the disk
    #             losses = model.get_current_losses()
    #             t_comp = (time.time() - iter_start_time) / opt.batch_size
    #             visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
    #             if opt.display_id > 0:
    #                 visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

    #         if total_step_cnt % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
    #             print(opt.name + 'saving the latest model (epoch %d, total_step_cnt %d)' %
    #                   (epoch, total_step_cnt))
    #             save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
    #             model.save_networks(save_suffix)
    #             # torch.save({
	   #             #      'step': total_step_cnt,
	   #             #      'epoch': epoch,
	   #             #      'netD': model.netD.state_dict(),
	   #             #      'netD_mul': model.netD_mul.state_dict(),
	   #             #      'optimizer_D': model.optimizer_D.state_dict(),
	   #             #      'optimizer_G': model.optimizer_G.state_dict(),
	   #             #      'model_fusion': model.model_fusion.state_dict(),
    #             # 	}, 
    #             # 	path = os.path.join(opt.checkpoints_dir, str(epoch) + '_' + opt.name + '_checkpoint.pth.tar')
    #             # )

    #         iter_data_time = time.time()

    #         if epoch_iter % opt.eval_freq == 0:
    #             evaluation(val_dataset, model, total_step_cnt, writer=writer)

    #     evaluation(test_dataset, model, total_step_cnt, writer=writer)

    #     print(opt.name + ' End of epoch %d / %d \t Time Taken: %d sec' %
    #           (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

    #     if epoch > opt.n_epochs:
    #         model.update_learning_rate()

