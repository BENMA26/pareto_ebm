import os
import pickle
import pytorch_lightning as pl

'''
callback for model replay buffer
'''
class SaveReplayBufferCallback(pl.Callback):
    def __init__(self):
        super().__init__()
    def on_train_epoch_end(self, trainer, pl_module):
        with open(os.path.join(pl_module.args.save_dir,f"{pl_module.current_epoch}.pkl"),"wb") as file:
            pickle.dump(pl_module.sampler,file)

'''
callback for restart training when model diverge
'''
class RestartTrainingCallback(pl.Callback):
    def __init__(self):
        super().__init__()

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.callback_metrics['loss'].abs().item() > 1e8:
            trainer.should_stop = True

'''
callback for unconditional generation 
'''
class GenerateCallback(pl.Callback):

    def __init__(self, batch_size=8, vis_steps=8, num_steps=256, every_n_epochs=5):
        super().__init__()
        self.batch_size = batch_size         
        self.vis_steps = vis_steps           
        self.num_steps = num_steps           
        self.every_n_epochs = every_n_epochs 

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            
            imgs_per_step = self.generate_imgs(pl_module)
            for i in range(imgs_per_step.shape[1]):
                step_size = self.num_steps // self.vis_steps
                imgs_to_plot = imgs_per_step[step_size-1::step_size,i]
                grid = torchvision.utils.make_grid(imgs_to_plot.clamp_(min=-1.0, max=1.0), nrow=imgs_to_plot.shape[0], normalize=True)
                trainer.logger.experiment.add_image("generation_{}".format(i), grid, global_step=trainer.current_epoch)

    def generate_imgs(self, pl_module):
        pl_module.eval()
        start_imgs = torch.rand((self.batch_size,) + pl_module.hparams["img_shape"]).to(pl_module.device)
        start_imgs = start_imgs * 2 - 1
        torch.set_grad_enabled(True) 
        imgs_per_step = pl_module.generate_samples(start_imgs, steps=self.num_steps, step_size=1, return_img_per_step=True)
        torch.set_grad_enabled(False)
        pl_module.train()
        return imgs_per_step

'''
callback for multi conditional generation
'''
class ConditionalGenerateCallback(pl.Callback):

    def __init__(self, batch_size=1, vis_steps=8, num_steps=256, every_n_epochs=5):
        super().__init__()
        self.batch_size = batch_size         # Number of images to generate
        self.vis_steps = vis_steps           # Number of steps within generation to visualize
        self.num_steps = num_steps           # Number of steps to take during generation
        self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            for i in range(10):
                imgs_per_step = self.conditional_generate_imgs(pl_module, i)

                for j in range(imgs_per_step.shape[1]):
                    step_size = self.num_steps // self.vis_steps
                    imgs_to_plot = imgs_per_step[step_size-1::step_size,j]
                    grid = torchvision.utils.make_grid(imgs_to_plot.clamp_(min=-1.0, max=1.0), nrow=imgs_to_plot.shape[0], normalize=True)
                    trainer.logger.experiment.add_image("conditional_generation_{}".format(i), grid, global_step=trainer.current_epoch)

    def conditional_generate_imgs(self, pl_module, conditional_index):
        pl_module.eval()
        start_imgs = torch.rand((self.batch_size,) + pl_module.hparams["img_shape"]).to(pl_module.device)
        start_imgs = start_imgs * 2 - 1
        torch.set_grad_enabled(True)  # Tracking gradients for sampling necessary
        imgs_per_step = pl_module.generate_conditional_samples(start_imgs, steps=self.num_steps, step_size=1, conditional_index = conditional_index ,return_img_per_step=True)
        torch.set_grad_enabled(False)
        pl_module.train()
        return imgs_per_step

'''
callback for show images in replay buffer
'''
class SamplerCallback(pl.Callback):

    def __init__(self, num_imgs=32, every_n_epochs=5):
        super().__init__()
        self.num_imgs = num_imgs            
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            try:
                exmp_imgs = torch.stack(random.choices(pl_module.examples, k=self.num_imgs), dim=0)

                exmp_imgs = exmp_imgs.to(pl_module.device)
                grid = torchvision.utils.make_grid(exmp_imgs.clamp_(min=-1.0, max=1.0), nrow=4, normalize=True)
                trainer.logger.experiment.add_image("sampler", grid, global_step=trainer.current_epoch)
            except Exception as e:
                print("Error in SamplerCallback: {}".format(e))
                pass

'''
callback for show images i
'''
class OutlierCallback(pl.Callback):

    def __init__(self, batch_size=1024):
        super().__init__()
        self.batch_size = batch_size

    def on_train_epoch_end(self, trainer, pl_module):
        with torch.no_grad():
            pl_module.eval()
            rand_imgs = torch.rand((self.batch_size,) + pl_module.hparams["img_shape"]).to(pl_module.device)
            rand_imgs = rand_imgs * 2 - 1.0
            rand_out = pl_module.cnn(rand_imgs)[0].mean()
            pl_module.train()

        trainer.logger.experiment.add_scalar("rand_out", rand_out, global_step=trainer.current_epoch)