import os
import argparse

# the directory that options.py resides in
file_dir = os.path.dirname(__file__)


class Options:
    def __init__(self):
        # TODO: Write the arguments
        self.parser = argparse.ArgumentParser(description="Retouch options")

        self.parser.add_argument('--config', help="configuration file *.yml", type=str, required=False, default='config.yml')
        self.parser.add_argument('--seed', help="configuration file *.yml", type=int, required=False, default=1)
        self.parser.add_argument('--init_data', help="configuration file *.yml", type=bool, required=False, default=True)

        # PATHS
        self.parser.add_argument("--base_dir", type=str, help="path to the training data", default=os.path.join("datasets/retouch-dataset/pre_processed"))
        self.parser.add_argument("--list_dir", type=str, help="path to the split", default=os.path.join(file_dir, "splits", "split_cirrus_balanced"))
        self.parser.add_argument("--pretrained_dir", type=str, help="path to the split", default=os.path.join(file_dir, "../pretrained"))
        self.parser.add_argument("--log_dir", type=str, help="log directory", default=os.path.join(file_dir, "log"))
        self.parser.add_argument("--model_name", type=str, help="name of the model", default='Blackbox')
        self.parser.add_argument("--data_mode", type=str, help="data mode", default='mnist')

        # Teacher Parameters
        self.parser.add_argument("--teaching_mode", type=str, help="name of the teaching mode", default='omniscient')
        self.parser.add_argument("--same_feat_space", type=bool, help="name of the teaching mode", default=True)
        self.parser.add_argument("--class_1", type=int, help="name of the teaching mode", default=3)
        self.parser.add_argument("--class_2", type=int, help="name of the teaching mode", default=7)
        self.parser.add_argument("--nb_train", type=int, help="name of the teaching mode", default=5000)
        self.parser.add_argument("--nb_test", type=int, help="name of the teaching mode", default=1000)

        self.parser.add_argument("--n_teacher_runs", type=int, help="name of the teaching mode", default=300)
        self.parser.add_argument("--dim", type=int, help="name of the teaching mode", default=784)
        self.parser.add_argument("--eta", type=float, help="name of the teaching mode", default=2e-3)
        self.parser.add_argument("--n_unroll", type=int, help="name of the teaching mode", default=1000)
        self.parser.add_argument("--n_unroll_blocks", type=int, help="name of the teaching mode", default=40)

        # conditional GAN
        self.parser.add_argument("--n_epochs", type=int, help="name of the teaching mode", default=40)
        self.parser.add_argument("--save_frequency", type=int, help="name of the teaching mode", default=2)
        self.parser.add_argument("--start_saving", type=int, help="name of the teaching mode", default=16)
        self.parser.add_argument("--batch", type=int, help="name of the teaching mode", default=64)
        self.parser.add_argument("--lr", type=float, help="name of the teaching mode", default=0.0002)
        self.parser.add_argument("--b1", type=float, help="name of the teaching mode", default=0.5)
        self.parser.add_argument("--b2", type=float, help="name of the teaching mode", default=0.999)
        self.parser.add_argument("--latent_dim", type=int, help="name of the teaching mode", default=100)
        self.parser.add_argument("--n_classes", type=int, help="name of the teaching mode", default=2)
        self.parser.add_argument("--img_size", type=int, help="name of the teaching mode", default=28)
        self.parser.add_argument("--channels", type=int, help="name of the teaching mode", default=1)
        self.parser.add_argument("--sample_interval", type=int, help="name of the teaching mode", default=400)

        # Optimization
        self.parser.add_argument("--use_augmentation", type=bool, help="use data augmentation", default=True)
        self.parser.add_argument("--learning_rate", type=float, help="learning rate", default=1e-3)
        self.parser.add_argument("--n_iter", type=int, help="number of iterations", default=1200)
        self.parser.add_argument("--batch_size", type=int, help="batch size", default=32)
        self.parser.add_argument("--lr_factor", type=int, help="batch size", default=10000)
        self.parser.add_argument("--scheduler_step_size", type=int, help="scheduler step size for lr decreasing", default=15)
        self.parser.add_argument("--gd_n", type=int, help="scheduler step size for lr decreasing", default=200)
        self.parser.add_argument("--optim", type=str, help="scheduler step size for lr decreasing", default='Adam')


        # System
        self.parser.add_argument("--no_cuda", help="if set disables CUDA", action="store_true")
        self.parser.add_argument("--num_workers", type=int, help="number of dataloader workers", default=10)
        self.parser.add_argument("--log_frequency", type=int, help="number of batches between each tensorboard log", default=20)

        self.parser.add_argument("--train_student", type=bool, help="number of batches between each tensorboard log", default=False)
        self.parser.add_argument("--train_baseline", type=bool, help="number of batches between each tensorboard log", default=False)
        self.parser.add_argument("--train_sgd", type=bool, help="number of batches between each tensorboard log", default=False)

    def parse(self):
        # self.options = self.parser.parse_args()
        # return self.options
        return self.parser
