import argparse

def HyperParameter():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=50, help="size of the batches")

    parser.add_argument("--n_dat", type=int, default=5000, help="number of used data")
    parser.add_argument("--n_val", type=int, default=50, help="number of test data")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")

    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")

    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--label_dim", type=int, default=4, help="dimensionality of the label space")


    parser.add_argument("--sp_size", type=int, default=500, help="size of each spectra length")

    parser.add_argument("--n_pars", type=int, default=5, help="number of each parameters sample")
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")

    parser.add_argument("--clip_value", type=float, default=0.1, help="lower and upper clip value for disc. weights")
    parser.add_argument("--sample_interval", type=int, default=5, help="interval between spec sampling")
    parser.add_argument("--epoch_interval", type=int, default=10, help="interval between epoch ")
    parser.add_argument("--epoch_sample", type=int, default=50, help="Random sample for epoch interval")
    parser.add_argument("--lambda_l1", type=float, default=10, help="Balance the adversial loss and l1 loss")
    parser.add_argument("--lambda_phy", type=float, default=10, help="Physcial loss weight")
    return parser.parse_args()