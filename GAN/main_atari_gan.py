from atari_gan import *
import ale_py
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cpu", help="Device name, default=cpu")
    args = parser.parse_args()


    #print(args)
    device = torch.device("cuda" if torch.cuda.is_available() else args.dev)
    envs = [
        InputWrapper(gym.make(name))
        for name in ('ALE/Breakout-v5', 'ALE/AirRaid-v5', 'ALE/Pong-v5')
    ]
    shape = envs[0].observation_space.shape

    #print(f"Device: {device}")
    #print(f"shape of observations space: {shape}")

    net_discr = Discriminator(input_shape=shape).to(device)
    net_gener = Generator(output_shape=shape).to(device)

    objective = nn.BCELoss()
    gen_optimizer = optim.Adam(params=net_gener.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    dis_optimizer = optim.Adam(params=net_discr.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    writer = SummaryWriter()

    gen_losses = []
    dis_losses = []
    iter_no = 0

    true_labels_v = torch.ones(BATCH_SIZE, device=device)
    fake_labels_v = torch.zeros(BATCH_SIZE, device=device)
    ts_start = time.time()

    for batch_v in iterate_batches(envs):
        # fake samples, input is 4D: batch, filters, x, y
        gen_input_v = torch.FloatTensor(BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1)
        gen_input_v.normal_(0,1)
        gen_input_v = gen_input_v.to(device)
        batch_v = batch_v.to(device)
        gen_output_v = net_gener(gen_input_v)

        # train discriminator
        dis_optimizer.zero_grad()
        dis_output_true_v = net_discr(batch_v)
        dis_output_fake_v = net_discr(gen_output_v.detach())
        dis_loss = objective(dis_output_true_v, true_labels_v) + \
                    objective(dis_output_fake_v, fake_labels_v)
        dis_loss.backward()
        dis_optimizer.step()
        dis_losses.append(dis_loss.item())

        # train the generator
        gen_optimizer.zero_grad()
        dis_output_v = net_discr(gen_output_v)
        gen_loss_v = objective(dis_output_v, true_labels_v)
        gen_loss_v.backward()
        gen_optimizer.step()
        gen_losses.append(gen_loss_v.item())

        # TensorBoard
        iter_no += 1
        if iter_no % REPORT_EVERY_ITER == 0:
            dt = time.time() - ts_start
            log.info("Iter %dd in %.2fs: gen_loss=%.3e, dis_loss=%.3e",
                     iter_no, dt, np.mean(gen_losses), np.mean(dis_losses))
            ts_start = time.time()
            writer.add_scalar("gen_loss", np.mean(gen_losses), iter_no)
            writer.add_scalar("dis_loss", np.mean(dis_losses), iter_no)
            gen_losses = []
            dis_losses = []
        if iter_no % SAVE_IMAGE_EVERY_ITER == 0:
            img = vutils.make_grid(batch_v.data[:64], normalize=True)
            writer.add_image("fake", img, iter_no)
            img = vutils.make_grid(batch_v.data[:64], normalize=True)
            writer.add_image("real", img, iter_no)

        if iter_no % 1000 == 0:
            print(f"🔥 Iteración {iter_no} alcanzada. ¡Vamos bien!", end="", flush=True)