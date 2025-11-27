import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import random
import os
from get_ManySig_CR_unequal import Domain_Dataset
from fe import *
from cls import *
from torch.utils.tensorboard import SummaryWriter
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Config:
    def __init__(
        self,
        batch_size: int = 32,
        epochs: int = 1,
        lr: float = 0.001,
        rand_num: int = 30,
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.rand_num = rand_num


conf = Config()


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def amplitude_mix(x, e=0.9, t=0.01):
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    complex_signal = (x[:, 0, :] + 1j * x[:, 1, :]).cpu().numpy()

    fft_signal = np.fft.fft(complex_signal, axis=-1, norm='ortho')

    amplitude = np.abs(fft_signal)
    phase = np.angle(fft_signal)

    alpha = np.random.uniform(0, e)

    amplitude_interpolated = (1 - alpha) * amplitude + alpha * amplitude[index, :]

    perturbation = np.random.uniform(-t, t, phase.shape)
    phase_perturbed = phase + perturbation

    fft_interpolated = amplitude_interpolated * np.exp(1j * phase_perturbed)

    interpolated_signal = np.fft.ifft(fft_interpolated, axis=-1, norm='ortho')

    interpolated_real = np.real(interpolated_signal)
    interpolated_imag = np.imag(interpolated_signal)

    interpolated_iq_signal = np.stack((interpolated_real, interpolated_imag), axis=1)

    interpolated_iq_signal = torch.tensor(interpolated_iq_signal).float().to(device)

    return interpolated_iq_signal


def train(FE, FC, tra_loader, trb_loader, trc_loader, optimizer, epoch, writer):
    FE.train()
    FC.train()

    total_loss = 0
    correct = 0
    n_batches = min(len(tra_loader), len(trb_loader), len(trc_loader))
    start_time = time.time()
    for (a, a_labels), (b, b_labels), (c, c_labels) in zip(tra_loader, trb_loader, trc_loader):
        x = torch.cat([a, b, c])
        x = x.to(device)
        labels = torch.cat([a_labels, b_labels, c_labels])
        labels = labels.long().to(device)

        data, target_a, target_b, lam = mixup_data(x, labels, alpha=0.7, use_cuda=True)
        data, target_a, target_b = data.to(device), target_a.to(device), target_b.to(device)

        x = amplitude_mix(data)

        optimizer.zero_grad()
        fe = FE(x)
        output = FC(fe)
        classifier_output = F.log_softmax(output, dim=1)
        label_loss = mixup_criterion(F.nll_loss, classifier_output, target_a, target_b, lam)
        label_loss.backward()
        optimizer.step()

        total_loss += label_loss.item()
        pred = classifier_output.argmax(dim=1, keepdim=True)
        correct += lam * pred.eq(target_a.view_as(pred)).sum().item() + (1 - lam) * pred.eq(
            target_b.view_as(pred)).sum().item()

    end_time = time.time()
    training_time = end_time - start_time
    mean_loss = total_loss / n_batches
    print('Train Epoch: {} \tLoss: {:.6f}, Accuracy: ({:.6f}%), TrainingTime: {:.6f} \n'.format(
        epoch,
        mean_loss,
        correct / n_batches,
        training_time)
    )
    writer.add_scalar('Accuracy/train', 100.0 * correct / n_batches, epoch)
    writer.add_scalar('Loss/train', mean_loss, epoch)


def val(FE, FC, vaa_loader, vab_loader, vac_loader, epoch, writer):
    FE.eval()
    FC.eval()

    with torch.no_grad():
        n_batches = min(len(vaa_loader), len(vab_loader), len(vac_loader))
        total_loss = 0
        total_accuracy = 0
        for (a, a_labels), (b, b_labels), (c, c_labels) in zip(vaa_loader, vab_loader, vac_loader):
            x = torch.cat([a, b, c])
            x = x.to(device)
            labels = torch.cat([a_labels, b_labels, c_labels])
            labels = labels.long().to(device)

            fe = FE(x)

            preds = FC(fe)

            output = F.log_softmax(preds, dim=1)
            loss = F.nll_loss(output, labels)

            total_loss += loss.item()
            total_accuracy += (preds.max(1)[1] == labels).float().mean().item()

        mean_loss = total_loss / n_batches
        mean_accuracy = total_accuracy / n_batches

        print('Val Epoch: {} \tLoss: {:.6f}, Accuracy: {:.6f} \n'.format(
            epoch,
            mean_loss,
            mean_accuracy,
        ))

        writer.add_scalar('Accuracy/validation', mean_accuracy, epoch)
        writer.add_scalar('Classifier_Loss/validation', mean_loss, epoch)

    return mean_loss


def test(FE, FC, test_dataloader):
    FE.eval()
    FC.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
            fe = FE(data)
            output = FC(fe)
            output = F.log_softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            acc = 100.0 * correct / len(test_dataloader.dataset)

    fmt = '\nTest set: Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            correct,
            len(test_dataloader.dataset),
            acc,
        )
    )
    return acc


class EarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model_fe, model_fc, path_fe, path_fc):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model_fe, model_fc, path_fe, path_fc)
        elif val_loss > self.best_loss:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(val_loss, model_fe, model_fc, path_fe, path_fc)
            self.best_loss = val_loss
            self.counter = 0

    def save_checkpoint(self, val_loss, model_fe, model_fc, path_fe, path_fc):
        print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model_fe, path_fe)
        torch.save(model_fc, path_fc)


def train_and_val_and_test(FE, FC, tra_dataset, trb_dataset, trc_dataset, vaa_dataset, vab_dataset, vac_dataset, optimizer, scheduler, epochs, writer, current_time):
    early_stopping = EarlyStopping(patience=20)
    for epoch in range(1, epochs + 1):
        tra_loader = DataLoader(tra_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=1)
        trb_loader = DataLoader(trb_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=1)
        trc_loader = DataLoader(trc_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=1)
        vaa_loader = DataLoader(vaa_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=1)
        vab_loader = DataLoader(vab_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=1)
        vac_loader = DataLoader(vac_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=1)
        train(FE, FC, tra_loader, trb_loader, trc_loader, optimizer, epoch, writer)
        val_loss = val(FE, FC, vaa_loader, vab_loader, vac_loader, epoch, writer)

        scheduler.step()

        early_stopping(val_loss, FE, FC, f'model_weight/TFMix_CR_{current_time}_fe.pth',
                       f'model_weight/TFMix_CR_{current_time}_fc.pth')
        if early_stopping.early_stop:
            print("Early stopping")
            break
        print("------------------------------------------------")


def main(current_time):
    writer = SummaryWriter(f'logs/TFMix_CR_{current_time}')
    random_seed = 300
    set_seed(random_seed)

    FE = base_complex_model().to(device)
    FC = Net().to(device)

    dates = ['1-1', '1-19', '14-7', '18-2']

    train_val_dates = [date for date in dates if date != current_time]

    datasets = {}
    for i, date in enumerate(train_val_dates):
        x_train, x_val, x_test, y_train, y_val, y_test = Domain_Dataset(date, conf.rand_num)
        datasets[f'train_dataset_{i}'] = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
        datasets[f'val_dataset_{i}'] = TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val))
        datasets[f'test_dateset_{i}'] = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))

    x_train, x_val, x_test, y_train, y_val, y_test = Domain_Dataset(current_time, conf.rand_num)
    test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=1)

    optim = torch.optim.Adam(list(FE.parameters()) + list(FC.parameters()), lr=conf.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.95)

    train_and_val_and_test(FE=FE,
                           FC=FC,
                           tra_dataset=datasets['train_dataset_0'],
                           trb_dataset=datasets['train_dataset_1'],
                           trc_dataset=datasets['train_dataset_2'],
                           vaa_dataset=datasets['val_dataset_0'],
                           vab_dataset=datasets['val_dataset_1'],
                           vac_dataset=datasets['val_dataset_2'],
                           optimizer=optim,
                           scheduler=scheduler,
                           epochs=conf.epochs,
                           writer=writer,
                           current_time=current_time,
                           )

    acc = test(FE, FC, test_dataloader=test_loader)
    f = open(f'result/TFMix_CR_Acc.txt', 'a+')
    f.write(str(acc) + " " + str(current_time) + '\n')


if __name__ == '__main__':
    dates_all = ['1-1', '1-19', '14-7', '18-2']
    for current_time in dates_all:
        main(current_time)





