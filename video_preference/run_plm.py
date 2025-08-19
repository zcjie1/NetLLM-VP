import argparse
from config import cfg  # 假设 config.py 里定义了默认值
import numpy as np
import random
import torch
import time
import sys
import os
import csv
from dataset.load_dataset import create_preference_dataset
from models.networking_head import NetworkingHead
from models.low_rank import peft_model
from models.pipeline import Pipeline
from utils.plms_utils import load_plm
from utils.console_logger import ConsoleLogger
from torch.optim import AdamW
from torch.utils.data import DataLoader


def save_model(args, model, save_dir):
    """
    save fune-tune model
    """
    if args.rank != -1:
        # save low rank matrices
        model.plm.save_pretrained(save_dir)
        # save other modules except plm
        torch.save(model.modules_except_plm.state_dict(),
                   os.path.join(save_dir, 'modules_except_plm.bin'))
    else:
        # low rank matrices are disabled, save whole model
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.bin'))


def load_model(args, model, model_dir):
    """
    load fune-tune model

    :return: the pretrained model corresponding to using model_dir
    """
    if args.rank != -1:
        # load low rank matrices
        model.plm.load_adapter(model_dir, adapter_name='default')
        # load other modules except plm
        model.modules_except_plm.load_state_dict(torch.load(
            os.path.join(model_dir, 'modules_except_plm.bin')))
    else:
        # low rank matrices are disabled, load whole model
        model.load_state_dict(torch.load(os.path.join(model_dir, 'model.bin')))
    return model


def adapt(args, pipeline, dataloader_train, dataloader_valid, models_dir, grad_accum_steps):
    ts = time.time()
    file_prefix = f'epochs_{args.epochs}_lr_{args.lr}_rank_{args.rank}_ts_{ts}'

    checkpoint_path = os.path.join(models_dir, file_prefix, 'checkpoint')
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    best_model_path = os.path.join(models_dir, file_prefix, 'best_model')
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    
    console_log = open(os.path.join(
        models_dir, file_prefix + '_console.log'), 'w')
    sys.stdout = ConsoleLogger(sys.__stdout__, console_log)

    if args.resume:
        if args.resume_path is None:
            if best_model_path and os.path.isdir(best_model_path) and len(os.listdir(best_model_path)) > 0:
                # 加载最佳模型
                print(
                    '\033[33mWarning:\033[0m args.resume_path is not set, using the default best model path:', best_model_path)
                pipeline = load_model(args, pipeline, best_model_path)
                print('Resume weights for training from:', best_model_path)
            else:
                print(
                    '\033[33mWarning:\033[0m resume_path and best_model_path are both not set, skipping model loading.')
                # 不加载模型，直接跳过
        else:
            pipeline = load_model(args, pipeline, args.resume_path)
            print('Resume weights for training from:', args.resume_path)

    if not args.freeze_plm:
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in pipeline.plm.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.lr},
            {'params': [p for n, p in pipeline.plm.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': args.lr},
            # {'params': pipeline.embed_vp.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            # {'params': pipeline.embed_ln.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            # {'params': pipeline.embed_multimodal.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr}
            {'params': pipeline.input_embedding_net.parameters(
            ), 'weight_decay': args.weight_decay, 'lr': args.lr}
        ]
        optimizer = AdamW(optimizer_grouped_parameters)
    else:
        # only tune networking head and multimodal encoder
        optimizer_grouped_parameters = [
            # {'params': pipeline.embed_vp.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            # {'params': pipeline.embed_ln.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr},
            # {'params': pipeline.embed_multimodal.parameters(), 'weight_decay': args.weight_decay, 'lr': args.lr}
            {'params': pipeline.input_embedding_net.parameters(
            ), 'weight_decay': args.weight_decay, 'lr': args.lr}
        ]
        optimizer = AdamW(optimizer_grouped_parameters)

    assert args.epochs_per_valid is None or args.steps_per_valid is None, "You can only specify args.epochs_per_valid or args.steps_per_valid."

    global_step = 0
    report_loss_per_steps = args.report_loss_per_steps
    tot_loss = 0
    log_loss = 0
    best_loss = float('inf')
    best_acc = 0
    best_epoch, best_step = 0, 0

    def validate():
        pipeline.eval()
        with torch.no_grad():
            validata_checkpoint_path = os.path.join(checkpoint_path)
            if not os.path.exists(validata_checkpoint_path):
                os.makedirs(validata_checkpoint_path)
            save_model(args, pipeline, validata_checkpoint_path)
            print(f'Checkpoint saved at', checkpoint_path)

            valid_loss = []
            all_preds = []
            all_labels = []

            for history, future, video_user_info in dataloader_valid:
                history, future = history.to(
                    args.device), future.to(args.device)
                loss, logits = pipeline(
                    history, future, video_user_info, teacher_forcing=False, return_logits=True)
                valid_loss.append(loss.item())

                if args.loss_func == 'bce':
                    probs = logits.squeeze(-1)
                    preds = (probs >= 0.5).long()
                else:
                    preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy().reshape(-1))
                all_labels.extend(future.cpu().numpy().reshape(-1))

            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)

            accuracy = np.mean(all_preds == all_labels)  # 确保是数组再计算mean
            valid_loss = sum(valid_loss) / len(valid_loss)

            print(f'Validation accuracy: {accuracy:.4f}')
            pipeline.train()
            return valid_loss, accuracy

    print(f'Training  - bs: {args.bs} - lr: {args.lr} - seed: {args.seed}')
    for epoch in range(args.epochs):
        pipeline.train()
        for step, (history, future, video_user_info) in enumerate(dataloader_train):
            global_step += 1
            history, future = history.to(args.device), future.to(args.device)
            history = history
            future = future
            # using scheduled sampling
            if args.scheduled_sampling:
                if np.random.rand() > args.mix_rate:
                    loss = pipeline(history, future,
                                    video_user_info, teacher_forcing=True)
                else:
                    loss = pipeline(history, future,
                                    video_user_info, teacher_forcing=False)
            else:
                loss = pipeline(history, future, video_user_info,
                                teacher_forcing=True)
            tot_loss += loss.item()
            loss = loss / grad_accum_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pipeline.plm.parameters(), 1.0)

            # perform gradient accumulation update
            # optimizer.step() 和 optimizer.zero_grad() 不会在每一步都执行
            # 只会在累积了 grad_accum_steps 个批次的梯度后才执行一次
            if ((step + 1) % grad_accum_steps == 0) or (step + 1 == len(dataloader_train)):
                optimizer.step()
                optimizer.zero_grad()

            # report training loss
            if global_step % report_loss_per_steps == 0:
                print("Epoch {}, global_step {}, average loss: {}".format(
                    epoch, global_step, (tot_loss - log_loss) / report_loss_per_steps), flush=True)
                log_loss = tot_loss

            # for debug
            # if global_step >= 300:
            #     save_model(args, pipeline, best_model_path)
            #     break

            # validation by steps
            if args.steps_per_valid is not None and global_step % args.steps_per_valid == 0:
                valid_loss, valid_acc = validate()
                if valid_acc > best_acc:
                    best_loss, best_acc, best_step = valid_loss, valid_acc, global_step
                    save_model(args, pipeline, best_model_path)
                    print(
                        f'Best model (step {best_step}, average valid loss {best_loss}, average valid acc {best_acc}) saved at', best_model_path)
                print('Valid loss', valid_loss, ' - ',
                      'Best loss', best_loss, 'at step', best_step)

            # save checkpoint by save_checkpoint_per_step
            if args.save_checkpoint_per_step is not None and global_step % args.save_checkpoint_per_step == 0:
                save_checkpoint_path = os.path.join(checkpoint_path, str(
                    global_step // args.save_checkpoint_per_step))  # save checkpoint
                if not os.path.exists(save_checkpoint_path):
                    os.makedirs(save_checkpoint_path)
                save_model(args, pipeline, save_checkpoint_path)
                print('save checkpoint at', save_checkpoint_path)

        # validation by epochs
        if args.epochs_per_valid is not None and epoch % args.epochs_per_valid == 0:
            valid_loss = validate()
            if valid_loss < best_loss:
                best_loss, best_epoch = valid_loss, epoch
                save_model(args, pipeline, best_model_path)
                print(
                    f'Best model (epoch {best_epoch}, average valid loss {best_loss}) saved at', best_model_path)
            print('Valid loss', valid_loss, ' - ', 'Best loss',
                  best_loss, 'at epoch', best_epoch)

        # save checkpoint by save_checkpoint_per_epoch
        if args.save_checkpoint_per_epoch is not None and epoch % args.save_checkpoint_per_epoch == 0 and epoch > 0:
            save_checkpoint_path = os.path.join(
                checkpoint_path, f'epoch{epoch}')  # save checkpoint
            if not os.path.exists(save_checkpoint_path):
                os.makedirs(save_checkpoint_path)
            save_model(args, pipeline, save_checkpoint_path)
            print('save checkpoint at', save_checkpoint_path)

    print('Done adaptation, average training loss =', tot_loss / global_step)


def test(args, pipeline, dataloader_test, models_dir, results_dir):
    file_prefix = f'his_{args.his_window}_fut_{args.fut_window}_epochs_{args.epochs}_bs_{args.bs * args.grad_accum_steps}_'\
                  f'lr_{args.lr}_seed_{args.seed}_rank_{args.rank}_scheduled_sampling_{args.scheduled_sampling}'
    best_model_path = os.path.join(
        models_dir, file_prefix, 'checkpoint', 'epoch150')
    result_path = os.path.join(results_dir, file_prefix + '_results.csv')
    print('result path:', result_path)
    model_path = args.model_path if args.model_path is not None else best_model_path
    print('Model path:', model_path)
    if os.path.exists(model_path):
        pipeline = load_model(args, pipeline, model_path)
        print('Load weights from:', model_path)
    else:
        print('\033[33mWarning:\033[0m', model_path,
              'not found, skip loading weights.')

    print(f'Testing - seed: {args.seed}')

    total = 0
    correct = 0

    # 打开 CSV 文件写入
    os.makedirs(results_dir, exist_ok=True)
    with open(result_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video_name', 'height', 'fps',
                        'video_time', 'predicted_class', 'ground_truth'])

        with torch.no_grad():
            for history, future, video_info in dataloader_test:
                history, future = history.to(
                    args.device), future.to(args.device)
                pred, gt = pipeline.inference(history, future, video_info)

                # 获取预测类别
                if args.loss_func == 'bce':
                    # shape: [batch_size, seq_len]
                    predicted_class = (pred.squeeze(-1) > 0.5).long()
                else:
                    predicted_class = torch.argmax(
                        pred, dim=2)  # [batch_size, seq_len]

                # 视频信息
                video_name = video_info[0]
                height = int(video_info[1])
                fps = int(video_info[2])
                video_time = int(video_info[3])

                for i in range(predicted_class.shape[0]):
                    for t in range(predicted_class.shape[1]):
                        pred_label = int(predicted_class[i, t])
                        gt_label = int(gt[i, t])
                        writer.writerow([
                            video_name,
                            height,
                            fps,
                            video_time,
                            pred_label,
                            gt_label
                        ])
                        # 统计准确率
                        total += 1
                        if pred_label == gt_label:
                            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f'Results written to: {result_path}')
    print(f'Test Accuracy: {accuracy:.4f} ({correct}/{total})')


def run(args):
    assert args.plm_type in cfg.plm_types
    assert args.plm_size in cfg.plm_sizes

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    if args.rank != -1:
        models_dir = os.path.join(cfg.plms_finetuned_dir, f'{args.plm_type}_{args.plm_size}_low_rank',
                                  f'freeze_plm_{args.freeze_plm}')
        results_dir = os.path.join(cfg.results_dir, f'{args.plm_type}_{args.plm_size}_low_rank',
                                   f'freeze_plm_{args.freeze_plm}')
    else:
        models_dir = os.path.join(cfg.plms_finetuned_dir, f'{args.plm_type}_{args.plm_size}',
                                  f'freeze_plm_{args.freeze_plm}')
        results_dir = os.path.join(cfg.results_dir, f'{args.plm_type}_{args.plm_size}',
                                   f'freeze_plm_{args.freeze_plm}')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # args.device_out and args.device_mid are used used for model parallelism (currently only necessary for llama)
    # For data/modules near the input side, we use args.device.
    # For data/modules near the output side, we use args.device_out.
    # For data/modules lying in the middle, we use args.device_mid (it can be None).
    # If args.device == args.device_out == args.device_mid (if not None), everything will be the same as using only one device.
    plm, tokenizer, _ = load_plm(args.plm_type, os.path.join(cfg.plms_dir, args.plm_type, args.plm_size), plm_size=args.plm_size,
                                 device_input_side=args.device, device_output_side=args.device_out, device_middle_side=args.device_mid)
    # other plm can simply be loaded on one device
    if (args.plm_type == 'opt' or args.plm_type == 'gpt2') and args.plm_size != 'large':
        plm = plm.to(args.device)

    if args.rank != -1:
        # 基于 PEFT 库的 Lora 微调初始化
        plm = peft_model(plm, args.plm_type, args.rank)

    # set up networking head
    input_dim = plm.hidden_size
    out_dim = 2   # 二分类任务，使用BCE损失函数
    if args.loss_func == 'bce':
        out_dim = 1  # 二分类任务，使用BCE损失函数
    if args.plm_type == 'opt' and args.plm_size == 'xxs':
        networking_head = NetworkingHead(
            input_dim=512, out_dim=out_dim, fut_window=args.fut_window).to(args.device_out)
    else:
        networking_head = NetworkingHead(
            input_dim=input_dim, out_dim=out_dim, fut_window=args.fut_window).to(args.device_out)
    plm.set_networking_head(networking_head)
    print('PLM model architecture:')
    print(plm)

    if args.plm_type == 'gpt2':
        embed_size = 1024
    if args.plm_type == 'llama':
        embed_size = 4096
    if args.plm_type == 'mistral':
        embed_size = 4096
    if args.plm_type == 'opt' and args.plm_size == 'xxs':
        embed_size = 512
    if args.plm_type == 'opt' and args.plm_size == 'xs':
        embed_size = 2048
    if args.plm_type == 'opt' and args.plm_size == 'small':
        embed_size = 2560
    if args.plm_type == 'opt' and args.plm_size == 'base':
        embed_size = 4096
    if args.plm_type == 'opt' and args.plm_size == 'large':
        embed_size = 5120
    if args.plm_type == 'llava':
        embed_size = 4096

    pipeline = Pipeline(plm,
                        loss_func=args.loss_func,
                        fut_window=args.fut_window,
                        device=args.device,
                        embed_size=embed_size,
                        using_multimodal=args.using_multimodal)
    # print_trainable_parameters(pipeline)

    if args.compile:
        assert torch.__version__ >= '2.0.0', 'Compile model requires torch version >= 2.0.0, but current torch version is ' + torch.__version__
        print("\033[33mWarning:\033[0m There seems to be some bugs in torch.compile. If batch size is too large, it will raise errors (I don't know why this happens).")
        # recommend to compile model when you are using PyTorch 2.0
        prompt_model = torch.compile(prompt_model).to(args.device)

    torch.set_float32_matmul_precision('high')

    if args.adapt:
        raw_dataset_train, raw_dataset_valid = create_preference_dataset(dataset_dir=cfg.dataset_dir,
                                                                         split_type=args.split_type,
                                                                         his_window=args.his_window,
                                                                         fut_window=args.fut_window,
                                                                         video_len=args.video_len)
        dataloader_train = DataLoader(
            raw_dataset_train, batch_size=args.bs, shuffle=True, pin_memory=True)
        dataloader_valid = DataLoader(
            raw_dataset_valid, batch_size=args.bs, shuffle=False, pin_memory=True)
        adapt(args, pipeline, dataloader_train, dataloader_valid,
              models_dir, args.grad_accum_steps)

    if args.test:
        raw_dataset_train, raw_dataset_test = create_preference_dataset(dataset_dir=cfg.dataset_dir, 
                                                                        split_type=args.split_type, 
                                                                        his_window=args.his_window, 
                                                                        fut_window=args.fut_window, 
                                                                        video_len=args.video_len)
        dataloader_test = DataLoader(
            raw_dataset_test, batch_size=args.bs, shuffle=True, pin_memory=True)
        test(args, pipeline, dataloader_test, models_dir, results_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process the input parameters to train the network.')

    # ========== model/plm settings related arguments ==========
    parser.add_argument('--adapt', action="store_true", help='adapt llm.')
    parser.add_argument('--test', action="store_true", help='test llm.')
    parser.add_argument('--plm-type', action="store",
                        dest='plm_type', help='type of plm.', default='t5-lm')
    parser.add_argument('--plm-size', action="store",
                        dest='plm_size', help='size of plm.', default='base')
    parser.add_argument('--model-path', action="store", dest='model_path', type=str,
                        help='(Optional) The directory of model weights to be loaded for testing.')
    parser.add_argument('--device', action='store', dest='device',
                        help='the device (cuda or cpu) to run experiment.')
    parser.add_argument('--device-out', action='store', dest='device_out',
                        help='the device (cuda or cpu) to place the split of model near the output.')
    parser.add_argument('--device-mid', action='store', dest='device_mid',
                        help='the device (cuda or cpu) to place the split of model between the input and output.')
    parser.add_argument('--freeze-plm', action='store_true',
                        dest='freeze_plm', help='freeze weights of plm during training')
    parser.add_argument('--compile', action='store_true', dest='compile',
                        help='(Optional) Compile model for speed up (available only for PyTorch 2.0).')
    parser.add_argument('--resume', action='store_true', dest='resume',
                        help='(Optional) Resume model weights from checkpoint for training.')
    parser.add_argument('--loss-func', action='store', dest='loss_func', type=str,
                        help='(Optional) The directory of model weights to be loaded for resuming training.')

    # ========== dataset loading/processing settings related arguments ==========
    parser.add_argument('--split-type', action='store', dest='split_type',
                        help='(Optional) How to divide the data set', type=str)
    parser.add_argument('--his-window', action='store', dest='his_window',
                        help='(Optional) Use the feature information of the last few seconds (default 3)', type=int)
    parser.add_argument('--fut-window', action='store', dest='fut_window',
                        help='(Optional) predict the perference of the few future second (default 3)', type=int)
    parser.add_argument('--video-len', action='store', dest='video_len',
                        help='(Optional) the length of video (default 10s)', type=int)

    # ========== training related settings ==========
    parser.add_argument('--epochs', action="store", dest='epochs',
                        help='(Optional) Neural network learning epochs.', type=int)
    parser.add_argument('--epochs-per-valid', action='store', dest='epochs_per_valid', type=int,
                        help='(Optional) The number of epochs per validation (default 3).')
    parser.add_argument('--steps-per-valid', action='store', dest='steps_per_valid', type=int,
                        help='(Optional) The number of steps per validation (default 50).')
    parser.add_argument('--report-loss-per-steps', action='store', dest='report_loss_per_steps', type=int, default=100,
                        help='(Optional) The number of steps per validation (default 100).')
    parser.add_argument('--lr', action="store", dest='lr',
                        help='(Optional) Neural network learning rate.', type=float)
    parser.add_argument('--weight-decay', action="store", dest='weight_decay',
                        help='(Optional) Neural network weight decay.', type=float, default=1e-4)
    parser.add_argument('--bs', action="store", dest='bs',
                        help='(Optional) Neural network batch size.', type=int)
    parser.add_argument('--grad-accum-steps', action="store",
                        dest='grad_accum_steps', type=int, default=16)
    parser.add_argument('--seed', action="store", dest='seed', type=int,
                        default=1, help='(Optional) Random seed (default to 1).')
    parser.add_argument('--multimodal', action="store_true",
                        dest='using_multimodal', help='using multimodal image features.')
    parser.add_argument('--save-checkpoint-per-epoch', action="store",
                        dest='save_checkpoint_per_epoch', help='save checkpoint per epoch', type=int)
    parser.add_argument('--save-checkpoint-per-step', action="store",
                        dest='save_checkpoint_per_step', help='save checkpoint per step', type=int)
    parser.add_argument('--rank', action="store", dest='rank',
                        help='the rank of low rank matrices', type=int, default=-1)
    parser.add_argument('--resume-path', action="store",
                        dest='resume_path', help='using for resume')
    parser.add_argument('--scheduled-sampling', action="store_true", dest='scheduled_sampling', 
                        help='using scheduled sampling, a common method to reduce exposure bias to improve '
                        'sequence generation by mixing teacher-forcing generation and auto-regressive generation. '
                        'see: https://www.activeloop.ai/resources/glossary/scheduled-sampling/')
    parser.add_argument('--mix-rate', action="store", dest='mix_rate',
                        help='the mixing rate when using scheduled sampling', type=float, default=0.1)

    args = parser.parse_args()

    # handle defautl settings
    args.split_type = cfg.split_type if args.split_type is None else args.split_type
    args.his_window = cfg.his_window if args.his_window is None else args.his_window
    args.fut_window = cfg.fut_window if args.fut_window is None else args.fut_window
    args.epochs = cfg.default_epochs if args.epochs is None else args.epochs
    args.lr = cfg.default_lr if args.lr is None else args.lr
    args.weight_decay = cfg.default_weight_decay if args.weight_decay is None else args.weight_decay
    args.bs = cfg.default_bs if args.bs is None else args.bs
    args.grad_accum_steps = cfg.default_grad_accum_step if args.grad_accum_steps is None else args.grad_accum_steps

    if args.device_out is None:
        args.device_out = args.device

    print(args)
    run(args)
