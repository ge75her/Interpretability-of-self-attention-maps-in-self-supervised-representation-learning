from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import time
from pathlib import Path
import json
import torch


from utils import *
from model import *
    
def train_dino(data_path,batch_size,lr,weight_decay,weight_decay_end,min_lr,out_dim,tau,total_epochs,warmup_epochs,momentum_teacher,output_dir,saveckp_freq,
            clip_grad,freeze_last_layer):

    transform = DataAugmentation()
    dataset = ImageFolder(data_path, transform=transform)
    data_loader = DataLoader(dataset,batch_size=batch_size,drop_last=True,)
    print(f"Data loaded: there are {len(dataset)} images.")

    #student =VisionTransformer(patch_size=8,drop_path_ratio=0.1,)  # stochastic depth
    #teacher = VisionTransformer(patch_size=8)
    student=torch.hub.load('facebookresearch/dino:main', 'dino_vits8',drop_path_rate=0.1,pretrained=True) #use pretrained model to speed up the training process
    teacher= torch.hub.load('facebookresearch/dino:main', 'dino_vits8',pretrained=True)
    embed_dim = student.embed_dim

    student = MultiCropWrapper(student, DINOHead(embed_dim,out_dim=out_dim,use_bn=False,norm_last_layer=True,))
    teacher = MultiCropWrapper(teacher,DINOHead(embed_dim, out_dim=out_dim, use_bn=False),)
    # move networks to gpu
    student=student.cuda()
    teacher=teacher.cuda()
    params_groups = get_params_groups(student)
    optimizer = torch.optim.AdamW(params_groups)
    # teacher and student start with the same weights
    teacher.load_state_dict(student.state_dict())
    dino_loss = Loss(tau=tau,out_dim=out_dim).cuda()
    #there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both vit network.")


    lr_schedule = cosine_scheduler(lr * (batch_size* get_world_size()) / 128.,min_lr, total_epochs, len(data_loader),warmup_epochs=warmup_epochs,)
    wd_schedule = cosine_scheduler(weight_decay,weight_decay_end,total_epochs, len(data_loader),)
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = cosine_scheduler(momentum_teacher, 1,total_epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    #start to training
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(output_dir, "checkpoint10.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        dino_loss=dino_loss,
    )
    
    start_epoch = to_restore["epoch"]
    start_time = time.time()
    print("Starting DINO training !")
    
    for epoch in range(start_epoch,total_epochs):
        train_stats=train_one_epoch(student,teacher,dino_loss,data_loader,optimizer,lr_schedule,wd_schedule,momentum_schedule,epoch,output_dir,
                                    total_epochs,clip_grad,freeze_last_layer)
        
        save_dict = {'student': student.state_dict(),'teacher': teacher.state_dict(),'optimizer': optimizer.state_dict(),'epoch': epoch + 1,'loss': dino_loss.state_dict()}
        if epoch%saveckp_freq==0:
            save_on_master(save_dict,os.path.join(output_dir,'checkpoint{}.pth'.format(epoch)))
        
        log_stats={**{f'train_{k}': v for k, v in train_stats.items()},'epoch': epoch}
        with (Path(output_dir)/'log.txt').open("a") as f:
            f.write(json.dumps(log_stats)+'\n')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



def train_one_epoch(student,teacher,dino_loss,data_loader,optimizer,lr_schedule,wd_schedule,momentum_schedule,epoch,output_dir
                ,total_epochs,clip_grad,freeze_last_layer):
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, total_epochs)
    for it, (images, _) in enumerate(metric_logger.log_every(data_loader, 3000, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda() for im in images]
        teacher_output=teacher(images[0]) #1st image
        student_output=student(images[1]) #2nd image
        loss=dino_loss(student_output,teacher_output)
        #print('loss:{:.4f}, stopping training'.format(loss.item()))

        optimizer.zero_grad()
        loss.backward()
        param_norms=clip_gradients(student,clip=clip_grad)
        cancel_gradients_last_layer(epoch,student,freeze_last_layer)
        optimizer.step()


        #EMA update teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__=='__main__':
    ''' Parameters:
    data_path(str): path of original MNIST dataset
    batch_size(int): parameter of data Loader, usually set a big value
    lr(float): learning rate at the end of linear warmup. The learning rate is linearly scaled with the batch size
    weight_decay(float): inital value of the weight decay
    weight_decay_end(float):final value of the weight decay
    min_lr(float):target learning rate at the end of optimization
    momentum_teacher(float): base EMA parameter for teacher update. During training with cosine schedule, the value is increated to 1.
    out_dim(int): dim of the DINOhead output, for MNIST dataset, don't need to set too big.
    tau(float): parameter for simCLR loss, set a big tau when batch_size is big
    clip_grad(float): maximal parameter gradient norm if using gradent clipping
    freeze_last_layer(int): number of epochs during which we keep the output layer fixed. Typically doing so during first epoch helps training
    output_dir(str): path to save logs and checkpoints
    '''
    data_path='data/MNIST'
    batch_size=128
    lr=1e-4
    weight_decay=0.04
    weight_decay_end=0.4
    min_lr=1e-8

    total_epochs=40
    warmup_epochs=3
    momentum_teacher=0.995
    out_dim=1024
    tau=0.1
    saveckp_freq=3

    clip_grad=3.0
    freeze_last_layer=1
    output_dir='/content/drive/MyDrive/cluttered_mnist_diff/logs'

    train_dino(data_path,batch_size,lr,weight_decay,weight_decay_end,min_lr,out_dim,tau,total_epochs,warmup_epochs,momentum_teacher,output_dir,saveckp_freq,
                clip_grad,freeze_last_layer)
