import torch
import torchvision
import os
from PIL import Image
import time
import tqdm

load=torchvision.transforms.Compose([torchvision.transforms.Resize((512,512)),torchvision.transforms.ToTensor()])
unload=torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),torchvision.transforms.Resize(( 512,512)),])

def getImageAndTarget(imagefile):
    image=Image.open("processed data/images/"+imagefile)
    target=Image.open("processed data/masks/"+imagefile)
    return load(image),load(target)
class Unet(torch.nn.Module):
    def __init__(self,n=3):
        super(Unet,self).__init__()
        #block 1
        self.block1_conv1=torch.nn.Conv2d(n,16,kernel_size=3,padding=1)
        self.block1_drop1=torch.nn.Dropout(0.1)
        self.block1_conv2=torch.nn.Conv2d(16,16,kernel_size=3,padding=1)
        self.block1_relu=torch.nn.ReLU()
        self.maxpool1=torch.nn.MaxPool2d(2)
        #block 2
        self.block2_conv1=torch.nn.Conv2d(16,32,kernel_size=3,padding=1)
        self.block2_drop1=torch.nn.Dropout(0.1)
        self.block2_conv2=torch.nn.Conv2d(32,32,kernel_size=3,padding=1)
        self.block2_relu=torch.nn.ReLU()
        self.maxpool2=torch.nn.MaxPool2d(2)
        #block 3
        self.block3_conv1=torch.nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.block3_drop1=torch.nn.Dropout(0.1)
        self.block3_conv2=torch.nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.block3_relu=torch.nn.ReLU()
        self.maxpool3=torch.nn.MaxPool2d(2)
        #block 4
        self.block4_conv1=torch.nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.block4_drop1=torch.nn.Dropout(0.1)
        self.block4_conv2=torch.nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.block4_relu=torch.nn.ReLU()
        self.maxpool4=torch.nn.MaxPool2d(2)
        #block 5
        self.block5_conv1=torch.nn.Conv2d(128,256,kernel_size=3,padding=1)
        self.block5_drop1=torch.nn.Dropout(0.1)
        self.block5_conv2=torch.nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.block5_relu=torch.nn.ReLU()
        #block 6 -upsample
        self.block6_up_conv1=torch.nn.ConvTranspose2d(256,128,kernel_size=2,padding=0,stride=2)
        self.block6_conv1=torch.nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.block6_drop1=torch.nn.Dropout(0.1)
        self.block6_conv2=torch.nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.block6_relu=torch.nn.ReLU()
        #block 7 -upsample
        self.block7_up_conv1=torch.nn.ConvTranspose2d(128,64,kernel_size=2,padding=0,stride=2)
        self.block7_conv1=torch.nn.Conv2d(128,64,kernel_size=3,padding=1)
        self.block7_drop1=torch.nn.Dropout(0.1)
        self.block7_conv2=torch.nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.block7_relu=torch.nn.ReLU()
        #block 8 -upsample
        self.block8_up_conv1=torch.nn.ConvTranspose2d(64,32,kernel_size=2,padding=0,stride=2)
        self.block8_conv1=torch.nn.Conv2d(64,32,kernel_size=3,padding=1)
        self.block8_drop1=torch.nn.Dropout(0.1)
        self.block8_conv2=torch.nn.Conv2d(32,32,kernel_size=3,padding=1)
        self.block8_relu=torch.nn.ReLU()
        #block 9 -upsample
        self.block9_up_conv1=torch.nn.ConvTranspose2d(32,16,kernel_size=2,padding=0,stride=2)
        self.block9_conv1=torch.nn.Conv2d(32,16,kernel_size=3,padding=1)
        self.block9_drop1=torch.nn.Dropout(0.1)
        self.block9_conv2=torch.nn.Conv2d(16,1,kernel_size=3,padding=1)
        self.block9_relu=torch.nn.ReLU()

    def forward(self,x):
        #out4 = torch.cat((out_up1, out2), dim=1)
        out=self.block1_conv1(x)
        out=self.block1_drop1(out)
        out_block1=self.block1_relu(self.block1_conv2(out))
        out=self.maxpool1(out_block1)

        #block 2 starts
        out=self.block2_conv1(out)
        out=self.block2_drop1(out)
        out_block2=self.block2_relu(self.block2_conv2(out))
        out=self.maxpool2(out_block2)

        #block3
        out=self.block3_conv1(out)
        out=self.block3_drop1(out)
        out_block3=self.block3_relu(self.block3_conv2(out))
        out=self.maxpool3(out_block3)
        #block4
        out=self.block4_conv1(out)
        out=self.block4_drop1(out)
        out_block4=self.block4_relu(self.block4_conv2(out))
        out=self.maxpool4(out_block4)
        #block 5
        out=self.block5_conv1(out)
        out=self.block5_drop1(out)
        out_block5=self.block5_relu(self.block5_conv2(out))
        #block 6
        out=self.block6_up_conv1(out_block5)

        out = torch.cat((out,out_block4), dim=1)
        out=self.block6_conv1(out)
        out=self.block6_drop1(out)
        out_block6=self.block6_relu(self.block6_conv2(out))
        #block 7
        out=self.block7_up_conv1(out_block6)
        out = torch.cat((out,out_block3), dim=1)
        out=self.block7_conv1(out)
        out=self.block7_drop1(out)
        out_block7=self.block7_relu(self.block7_conv2(out))
        #block 8
        out=self.block8_up_conv1(out_block7)
        out = torch.cat((out,out_block2), dim=1)
        out=self.block8_conv1(out)
        out=self.block8_drop1(out)
        out_block8=self.block8_relu(self.block8_conv2(out))
        #block 9
        out=self.block9_up_conv1(out_block8)
        out = torch.cat((out,out_block1), dim=1)
        out=self.block9_conv1(out)
        out=self.block9_drop1(out)
        out_block9=self.block9_conv2(out)
        return torch.sigmoid(out_block9)
def train(lr=0.01,epochs=1000,load=False):
    model=Unet()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optim=torch.optim.Adam(model.parameters(),lr=lr)
    loss=torch.nn.BCELoss()
    if load:
        model.load_state_dict(torch.load("processed data/model/TableUnetlossCrossed.pth"))
    allloss=[0]
    lossChanges=[0.0001,0.00001,0.000001,0.0000001,0.00000001]
    lr_counter=0
    high_loss_value=0
    print("*********************Training Started********************")
    for i in range(epochs):
        totalloss=0
        start = time.process_time()
        counter=0
        for j in tqdm.tqdm(os.listdir("processed data/images")):
            image = None
            if j.endswith('.jpg'):
                try:
                    image,target=getImageAndTarget(j)
                except:
                    continue
            if image==None or image.shape[0]==4:
                continue


            image = image.to(device)
            target = target.to(device)
            out=model(image.unsqueeze(0))
            loss_val=loss(out,target.unsqueeze(0))
            totalloss+=loss_val.item()
            optim.zero_grad()
            loss_val.backward()
            optim.step()
            counter+=1
        end=time.process_time()
        print("epoch:",i,"|loss:",totalloss,"| Change:",allloss[-1]-totalloss,"|Processed:",counter,"|Time taken:",end-start)
        difference=allloss[-1]-totalloss
        allloss.append(totalloss)
        if difference<0.0:
            print("saving ........")
            torch.save(model.state_dict(), "processed data/model/TableUnetlossCrossed512.pth")
            high_loss_value+=1
        elif difference>0 and high_loss_value>0:
            high_loss_value-=1
        if high_loss_value==2 and lr_counter<=len(lossChanges):
            print("************Changing Learning Rate***************")
            print("New Learning rate--------->",lossChanges[lr_counter])
            for param_group in optim.param_groups:
                param_group['lr'] = lossChanges[lr_counter]
                lr_counter+=1
            high_loss_value=0

        elif high_loss_value>2:
            print("------------------------------Training Ended Due to OverFitting-----------------")
            torch.save(model.state_dict(), "processed data/model/TableUnetEnd512.pth")
            break
        if i%10==0:
            print("***************** Saving Model   ****************")
            torch.save(model.state_dict(), "processed data/model/TableUnet512.pth")
    print("*************************Training SuceessFul************************")
    print("Saving Final State---->")
    torch.save(model.state_dict(), "model/TableFinal256.pth")
    print("Final State Saved as :/model/TableFinal256.pth")
train(epochs=1000,lr=0.001,load=False)
