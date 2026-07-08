import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


#Phase 2: Building the model

class CNN(nn.Module):
    def __init__(self): 
        super().__init__()
        self.hl_1=nn.Conv2d(1,32,3,padding=1) #input channels, output channels, kernel size, padding
        self.hl_2=nn.Conv2d(32,64,3,padding=1)
        self.pool=nn.MaxPool2d(2)
        self.hl_3=nn.Linear(3136,128)
        self.hl_4=nn.Linear(128,10)

    
    def forward(self,x):
        x=torch.relu(self.hl_1(x))
        x=self.pool(x)
        x=torch.relu(self.hl_2(x))
        x=self.pool(x)
        x=x.view(x.shape[0],-1)
        x=torch.relu(self.hl_3(x))
        x=self.hl_4(x)
        return x
    
if __name__=="__main__":
    #Phase 1: Data Preparation

    training= transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5),(0.5))
    ])

    testing= transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5),(0.5)) 
    ])

    train_set= torchvision.datasets.MNIST(root='./root',train =True,download=True,transform=training)
    test_set=torchvision.datasets.MNIST(root='./root',train=False,download=True,transform=testing)
    train_loader=torch.utils.data.DataLoader(train_set,batch_size=64,shuffle=True)
    test_loader=torch.utils.data.DataLoader(test_set,batch_size=64,shuffle=False)


    #Phase 3: Initializing the model, defining the loss function and the optimizer

    model=CNN()
    criterion= nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.01)

    #Phase 4: Training the model

    epochs=5
    for epoch in range(epochs):
        model.train()
        running_loss=0.0
        for batch in train_loader:
            optimizer.zero_grad()   
            images,labels=batch
            outputs=model(images)
            loss=criterion(outputs,labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch: {epoch+1} and loss: {running_loss/len(train_loader)}")

        checkpoint={
            "model_state": model.state_dict(),
            "optim_state":optimizer.state_dict(),
            "epoch_num": epoch,
            "running_loss":running_loss
        }
        torch.save(checkpoint,"./model/chkpoint.pth")

        #Phase 5: Evaluating the model

        model.eval()
        testloss=0.0
        correct=0
        total=0
        with torch.no_grad():
            for batch in test_loader:
                images,labels=batch
                output=model(images)
                testloss+=criterion(output,labels).item()   
                predicted=torch.argmax(output,dim=1)
                correct+= (predicted==labels).sum().item()
                total+=labels.size(0)
            print(f"Test Loss: {testloss/len(test_loader)}")

        print(f"Accuracy: {correct/total}")

