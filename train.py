import torch
from torch.utils.data import DataLoader
from torchmetrics import F1Score
from nn import GraphAttentionNetwork, GraphConvolutionalNetwork
import wandb

class Trainer:
    def __init__(self,model,optimizer,batchsize,epochs,lossfn,percentage,type="mlp"):
        self.model=model
        self.percentage=percentage
        self.type=type
        self.optimizer=optimizer
        self.batchsize=batchsize
        self.epochs=epochs
        self.lossfn=lossfn
        self.loadDataset()

    def train(self):
        it=0
        for epoch in range(self.epochs):
            for id,data in enumerate(self.trainDataloader):
                it+=1
                input,label=data[0].float(),data[1]
                patches=self.getPatches(input)
                patches=patches.reshape(64*self.batchsize,-1)
                label=label.reshape(64*self.batchsize,-1)
                
                patches.to(device)
                label.to(device)
                mod=1
                output=self.model(patches)
                
                loss=self.lossfn(output,label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    predicted=torch.argmax(output,dim=1)
                    label=torch.argmax(label,dim=1)
                    f1=F1Score(num_classes=13)
                    accuracy=f1(predicted,label)

                if it%mod==0:
                    print("Epoch:",epoch,"Iteration:",it,"Loss:",loss.data.mean(),"Training accuracy:",accuracy)
                    wandb.log({"epoch_train":epoch,"iteration_train":it,"loss_train":loss.data.mean(),"accuracy_train":accuracy})
            self.test() 
    def test(self):
        print("testing phase")
        count=0
        it=0
        test_loss=0
        test_accuracy=0
        with torch.no_grad():
            for id,data in enumerate(self.testDataloader):
                    it+=1
                    input,label=data[0].float(),data[1]
                    patches=self.getPatches(input)
                    patches=patches.reshape(64*self.batchsize,-1)
                    label=label.reshape(64*self.batchsize,-1)
                    
                    patches.to(device)
                    label.to(device)
                    mod=1
                    typesToChange=["resnetFrom0","resnetPretrainedFineTuneFc","resnetPretrainedFineTuneAll","mobilenetPretrainedFineTuneAll"]
                    if self.type in typesToChange:
                        # mod=1
                        patches =patches.reshape(64*self.batchsize,50,50,3)
                        patches=torch.einsum("abcd->adbc",patches)
                    
                    output=self.model(patches)
                    loss=self.lossfn(output,label)
                    predicted=torch.argmax(output,dim=1)
                    label=torch.argmax(label,dim=1)
                    f1=F1Score(num_classes=13)
                    accuracy=f1(predicted,label)
                    test_accuracy+=accuracy
                    test_loss+=loss.data.mean()
        print("mean_loss_test",test_loss/it,"mean_accuracy_test:",test_accuracy/it)
        wandb.log({"mean_loss_test":test_loss/it,"mean_accuracy_test":test_accuracy/it})

    def loadDataset(self):
        self.trainDataset=ChessDataset(type="train",percentage=self.percentage)
        self.testDataset=ChessDataset(type="test",percentage=self.percentage)
        self.trainDataloader = DataLoader(self.trainDataset, batch_size=self.batchsize, shuffle=True)
        self.testDataloader = DataLoader(self.testDataset, batch_size=self.batchsize, shuffle=True)

type="gnn"
wandb.init(project='NeuralNetworkProject', entity='bbooss97',name=type)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

if type=="gnn":
    model=GraphAttentionNetwork(13)


model.to(device)
wandb.watch(model)
batchsize=5
epochs=5
loss=torch.nn.CrossEntropyLoss()

optmimizer=torch.optim.Adam(model.parameters())
trainer=Trainer(model,optmimizer,batchsize,epochs,loss,type=type,percentage=1)

