from torch_geometric.datasets import PPI
import torch
from torch_geometric.loader import DataLoader
from nn import GraphAttentionNetwork, GraphConvolutionalNetwork
import wandb
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F

class Trainer:
    def __init__(self,model,optimizer,batchsize,epochs,lossfn,percentage,datasetName,type="mlp"):
        self.model=model
        self.percentage=percentage
        self.type=type
        self.optimizer=optimizer
        self.batchsize=batchsize
        self.epochs=epochs
        self.lossfn=lossfn
        self.loadDataset(datasetName)

    def train(self):
        it=0
        self.model.train()
        for epoch in range(self.epochs):
            it+=1    
            self.optimizer.zero_grad()        
            if self.datasetName=="ppi":
                out = model(self.dataset[0])
                loss=torch.nn.BCEWithLogitsLoss(reduction='mean')(out, self.dataset[0].y)
            else:
                out = model(self.dataset)
                loss = F.nll_loss(out[self.dataset.train_mask], self.dataset.y[self.dataset.train_mask])
            loss.backward()
            self.optimizer.step()
            self.test()
            #         print("Epoch:",epoch,"Iteration:",it,"Loss:",loss.data.mean(),"Training accuracy:",accuracy)
            #         wandb.log({"epoch_train":epoch,"iteration_train":it,"loss_train":loss.data.mean(),"accuracy_train":accuracy})
    def test(self):
        print("testing phase")
        model.eval()
        if self.datasetName=="ppi":
            pred = model(self.dataset[1])
            y=self.dataset[1].y
            acc=torch.nn.BCEWithLogitsLoss(reduction='mean')(pred, y)
            # correct = (pred == y).sum()
            # # acc=F1Score(121,0)(pred,y)
            # acc = int(correct) / int(self.dataset[1].y.shape[0])
        else:
            pred = model(self.dataset).argmax(dim=1)
            correct = (pred[self.dataset.test_mask] == self.dataset.y[self.dataset.test_mask]).sum()
            acc = int(correct) / int(self.dataset.test_mask.sum())
        print(f'Accuracy: {acc:.4f}')

        # print("mean_loss_test",test_loss/it,"mean_accuracy_test:",test_accuracy/it)
        # wandb.log({"mean_loss_test":test_loss/it,"mean_accuracy_test":test_accuracy/it})

    def loadDataset(self,datasetName="cora",type="train"):
        self.datasetName = datasetName
        #should be cora, citeseer and pubmed or ppi
        if datasetName=="ppi":
            self.dataset=PPI(root="./data/ppi")
        else:
            self.dataset=Planetoid(name=datasetName,root="./data/"+datasetName)
        print("name of dataset: {},number of classes: {},number of nodes features: {},number of graphs: {}".format(self.dataset,self.dataset.num_classes,self.dataset.num_node_features,len(self.dataset)))
        if datasetName!="ppi":
            self.dataset=self.dataset[0]
        
inOut={
    "cora":(1433,7),
    "citeseer":(3703,6),
    "pubmed":(500,3),
    "ppi":(50,121)
}
datasetName="ppi"
type="gat"
device = torch.device('cpu')
if type=="gcn":
    model=GraphConvolutionalNetwork(inOut[datasetName][0],13,inOut[datasetName][1])
elif type=="gat":
    model=GraphAttentionNetwork(inOut[datasetName][0],13,inOut[datasetName][1])

model.to(device)
# wandb.watch(model)
batchsize=5
epochs=1000
loss=torch.nn.CrossEntropyLoss()

optmimizer=torch.optim.Adam(model.parameters())
trainer=Trainer(model,optmimizer,batchsize,epochs,loss,type=type,percentage=1,datasetName=datasetName)
trainer.train()
