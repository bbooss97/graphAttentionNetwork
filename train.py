from torch_geometric.datasets import PPI
import torch
from torch_geometric.loader import DataLoader
from nn import GraphAttentionNetwork, GraphConvolutionalNetwork, Mlp
import wandb
from torch_geometric.datasets import Planetoid
import torch.nn.functional as F


class Trainer:
    def __init__(self,model,optimizer,epochs,datasetName):
        self.model=model
        self.optimizer=optimizer
        self.epochs=epochs
        self.loadDataset(datasetName)

    #training loop
    def train(self):
        it=0
        self.model.train()
        for epoch in range(self.epochs):
            it+=1    
            self.optimizer.zero_grad()    
            # if the database is ppi then the task is the multilabeling classification 
            #otherwise is node classification
            if self.datasetName=="ppi":
                out = model(self.dataset[0])
                #the loss in this case is the binary cross entropy with logits loss
                #the logits are numbers between -inf and inf not yet normalized for every class
                #in this case i have multiple graphs so i just pass the first graph to train on
                #i could improve it to train on all the others like in a kfold cross validation
                loss=torch.nn.BCEWithLogitsLoss(reduction='mean')(out, self.dataset[0].y)
            else:
                out = model(self.dataset)
                #the loss here is the negative log likelihood 
                #i mask the graph since i have just one graph to train and then i evaluate with a different mask
                #i softmax the output to apply the nll loss
                out=F.log_softmax(out, dim=1)
                loss = F.nll_loss(out[self.dataset.train_mask], self.dataset.y[self.dataset.train_mask])
            #backpropagation
            loss.backward()
            self.optimizer.step()
            print("mean_loss_train",loss.data.mean())
            wandb.log({"mean_loss_train":loss.data.mean()})
            #i evaluate the model on the test part of the graph
            self.test()

    def test(self):
        print("testing phase")
        model.eval()
        #if the dataset is ppi then the task is the multilabeling classification
        #so i get the output and then i apply the sigmoid to get the probability of each class
        #i threshold the probability to get the class for each label
        #then i compare the output with the ground truth
        with torch.no_grad():
            if self.datasetName=="ppi":
                #i evaluate the model on the second graph of the dataset not the first of the trainign
                pred = model(self.dataset[1])
                y=self.dataset[1].y
                loss=torch.nn.BCEWithLogitsLoss(reduction='mean')(pred, y)
                pred=F.sigmoid(pred)
                pred=torch.where(pred>0.5,1,0)
                correct=(pred==y).sum()
                acc=correct/len(y)
                # correct = (pred == y).sum()
                # # acc=F1Score(121,0)(pred,y)
                # acc = int(correct) / int(self.dataset[1].y.shape[0])
            else:
                #if the dataset is cora, citeseer or pubmed then the task is the node classification
                #so i have the nll loss at the end on the log softmax output
                pred = model(self.dataset)
                #i softmax the prdictions and i apply the negative log likelihood to the test graph (masked)and the argmax to get the class
                pred=F.log_softmax(pred, dim=1)
                loss = F.nll_loss(pred[self.dataset.test_mask], self.dataset.y[self.dataset.test_mask])
                #calculate accuracy
                pred=pred.argmax(dim=1)
                correct = (pred[self.dataset.test_mask] == self.dataset.y[self.dataset.test_mask]).sum()
                acc = int(correct) / int(self.dataset.test_mask.sum())

        #log the results
        print("mean_loss_test",loss.data.mean(),"mean_accuracy_test:",acc)
        wandb.log({"mean_loss_test":loss.data.mean(),"mean_accuracy_test":acc})


    #this function loads the dataset based on the name 
    def loadDataset(self,datasetName="cora"):
        self.datasetName = datasetName
        #should be cora, citeseer and pubmed or ppi
        if datasetName=="ppi":
            self.dataset=PPI(root="./data/ppi")
        else:
            self.dataset=Planetoid(name=datasetName,root="./data/"+datasetName)
        #print the dataset generalities
        print("name of dataset: {},number of classes: {},number of nodes features: {},number of graphs: {}".format(self.dataset,self.dataset.num_classes,self.dataset.num_node_features,len(self.dataset)))
        if datasetName!="ppi":
            self.dataset=self.dataset[0]


if __name__ == "__main__":
    # this is the number of features of the input and the output dimension that the model will have based on the dataset
    inOut={
        "cora":(1433,7),
        "citeseer":(3703,6),
        "pubmed":(500,3),
        "ppi":(50,121)
    }
    # i set the name of the dataset here and the type of model
    datasetName="pubmed"
    type="gcn"
    device = torch.device('cpu')
    #i create the model based on the settings of dataset name and type
    if type=="gcn":
        model=GraphConvolutionalNetwork(inOut[datasetName][0],13,inOut[datasetName][1])
    elif type=="gat":
        model=GraphAttentionNetwork(inOut[datasetName][0],13,inOut[datasetName][1])
    elif type=="mlp":
        model=Mlp(inOut[datasetName][0],50,inOut[datasetName][1])
    # i create the the wandb project and the run    
    name=type+" "+datasetName
    run=wandb.init(project='graphAttentionNetwork', entity='bbooss97',name=name)
    model.to(device)
    run.watch(model)
    #define the settings for the training
    epochs=200
    optmimizer=torch.optim.Adam(model.parameters())
    #create the trainer and start the training process
    trainer=Trainer(model,optmimizer,epochs,datasetName=datasetName)
    trainer.train()
