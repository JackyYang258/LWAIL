import torch
import argparse
# from dataset import get_dataset, RepeatedDataset
# from get_args import get_git_diff, git_commit
# from NN import other contrastive objectives
# from advance_NN import *
import subprocess
from tqdm import tqdm
from datetime import datetime
import numpy as np
import random
import d4rl
import gym
import wandb
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam 

def get_git_diff():

    tmp = subprocess.run(['git', 'diff', '--exit-code'], capture_output=True)
    tmp2 = subprocess.run(['git', 'diff', '--cached', '--exit-code'], capture_output=True)
    return tmp.stdout.decode('ascii').strip() + tmp2.stdout.decode('ascii').strip()
    
def git_commit(runtime):

    tmp = subprocess.run(['git', 'commit', '-a', '-m', runtime], capture_output=True)
    return tmp.stdout.decode('ascii').strip()

device = torch.device('cuda:0')

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)    

def get_dataset(dataset, cut_len=10000000000000000000000, fill=False):
    original_len = dataset['observations'].shape[0]
    # print(np.array([dataset[i]["action"] for i in range(min(cut_len, len(dataset)))]))
    print("len:", original_len, dataset["observations"][0].shape, dataset["next_observations"][0].shape)
    states = torch.from_numpy(np.array(dataset['observations'][:cut_len])).double()
    actions = torch.from_numpy(np.array(dataset["actions"][:cut_len])).double()
    next_states = torch.from_numpy(np.array(dataset["next_observations"][:cut_len])).double()
    terminals = torch.zeros(min(original_len, cut_len)).double()
    terminals[-1] = 1
    for i in range(min(original_len, cut_len) - 1):
        if ((next_states[i] - states[i+1]) ** 2).sum() > 1e-6: 
            terminals[i] = 1
    # steps = torch.from_numpy(np.array(dataset["step"][:cut_len])).double()
    print("terminal sum:", terminals.sum())

    if fill:  
        states, actions, next_states = states.repeat((original_len - 1) // cut_len + 1, 1), actions.repeat((original_len - 1) // cut_len + 1, 1), next_states.repeat((original_len - 1) // cut_len + 1, 1)
        terminals = terminals.repeat((original_len - 1) // cut_len + 1)
    return states, actions, next_states, terminals

class RepeatedDataset:
    def __init__(self, datas, batch_size, start_with_random=True):
        self.datas = []
        for data in datas: # list of arrays with the same first dimension.
            self.datas.append(data.clone())
        self.counter, self.idx, self.batch_size = 0, torch.randperm(self.datas[0].shape[0]), batch_size
        if start_with_random:
            for _ in range(len(self.datas)):
                print("shape:", self.datas[_].shape)
                self.datas[_] = self.datas[_][self.idx]
    
    def __len__(self):
        return self.datas[0].shape[0] // self.batch_size    
    
    def getitem(self):
        if self.counter + self.batch_size > len(self.idx):
            self.counter, self.idx = 0, torch.randperm(self.datas[0].shape[0])
            for _ in range(len(self.datas)):
                self.datas[_] = self.datas[_][self.idx]
        ret = []
        for _ in range(len(self.datas)):
            ret.append(self.datas[_][self.counter:self.counter+self.batch_size])
        self.counter += self.batch_size
        """
        print(self.counter, self.counter+self.batch_size)
        
        for _ in range(len(self.datas)):
            print(self.datas[_][self.counter:self.counter+self.batch_size])
        """
        if len(self.datas) == 1: return ret[0]
        else: return ret

class ContrastiveEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        middle_size = 128
        self.net = nn.Sequential(
            nn.Linear(input_size, middle_size),
            nn.ReLU(),
            nn.Linear(middle_size, middle_size),
            nn.ReLU(),
            nn.Linear(middle_size, output_size))
    
    def forward(self, s):
        return self.net(s)

class Contrastive(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.feature_size = 32
        self.encoder = ContrastiveEncoder(input_size, self.feature_size)
        self.W = torch.nn.Parameter(torch.rand(self.feature_size, self.feature_size)) # note the "distance" is euclidean in embedding space; W does not have to be semi positive-definite
        
    def encode(self, x):
        return self.encoder(x)
        
    def forward(self, s1, s2):
        z1, z2 = self.encode(s1), self.encode(s2)
        logits = torch.matmul(z1, torch.matmul(self.W, z2.T))
        # print("logit shape:", logits.shape)
        #print("logits-before:", logits)
        logits -= torch.max(logits, 1)[0][:, None]
        #print("logits-after:", logits)
        return logits 

class Contrastive_PD(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.feature_size = 32
        self.encoder = ContrastiveEncoder(input_size, self.feature_size)
        self.W = torch.nn.Parameter(torch.rand(self.feature_size, self.feature_size)) # note the "distance" is euclidean in embedding space; W does not have to be semi positive-definite
        
    def encode(self, x):
        v = self.encoder(x)
        return v / torch.norm(v, dim=-1, keepdim=True)
        
    def forward(self, s1, s2):
        z1, z2 = self.encode(s1), self.encode(s2)
        # logits = torch.matmul(z1, torch.matmul(self.W, z2.T))
        #print("logits-before:", logits)
        W2 = torch.matmul(F.softplus(self.W), F.softplus(self.W.T))
        logits = torch.matmul(z1, torch.matmul(W2, z2.T))
        # print("logit shape:", logits.shape)
        #print("logits-before:", logits)
        logits -= torch.max(logits, 1)[0][:, None]
        #print("logits-after:", logits)
        return logits 


def get_args():
    parser =  argparse.ArgumentParser()
    parser.add_argument("--seed", help="seed",type=int, default=1234567)
    parser.add_argument("--env_name", help="env_name", type=str, default='walker2d-random-v2')
    parser.add_argument("--batch_size", help="BS", type=int, default=4096)
    parser.add_argument("--N", help="N", type=int, default=200)
    parser.add_argument("--eval", help="eval", type=int, default=0)
    parser.add_argument("--type", help="type", type=str, default="pd")
    args = parser.parse_args()
    return args

def train_contrastive_model(env_name, model_name, states_TA, next_states_TA):
    if model_name.find("_pd") != -1:
        model = Contrastive_PD(states_TA.shape[-1]).to(device).double()
    else: 
        model = Contrastive(states_TA.shape[-1]).to(device).double()
    
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters())
    train_loader = RepeatedDataset([states_TA.to(device).double(), next_states_TA.to(device).double()], 4096)
    for i in tqdm(range(N)):
        for batch in tqdm(range(len(train_loader))):
            states, next_states = train_loader.getitem()
            contrastive_logits = model(states, next_states)
            labels = torch.arange(contrastive_logits.shape[0]).long().to(device) # as similar as matrix I as possible
            loss = cross_entropy_loss(contrastive_logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            wandb.log({"CEloss": loss})
    torch.save(model, "model/"+env_name+"/"+model_name+".pt")
        
    return model
if __name__ == "__main__":
    args = get_args()    
    #data = torch.load("data/"+args.data_path+"/TA-"+args.data_name+".pt")
    
    # print("data type 0:", data)
    
    data = gym.make(args.env_name).get_dataset()
    
    #print("data type 1:", data)
    #exit(0)
    
    runtime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    # if len(get_git_diff()) > 0:
    #     git_commit(runtime+"_contrastive") 
        
    seed = args.seed
    if args.type == "normal": suffix = ""
    elif args.type == "pd": suffix = "_pd"
    else: raise NotImplementedError("Error!")
    
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) # when using multiple GPUs torch.cuda.manual_seed(seed)
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    
    if args.eval == 0:
        runtime = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        # if len(get_git_diff()) > 0:
        #     git_commit(runtime) 
        
        a = args.env_name
        wandb.init(entity="kaiyan3",project="project2_contrastive", name=runtime+"_"+str(args.seed)+"_"+a+"_contrastive")
        
        states_TA, actions_TA, next_states_TA, terminals_TA = get_dataset(data)
        non_terminal = torch.nonzero(terminals_TA == 0).view(-1)
        print("nc:", non_terminal)
        states_TA, next_states_TA = states_TA[non_terminal], next_states_TA[non_terminal]
        print("shape:", states_TA.shape)
        train_loader = RepeatedDataset([states_TA.to(device).double(), next_states_TA.to(device).double()], args.batch_size)
        FLAG = 0
        if args.type == "normal": model = Contrastive(states_TA.shape[-1]).to(device).double()
        elif args.type == "pd": model = Contrastive_PD(states_TA.shape[-1]).to(device).double()

        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(),)
        for i in tqdm(range(args.N)):
            for batch in tqdm(range(len(train_loader))):
                states, next_states = train_loader.getitem()
                if FLAG == 0:
                    contrastive_logits = model(states, next_states)
                    labels = torch.arange(contrastive_logits.shape[0]).long().to(device) # as similar as matrix I as possible
                    loss = cross_entropy_loss(contrastive_logits, labels)
                else:
                    contrastive_logits1 = model(states, next_states)
                    contrastive_logits2 = model(next_states, states)
                    labels = torch.arange(contrastive_logits1.shape[0]).long().to(device) # as similar as matrix I as possible
                    loss = cross_entropy_loss(contrastive_logits1, labels) + cross_entropy_loss(contrastive_logits2, labels)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                wandb.log({"CEloss": loss})
            # print(states.shape)
           
        torch.save(model, "model/"+args.env_name+"_contrastive"+str(suffix)+".pt")
    print("evaluation!")
    model = torch.load("model/"+args.env_name+"_contrastive"+str(suffix)+".pt")
    model.to(device)
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]}")
    
    if args.eval == 1:
        states_TA, actions_TA, next_states_TA, terminals_TA = get_dataset(data)
        states_TA, next_states_TA = states_TA.to(device).double(), next_states_TA.to(device).double()
    for i in range(100):
        # print("next state:", ((model.encode(states_TA[i].view(1, -1)) - model.encode(next_states_TA[i].view(1, -1))) ** 2).sum(dim=1))
        j = np.random.randint(states_TA.shape[0])
        
        if args.type in ["normal", "pd", "spd", "twinpd"]: d = lambda X, Y: ((model.encode(X) - model.encode(Y)) ** 2).sum(dim=1)
        elif args.type == "sphere": d = lambda X, Y: torch.arccos(model.encode(X) - model.encode(Y))
        else: raise NotImplementedError("Error!")
        print("next state:", d(states_TA[i].view(1, -1), next_states_TA[i].view(1, -1)))
        print("i", i, "j:", j, "random state:", d(states_TA[i].view(1, -1), states_TA[j].view(1, -1)))
