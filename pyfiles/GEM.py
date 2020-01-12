import numpy as np
import torch
import pdb
import quadprog


cuda_available = False
if torch.cuda.is_available():
    cuda_available = True

class GEMLearning(torch.nn.Module):
    def __init__(self, **kwargs):
        super(GEMLearning, self).__init__()
        self.net = kwargs['net']
        self.tasks = kwargs['tasks']
        self.optim = kwargs['optim']
        self.criterion = kwargs['criterion']
        self.mem_size = kwargs['mem_size']
        #self.num_input = kwargs['num_input']
        self.traindata_len = kwargs['traindata_len']
        self.testdata_len = kwargs['testdata_len']
        self.batch_size = kwargs['batch_size']
        
        # Initiallize Episodic Memory
        self.ep_mem = torch.FloatTensor(self.tasks, self.mem_size, 28*28)
        self.ep_labels = torch.LongTensor(self.tasks, self.mem_size)
        if cuda_available:
            self.ep_mem = self.ep_mem.cuda()
            self.ep_labels = self.ep_labels.cuda()

        # Save each parameters' number of elements(numels)
        self.grad_numels = []
        for params in self.parameters():
            self.grad_numels.append(params.data.numel())
        print(self.grad_numels)
        # Make matrix for gradient w.r.t. past tasks
        self.G = torch.zeros((sum(self.grad_numels), self.tasks))
        if cuda_available:
            self.G = self.G.cuda()

        # Make matrix for accuracy w.r.t. past tasks
        self.R = torch.zeros((self.tasks, self.tasks))
        if cuda_available:
            self.R = self.R.cuda()

        #msg = "Optimizer: {}\nCriterion: {}\nEpisodic Memory Size: {}\n"%(self.optim, self.criterion, self.mem_size)
        print(self.optim)
        print(self.criterion)
        print("Memory size: ", self.mem_size)
        #self.log_file.write(msg)
        
    def train(self, data_loader, task):
        self.cur_task = task
        running_loss = 0.0
        input_stack = torch.zeros((self.traindata_len, 28*28))
        label_stack = torch.zeros((self.traindata_len))
        if cuda_available:
            input_stack = input_stack.cuda()
            label_stack = label_stack.cuda()
        
        
        for i, data in enumerate(data_loader):
            #print(data)
            x, y = data
            #input_stack = np.vstack((input_stack, x))
            #label_stack = np.hstack((label_stack, y))
            input_stack[i*self.batch_size: (i+1)*self.batch_size] = x.clone()
            label_stack[i*self.batch_size: (i+1)*self.batch_size] = y.clone()
            if cuda_available:
                x = x.cuda()
                y = y.cuda()
                
            
            if self.cur_task > 0:
                #pdb.set_trace()
                # Compute gradient w.r.t. past tasks by using episodic memory
                for k in range(0, self.cur_task):
                    self.zero_grad()
                    pred_ = self.net(self.ep_mem[k])
                    pred_[:, : k * 10] = -10e10
                    pred_[:, (k+1) * 10:] = -10e10
                    
                    pred_ = pred_[:, k*10: (k+1)*10]
                    
                    label_ = self.ep_labels[k]
                    loss_ = self.criterion(pred_, label_)
                    loss_.backward()
        
                    # Copy parameters into Matrix "G"
                    j = 0
                    for params in self.parameters():
                        if params is not None:
                            if j == 0:
                                stpt = 0
                            else:
                                stpt = sum(self.grad_numels[:j])
            
                            endpt = sum(self.grad_numels[:j+1])
                            self.G[stpt:endpt, k].data.copy_(params.grad.view(self.G[stpt:endpt, k].data.size()))
                            j += 1
                    
            self.zero_grad()
            
            # Compute gradient w.r.t. current continuum
            pred = self.net(x)#[:, self.cur_task * 10 : (self.cur_task + 1) * 10]
            pred[:, : self.cur_task * 10] = -10e10
            pred[:, (self.cur_task+1) * 10:] = -10e10
                    
            pred = pred[:, self.cur_task*10: (self.cur_task+1)*10]
            loss = self.criterion(pred, y)
            loss.backward()
            running_loss += loss.detach().item()
            
            if i % 100 == 99:
                msg = '[%d\t%d] AVG. loss: %.3f\n'% (task+1, i+1, running_loss/(i*5))
                print(msg)
                #self.log_file.write(msg)
                running_loss = 0.0
            
            if self.cur_task > 0:
                grad = []
                for param in self.net.parameters():
                    grad.append(param.detach().view(-1))
                
                grad = torch.cat(grad)
                self.G[:, self.cur_task].data.copy_(grad)
                
                # Solve Quadratic Problem 
                prod = torch.mm(self.G[:, self.cur_task].unsqueeze(0), self.G[:, :self.cur_task])
                if(prod < 0).sum() != 0: # There are some violations: do projection
                    mem_grad_np = self.G[:, :self.cur_task+1].cpu().t().double().numpy()
                    curtask_grad_np = self.G[:, self.cur_task].unsqueeze(1).cpu().contiguous().view(-1).double().numpy()
                    
                    t = mem_grad_np.shape[0]
                    P = np.dot(mem_grad_np, np.transpose(mem_grad_np))
                    P = 0.5 * (P + np.transpose(P)) + np.eye(t) * 1e-3#eps
                    q = np.dot(mem_grad_np, curtask_grad_np) * (-1)
                    G = np.eye(t)
                    h = np.zeros(t) + 0.5 #margin: hyperparameter 
                    v = quadprog.solve_qp(P, q, G, h)[0]
                    x = np.dot(v, mem_grad_np) + curtask_grad_np
                    #print(torch.Tensor(x).shape)
                    #print(self.G[:, self.cur_task].shape)
                    self.G[:, self.cur_task].copy_(torch.Tensor(x))
                    #grad = torch.Tensor(x).view(-1).detach().clone()
    
                    # Copy gradients into params
                    j = 0
                    for params in self.parameters():
                        if params is not None:
                            if j == 0:
                                stpt = 0
                            else:
                                stpt = sum(self.grad_numels[:j])
        
                            endpt = sum(self.grad_numels[:j+1])
                            
                            copy_object_grad = self.G[stpt:endpt, self.cur_task].contiguous().view(params.grad.data.size())
                            params.grad.data.copy_(copy_object_grad)
                            j += 1
            
            self.optim.step()
            
        perm = torch.randperm(self.traindata_len)
        perm = perm[:self.mem_size]
        self.ep_mem[self.cur_task] = input_stack[perm].detach().clone().float()
        self.ep_labels[self.cur_task] = label_stack[perm].detach().clone()

    def eval(self, data_loader, task):
        total = 0
        correct = 0
        self.net.eval()
        for i, data in enumerate(data_loader):
            x, y = data
            if cuda_available:
                x = x.cuda()
                y = y.cuda()
                
            output = self.net(x)[:, task * 10: (task+1) * 10]
            _, predicted = torch.max(output, dim=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
            self.R[self.cur_task][task] = 100 * correct / total
        
    '''
    def file_close(self):
        self.log_file.close()
    '''