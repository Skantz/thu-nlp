import torch
import numpy as np

from custom_loss import binary_crossentropy_with_ranking
from sklearn.metrics import roc_auc_score

#EMBED_DIM = 10
#HIDDEN_DIM = 73
EARLY_STOP_VAL = 10**-2
EARLY_STOP_ACC = 0.99 #0.99

class SentRNN(torch.nn.Module):
    def __init__(self, embed_dim_in, embed_dim_out, seq_len, out_size=1, embeddings=None):
        #if embeddings == None:
        ##    raise NotImplementedError
        super(SentRNN, self).__init__()
        self.embed_dim_in = embed_dim_in
        self.embed_dim_out = embed_dim_out
        self.seq_len = seq_len

        self.encoder = torch.nn.Embedding(num_embeddings=embed_dim_in, embedding_dim=embed_dim_out, sparse=False, )
        self.recurrent = torch.nn.GRU(embed_dim_out, embed_dim_out, num_layers=1, batch_first=True, bias=True , dropout=0 , bidirectional=True)
        #self.recurrent = torch.nn.RNN(EMBED_DIM, 84, num_layers=2, batch_first=True, bias=False, dropout=0.5, bidirectional=True, nonlinearity='relu')
        self.fconnected = torch.nn.Linear(seq_len, out_size)
        self.fconnected1 = torch.nn.Linear(seq_len, 15)
        self.fconnected2 = torch.nn.Linear(15, 1)

        if embeddings:
            self.encoder.weight.data.copy_(embeddings)


        self.device = 'cpu'
        if torch.cuda.is_available():
            self = self.to('cuda')
            self.device = 'cuda'

        self.hidden_state = torch.zeros(2, 32, embed_dim_out).to(self.device)#('cuda')
        #self.hidden_state = torch.randn(self.hidden_state.shape).to(self.device)

    def forward(self, x, label, x_original_length=None):
        #self.hidden_state = torch.zeros(2, 32, 84).long().to('cuda')#('cuda')
        #print(x.shape)
        x = self.encoder(x.long())
        #print(x.shape)
        #print("x shape after embed", x.shape)
        x, hidden_state = self.recurrent(x.float(), self.hidden_state.float()) #self.hidden_state) #, self.hidden_state)
        #self.hidden_state = hidden_state
        #print(hidden_state.shape)
        #print(x.shape)
        x = x[:, :, -1]
        #x, _ = torch.max(x, dim=2)  
        x = self.fconnected(x)
        #x = self.fconnected1(x) #torch.transpose(x, 0, 1))
        #x = self.fconnected2(x)
        #x = torch.sigmoid(x)
        label[label < 0] = 0
        #label = torch.nn.functional.one_hot(label, num_classes=2)
        #x = torch.nn.functional.sigmoid(x.squeeze(0))
        #loss = torch.nn.binary_cross_entropy
        x = torch.sigmoid(x)
        #print(x_original_length)
        loss = torch.nn.functional.binary_cross_entropy(x.squeeze(1), label.float(), reduction="mean",)# ignore_index=x_original_length.item())
        #loss = binary_crossentropy_with_ranking(x.squeeze(1), label.float())
        #lp = torch.nn.functional.softmax(x, dim=1)
        self.hidden_state = self.hidden_state.detach()
        #self.recurrent.zero_grad()
        #x is not prob.., 
        #print(x)
        return loss, x

    def train(self, train_x, train_y, original_lengths=None):

        #optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        optimizer = torch.optim.Adam([
                {"params": self.encoder.parameters(), "lr": 0.01},
                {"params": self.recurrent.parameters(), "lr": 0.01},
                {"params": self.fconnected.parameters(), "lr": 0.01},
                ], lr=0.001, weight_decay=10**-3)
        
        #optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, nesterov=True)
        #train_x = train_x.unsqueeze(1)
        #train_y = train_y.unsqueeze(1) 
        #train_x = train_x.reshape(32, 1, -1)
        #train_y = train_y.reshape(32, 1, -1)
        train_x, val_x = train_x[:int(train_x.shape[0]*0.8)], train_x[int(0.8*train_x.shape[0]):]
        train_y, val_y = train_y[:int(train_y.shape[0]*0.8)], train_y[int(0.8*train_y.shape[0]):]
        #train_len, val_len = original_lengths[:int(train_y.shape[0]*0.8)], original_lengths[int(train_y.shape[0]*0.8):]
        zipped = list(zip(train_x, train_y,)) #train_len))
        valzip = list(zip(val_x, val_y, ))#val_len))
        dl = torch.utils.data.DataLoader(zipped,batch_size=32,shuffle=False, drop_last=True)
        vl = torch.utils.data.DataLoader(valzip,batch_size=32,shuffle=False, drop_last=True)
        iter = 0
        max_ep = 100

        checkpoint_name = "best_val_model.pth"
        checkpoint_accu = 0.0
        for epoch in range(max_ep):
            eploss = 0
            epcorr = 0
            epall  = 0
            preds = []
            labels_for_eval = []
            for (target, label) in dl:
                optimizer.zero_grad()
                #target = torch.nn.functional.one_hot(target.long(), EMBED_DIM).to(self.device)
                target = target.to(self.device)
                label  = label.to(self.device)
                loss, x = self.forward(target, label)# seq_len)
                loss.backward()
                optimizer.step()
                eploss += loss.item()
                x = x.squeeze(1)
                preds += x.cpu().tolist()
                labels_for_eval += label.cpu().tolist()
                x[x < 0.5]  = 0
                x[x >= 0.5] = 1
                epall += len(x)
                corr_t = (x == label)
                corr_t = corr_t[corr_t == True]
                epcorr += len(corr_t)
                if iter % 1 == 0:
                    pass
                    #print(target.shape)
                    #print(loss)
                    #print("inp shape", target.shape)
                iter += 1
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
            if epoch % 5 == 0:
                print(epoch, "/", max_ep, "eploss", eploss/len(dl))
                print("epacc:", epcorr/epall)
                minl = min(len(train_y), len(preds))
                train_y_transl = train_y.clone()
                train_y_transl[train_y_transl < 0] = 0

                roc = roc_auc_score(labels_for_eval[:minl], preds[:minl])
                print("train roc", roc)
            #print(x)
            if eploss/len(dl) < EARLY_STOP_VAL or epcorr/epall > EARLY_STOP_ACC:
                break

            eploss, epall, epcorr = 0, 0, 0
            preds = []
            for (target, label) in vl: # seq_len) in vl:
                #target = torch.nn.functional.one_hot(target.long(), EMBED_DIM).to(self.device)
                target = target.to(self.device)
                label  = label.to(self.device)
                with torch.no_grad():
                    _, x = self.forward(target, label)
                #eploss += loss.item()

                #print(x)
                x = x.squeeze(1)
                preds += x.cpu().tolist()
                x[x < 0.5]  = 0
                x[x >= 0.5] = 1
                epall += len(x)
                corr_t = (x == label)
                corr_t = corr_t[corr_t == True]
                epcorr += len(corr_t)
                #print("this should be constant", len(dl))
            #print(epoch, "/", max_ep, "val-loss", eploss/len(dl))
            valaccu = epcorr/epall
            minl = min(len(val_y), len(preds))
            roc = roc_auc_score(val_y[:minl], preds[:minl])
            if roc > checkpoint_accu:
                torch.save(self, checkpoint_name)
                checkpoint_accu = roc
            if epoch % 5 == 0:
                print("valaccu:", epcorr/epall)
                print("val roc", roc)



    def test(self, test_x, test_y=None):
        #train_y = train_y.reshape(32, 1, -1)
        if test_y == None:
            test_y = torch.zeros_like(test_x) #dummy
        zipped = list(zip(test_x, test_y))
        dl = torch.utils.data.DataLoader(zipped,batch_size=32,shuffle=False, drop_last=True)
        #preds = np.zeros(shape=(list(test_y.shape)))
        preds = []
        iter = 0
        for (target, label) in dl:
            #target = torch.nn.functional.one_hot(target.long(), EMBED_DIM).to(self.device)
            target = target.to(self.device)
            label  = label.to(self.device)
            with torch.no_grad():
                loss, x = self.forward(target, label)
                x = x.squeeze(1)
                #preds[iter*32:(iter + 1)*32] = x.cpu().numpy()
                x_test = x
                #reference, not copy 
                #x_test[x < 0.5] = -1
                #x_test[x >= 0.5] = 1
                #print(x_test)
                #print(label)
                #print(x_test == label)
                preds += x.cpu().tolist()
                #print("loss", loss)
            iter += 1
        return np.array(preds)
