import logging
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader


class Trainer(object):
    def __init__(self, model, memory, device, batch_size):
        """
        Train the trainable model of a policy
        """
        self.model = model
        self.device = device
        self.criterion = nn.MSELoss().to(device)
        self.memory = memory
        self.data_loader = None
        self.batch_size = batch_size
        self.optimizer = None

    def set_learning_rate(self, learning_rate):
        logging.info('Current learning rate: %f', learning_rate)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)

    def optimize_epoch(self, num_epochs):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        # print("mem size",self.memory.__len__())
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        average_epoch_loss = 0
        for epoch in range(num_epochs):
            epoch_loss = 0
            for data in self.data_loader:

                inputs, values = data
                inputs = Variable(inputs)
                values = Variable(values)
                # print(inputs[0 , : , :])
                # print(values)
                normalized_inputs = nn.functional.normalize(inputs, p=2.0, dim=2, eps=1e-12, out=None)
                # normalized_values = nn.functional.normalize(values, p=2.0, dim=1, eps=1e-12, out=None)
                # print("normalized ::::::::::: ",normalized_inputs)
                self.optimizer.zero_grad()  
                ####### for gcn
                # print("shape of normalized inputs  :" ,normalized_inputs.size())
                # print("normalized inputs : ",normalized_inputs[0 , : , :])
                # print("shape of normalized inputs before passing to gcn ", normalized_inputs.size())
                #######
                outputs = self.model(inputs)
                # print("outputs : ",outputs)
                loss = self.criterion(outputs, values)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.data.item()

            average_epoch_loss = epoch_loss / len(self.memory)
            # print("epoch loss : ",epoch_loss)
            # print("len self memory : ",len(self.memory))
            logging.debug('Average loss in epoch %d: %.2E', epoch, average_epoch_loss)

        return average_epoch_loss

    def optimize_batch(self, num_batches):
        if self.optimizer is None:
            raise ValueError('Learning rate is not set!')
        if self.data_loader is None:
            self.data_loader = DataLoader(self.memory, self.batch_size, shuffle=True)
        losses = 0
        for _ in range(num_batches):
            inputs, values = next(iter(self.data_loader))
            inputs = Variable(inputs)
            values = Variable(values)
            normalized_inputs = nn.functional.normalize(inputs, p=2.0, dim=1, eps=1e-12, out=None)
            self.optimizer.zero_grad()
            outputs = self.model(normalized_inputs)
            loss = self.criterion(outputs, values)
            loss.backward()
            self.optimizer.step()
            losses += loss.data.item()

        average_loss = losses / num_batches
        logging.debug('Average loss : %.2E', average_loss)

        return average_loss



def create_sequence_from_batch(data):
    # batch size new = batch size old / sequence len
    # batch size old should be multiple of sequence len
    dataloader  = DataLoader(self.memory, self.batch_size, shuffle=False)
    for data in dataloader:
        s1 , s2 , p1 , p2 = data
        s1 = Variable(s1)
        s2 = Variable(s2)
        p1 = Variable(p1)
        p2 = Variable(p2)

        s1 = s1.reshape(-1,sequence_len , s1.shape[1])
        s2 = s1.reshape(-1,sequence_len , s1.shape[1])
        p1 = p1.reshape(-1,sequence_len , p1.shape[1])
        p2 = p2.reshape(-1,sequence_len , p2.shape[1])

        print(s1.shape)
        break
    