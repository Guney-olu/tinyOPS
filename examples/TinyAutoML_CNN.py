"""
CNN training on image dataset using tinygrad
"""

from typing import List, Callable
from tinygrad import Tensor, TinyJit, nn, GlobalCounters
from tinygrad.helpers import getenv, colored
from tqdm import trange
import argparse
import sys
from DataLoader import load_custom_data


def load_data(train_data_dir,test_data_dir):
    X_train,Y_train = load_custom_data(train_data_dir)
    X_test,Y_test = load_custom_data(test_data_dir)
    return X_train,Y_train,X_test,Y_test

class Model:
    def __init__(self):
        self.layers: List[Callable[[Tensor],Tensor]]=[
            nn.Conv2d(1,32,5), Tensor.relu,
            nn.Conv2d(32,32,5), Tensor.relu,
            nn.BatchNorm2d(32), Tensor.max_pool2d,
            nn.Conv2d(32,64,3), Tensor.relu,
            nn.Conv2d(64,64,3), Tensor.relu,
            nn.BatchNorm2d(64), Tensor.max_pool2d,
            lambda x: x.flatten(1), nn.Linear(576,144)
        ]
    def __call__(self, x:Tensor) -> Tensor:
        return x.sequential(self.layers)

def model_run(epochs,X_train, Y_train, X_test, Y_test):
    X_train,X_test = X_train.float(), X_test.float()
    model = Model()
    opt = nn.optim.Adam(nn.state.get_parameters(model))
    
    @TinyJit
    def train_step()-> Tensor:
        with Tensor.train():
            opt.zero_grad()
            samples = Tensor.randint(512, high=X_train.shape[0])
            loss = model(X_train[samples]).sparse_categorical_crossentropy(Y_train[samples]).backward()
            opt.step()
            return loss
        
    @TinyJit
    def get_test_acc() -> Tensor:
      return (model(X_test).argmax(axis=1) == Y_test).mean() * 100
  
    test_acc = float('nan')
    for i in (t := trange(epochs)):
        GlobalCounters.reset()
        loss = train_step()
        if i % 10 == 9:
            test_acc = get_test_acc().item()
        t.set_description(f"loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%")

    if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
        if test_acc >= target and test_acc != 100.0:
            print(colored(f"{test_acc=} >= {target}", "green"))
        else:
            raise ValueError(colored(f"{test_acc=} < {target}", "red"))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog = "TinyCNNTest",
        description="Running tiny cnn code",
        epilog = "Help"
    )
    parser.add_argument('-e','--epoch')
    parser.add_argument('train')
    parser.add_argument("test")
    args = parser.parse_args()
    X_train, Y_train, X_test, Y_test = load_data(train_data_dir=args.train,test_data_dir=args.test)
    model_run(int(args.epoch),X_train,Y_train,X_test,Y_test)



