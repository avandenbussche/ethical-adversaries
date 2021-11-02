import numpy as np
import torch
import torch.optim as optim
from attack_model import \
    transform_dataset, \
    transform_dataset_census, \
    transform_dataset_credit, \
    attack_keras_model, \
    protected_attributes_for_optimization, \
    protected_attributes_for_comparison, \
    protected_attributes_all, \
    protected_attributes_all_indices_dict
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import trange
from tqdm.notebook import tnrange
from torch.utils.data.dataset import ConcatDataset

from sklearn import preprocessing
import pandas as pd
import os
import argparse
import logging
from torch.autograd import Function
import matplotlib.pyplot as plt
from bayesian_model import BayesianModel as bm
from pycm import ConfusionMatrix

from secml.array.c_array import CArray

from Differential_Fairness.differential_fairness import computeSmoothedEDF

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.DEBUG)


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class Net(nn.Module):

    def __init__(self, input_shape, grl_lambda=100):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self._grl_lambda = grl_lambda
        self.fc1 = nn.Linear(input_shape, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 1)
        if self._grl_lambda != 0:
            self.grl = GradientReversal(grl_lambda)
            self.fc5 = nn.Linear(32, 2)
        # self.grl = GradientReversal(100)

    def forward(self, x):
        hidden = self.fc1(x)
        hidden = F.relu(hidden)
        hidden = F.dropout(hidden, 0.1)

        y = self.fc4(hidden)
        # y = F.dropout(y, 0.1)

        if self._grl_lambda != 0:
            s = self.grl(hidden)
            s = self.fc5(s)
            # s = F.sigmoid(s)
            # s = F.dropout(s, 0.1)
            return y, s
        else:
            return y


def get_metrics(results, args, threshold, fraction):
    "Create the metrics from an output df."

    # Calculate biases after training
    dem_parity = abs(
        bm(results).P(pred=lambda x: x > threshold).given(race=0)
        - bm(results).P(pred=lambda x: x > threshold).given(
            race=1))

    eq_op = abs(
        bm(results).P(pred=lambda x: x > threshold).given(race=0, compas=True)
        - bm(results).P(pred=lambda x: x > threshold).given(race=1, compas=True))

    dem_parity_ratio = abs(
        bm(results).P(pred=lambda x: x > threshold).given(race=0)
        / bm(results).P(pred=lambda x: x > threshold).given(
            race=1))

    diff_fair_optimized = computeSmoothedEDF(results[protected_attributes_for_optimization].astype(int).values, (results['pred'] > threshold).astype(int).values)

    cm = ConfusionMatrix(actual_vector=(results['true'] == True).values,
                         predict_vector=(results['pred'] > threshold).values)
    if args.dataset == 'compas':
        cm_high_risk = ConfusionMatrix(actual_vector=(results['compas'] > 8).values,
                             predict_vector=(results['pred'] > 8).values)

        result = {"DP": dem_parity,
                  "EO": eq_op,
                  "DP ratio": dem_parity_ratio,
                  "acc": cm.Overall_ACC,
                  "acc_ci_min": cm.CI95[0],
                  "acc_ci_max": cm.CI95[1],
                  "f1": cm.F1_Macro,
                  "acc_high_risk": cm_high_risk.Overall_ACC,
                  "acc_ci_min_high_risk": cm_high_risk.CI95[0],
                  "acc_ci_max_high_risk": cm_high_risk.CI95[1],
                  "f1_high_risk": cm_high_risk.F1_Macro,
                  "adversarial_fraction": fraction,
                  "DF (optimized: {})".format(protected_attributes_for_optimization): diff_fair_optimized
                  }

        for s in protected_attributes_for_comparison:
            diff_fair_s = computeSmoothedEDF(results[s].astype(int).values, (results['pred'] > threshold).astype(int).values)
            key = "DF (comparison: {})".format(s)
            result[key] = diff_fair_s

    else:
        result = {"DP": dem_parity,
                  "EO": eq_op,
                  "DP ratio": dem_parity_ratio,
                  "acc": cm.Overall_ACC,
                  "acc_ci_min": cm.CI95[0],
                  "acc_ci_max": cm.CI95[1],
                  "f1": cm.F1_Macro,
                  "adversarial_fraction": fraction
                  }

    return result

def train_and_evaluate(train_loader: DataLoader,
                       val_loader: DataLoader,
                       test_loader: DataLoader,
                       device,
                       args,
                       input_shape,
                       grl_lambda=None,
                       model=None):
    """

    :param train_loader: Pytorch-like DataLoader with training data.
    :param val_loader: Pytorch-like DataLoader with validation data.
    :param test_loader: Pytorch-like DataLoader with testing data.
    :param device: The target device for the training.
    :return: A tuple: (trained Pytorch-like model, dataframe with results on test set)
    """

    torch.manual_seed(0)

    grl_lambda = grl_lambda if grl_lambda is not None else args.grl_lambda

    if args.reset_attack or model is None:
        # Redefine the model
        model = Net(input_shape=input_shape, grl_lambda=grl_lambda).to(device)

    criterion = nn.MSELoss().to(device)
    criterion_bias = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = ReduceLROnPlateau(optimizer, threshold=0.3, cooldown=5)

    training_losses = []
    validation_losses = []

    t_prog = trange(args.epochs, desc='Training neural network', leave=False, position=1, mininterval=5)
    # t_prog = trange(50)

    for epoch in t_prog:
        model.train()

        batch_losses = []
        for x_batch, y_batch, _, s_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            s_batch = s_batch.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            if grl_lambda is not None and grl_lambda != 0:
                outputs, outputs_protected = model(x_batch)
                loss = criterion(outputs, y_batch) + criterion_bias(outputs_protected, s_batch.argmax(dim=1))
            else:
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        training_loss = np.mean(batch_losses)
        training_losses.append(training_loss)

        with torch.no_grad():
            val_losses = []
            for x_val, y_val, _, s_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                s_val = s_val.to(device)
                model.eval()
                if grl_lambda is not None and grl_lambda != 0:
                    yhat, s_hat = model(x_val)
                    val_loss = (criterion(y_val, yhat) + criterion_bias(s_val, s_hat.argmax(dim=1))).item()
                else:
                    yhat = model(x_val)
                    val_loss = criterion(y_val, yhat).item()
                val_losses.append(val_loss)
            validation_loss = np.mean(val_losses)
            validation_losses.append(validation_loss)

            scheduler.step(val_loss)

        t_prog.set_postfix({"epoch": epoch, "training_loss": training_loss,
                            "validation_loss": validation_loss}, refresh=False)  # print last metrics

    if args.show_graphs:
        plt.plot(range(len(training_losses)), training_losses, label="Training Loss")
        plt.plot(range(len(validation_losses)), validation_losses, label="Validation Loss")
        # plt.scatter(x_tensor, y_out.detach().numpy())
        plt.title('Loss vs Epoch')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()
    with torch.no_grad():
        test_losses = []
        test_results = []
        for x_test, y_test, ytrue, s_true in test_loader:
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            s_true = s_true.to(device)
            model.eval()
            if grl_lambda is not None and grl_lambda != 0:
                yhat, s_hat = model(x_test)
                test_loss = (criterion(y_test, yhat) + criterion_bias(s_true, s_hat.argmax(dim=1))).item()
                test_losses.append(val_loss)
                test_results.append({"y_hat": yhat, "y_true": ytrue, "y_compas": y_test, "s": s_true, "s_hat": s_hat, "x": x_test})
            else:
                yhat = model(x_test)
                test_loss = (criterion(y_test, yhat)).item()
                test_losses.append(val_loss)
                test_results.append({"y_hat": yhat, "y_true": ytrue, "y_compas": y_test, "s": s_true, "x": x_test})

        # print({"Test loss": np.mean(test_losses)})

    results = test_results[0]['y_hat']
    outcome = test_results[0]['y_true']
    compas = test_results[0]['y_compas']
    protected_results = test_results[0]['s']
    x = test_results[0]['x']
    if grl_lambda is not None and grl_lambda != 0:
        protected = test_results[0]['s_hat']
    for r in test_results[1:]:
        results = torch.cat((results, r['y_hat']))
        outcome = torch.cat((outcome, r['y_true']))
        compas = torch.cat((compas, r['y_compas']))
        protected_results = torch.cat((protected_results, r['s']))
        x = torch.cat((x, r['x']))
        if grl_lambda is not None and grl_lambda != 0:
            protected = torch.cat((protected, r['s_hat']))

    # print("Shape of x: {}".format(x.shape))
    # print("Shape of protected_results: {}".format(protected_results.shape))
    # print("First row of x: {}".format(x[0]))

    df = pd.DataFrame(data=results.cpu().numpy(), columns=['pred'])

    df['true'] = outcome.cpu().numpy()
    df['compas'] = compas.cpu().numpy()
    for index, protected_attribute in enumerate(protected_attributes_for_optimization):
        df[protected_attribute] = protected_results.cpu().numpy()[:, index]
    for unprotected_attribute in set(protected_attributes_all).difference(set(protected_attributes_for_optimization)):
        df[unprotected_attribute] = x.cpu().numpy()[:, protected_attributes_all_indices_dict[unprotected_attribute]]
    if grl_lambda is not None and grl_lambda != 0:
        for index, protected_attribute in enumerate(protected_attributes_for_optimization):
            df[protected_attribute+"_hat"] = protected.cpu().numpy()[:, index]

    return model, df


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.debug("using device {} for pytorch.".format(device))

    # Make sure entire df is printed
    pd.set_option('display.max_colwidth', -1)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    if args.dataset == "compas":
        df = pd.read_csv(os.path.join("..", "data", "csv", "scikit",
                                      "compas_recidive_two_years_sanitize_age_category_jail_time_decile_score.csv"))
        df_binary, Y, S, Y_true = transform_dataset(df)
        print("#")
        print("#")
        print("#")
        print("ALL PROTECTED ATTRIBUTES")
        print("#")
        print("#")
        print("#")
        print(S)
        print("#")
        print("#")
        print("#")
        print("Optimized protected attributes: {}".format(protected_attributes_for_optimization))
        print("Compared protected attributes: {}".format(protected_attributes_for_comparison))
        print(" ")
        print(" ")
        print(" ")
        Y = Y.to_numpy()
        l_tensor = torch.tensor(Y_true.to_numpy().reshape(-1, 1).astype(np.float32))
    elif args.dataset == "adult":
        ##load the census income data set instead of the COMPAS one
        df = pd.read_csv(os.path.join("..", "data", "csv", "scikit", "adult.csv"))
        df_binary, Y, S, _ = transform_dataset_census(df)
        l_tensor = torch.tensor(Y.reshape(-1, 1).astype(np.float32))
    elif args.dataset == "german":
        ##load the census income data set instead of the COMPAS one
        df = pd.read_csv(os.path.join("..", "data", "csv", "scikit", "german.data"), header=None, sep="\s")
        df_binary, Y, S, _ = transform_dataset_credit(df)
        l_tensor = torch.tensor(Y.reshape(-1, 1).astype(np.float32))
    else:
        raise ValueError(
            "The value given to the --dataset parameter is not valid; try --dataset=compas or --dataset=adult")

    print("#")
    print("#")
    print("#")
    print("MEAN")
    print("#")
    print("#")
    print("#")
    print(np.mean(Y))
    print(" ")
    print(" ")
    print(" ")

    x_tensor = torch.tensor(df_binary.to_numpy().astype(np.float32))
    y_tensor = torch.tensor(Y.reshape(-1, 1).astype(np.float32))
    s_tensor = torch.tensor(preprocessing.OneHotEncoder().fit_transform(np.array(S).reshape(-1, len(protected_attributes_for_optimization))).toarray())

    dataset = TensorDataset(x_tensor, y_tensor, l_tensor, s_tensor)  # dataset = CustomDataset(x_tensor, y_tensor)

    base_size = len(dataset) // 10
    split = [7 * base_size, 1 * base_size, len(dataset) - 8 * base_size]  # Train, validation, test

    train_dataset, val_dataset, test_dataset = random_split(dataset, split)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size)

    x_train_tensor = train_dataset[:][0]
    y_train_tensor = train_dataset[:][1]
    l_train_tensor = train_dataset[:][2]
    s_train_tensor = train_dataset[:][3]

    global_results = []

    # get the classification threshold, we use the same scale for compas so 4 instead of 0.5
    threshold = 4 if args.dataset == 'compas' else 0.5

    _, results = train_and_evaluate(train_loader, val_loader, test_loader, device, args, input_shape=x_tensor.shape[1],
                                    grl_lambda=0)

    print("#")
    print("#")
    print("#")
    print("RESULTS")
    print("#")
    print("#")
    print("#")
    print(results)
    print(" ")
    print(" ")
    print(" ")

    result = get_metrics(results, args, threshold, 0)
    global_results.append(result)

    df = pd.DataFrame(global_results)

    print("#")
    print("#")
    print("#")
    print("DF -1")
    print("#")
    print("#")
    print("#")
    print(df)
    print(" ")
    print(" ")
    print(" ")

    t_main = trange(args.iterations, desc="Attack", leave=False, position=0)

    # Define model as none, later it will be set and re-attacked
    model = None

    for i in t_main:
        # Train network
        model, results = train_and_evaluate(train_loader, val_loader, test_loader, device, args,
                                            input_shape=x_tensor.shape[1], model=model)

        result = get_metrics(results, args, threshold, fraction=(i*args.attack_size)/(base_size * 7))
        t_main.set_postfix(result)
        global_results.append(result)

        # Attack
        result_pts, result_class, labels = attack_keras_model(
            CArray(x_train_tensor),
            Y=CArray((y_train_tensor[:, 0] > threshold).int()),
            S=s_train_tensor,
            nb_attack=args.attack_size)

        # incorporate adversarial points
        result_pts = torch.tensor(np.around(result_pts.astype(np.float32), decimals=3)).clamp(0.0, 1.0)
        result_pts[result_pts != result_pts] = 0.0
        result_class[result_class != result_class] = 0.0

        x_train_tensor = torch.cat((x_train_tensor, result_pts))
        y_train_tensor = torch.cat(
            (y_train_tensor, torch.tensor(result_class.reshape(-1, 1).astype(np.float32)).clamp(0, 10)))
        l_train_tensor = torch.cat((l_train_tensor, torch.tensor(labels.tondarray().reshape(-1, 1).astype(np.float32))))
        
        # Generate array of random s values, one column per number of protected attributes
        s = np.random.randint(2, size=(len(result_class), len(protected_attributes_for_optimization)))
        s_train_tensor = torch.cat((s_train_tensor, torch.tensor(np.dstack((s,1-s)).reshape(len(result_class), 2*len(protected_attributes_for_optimization)).astype(np.float64))))

        train_dataset = TensorDataset(x_train_tensor, y_train_tensor, l_train_tensor, s_train_tensor)
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
        logging.debug("New training dataset has size {} (original {}).".format(len(train_loader), base_size * 7))

        df = pd.DataFrame(global_results)

        print("#")
        print("#")
        print("#")
        print("DF {}".format(i))
        print("#")
        print("#")
        print("#")
        print(df)
        print(" ")
        print(" ")
        print(" ")

    # Finally save experimental data if a save dir is specified
    if args.save_dir:
        import json
        from datetime import datetime
        if os.path.isdir(args.save_dir):
            timestamp: str = datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")
            folder: str = "{}_{}".format(args.dataset, timestamp)
            os.mkdir(os.path.join(args.save_dir, folder))

            # save history
            df.to_csv(os.path.join(args.save_dir, folder, "history.csv"))

            # save experiment settings
            with open(os.path.join(args.save_dir, folder, "settings.json"), "w") as fp:
                json.dump(args.__dict__, fp)

            # save latest model
            torch.save(model.state_dict(), os.path.join(args.save_dir, folder, "model.pt"))

        else:
            raise ValueError("Path is not valid.")


if __name__ == '__main__':
    # Define arguments for cli and run main function
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--iterations', help="Number of attack iterations", default=20, type=int)
    parser.add_argument('--batch-size', help="Size of each minibatch for the classifier", default=256, type=int)
    parser.add_argument('--show-graphs', help="Shows graph of training, etc. if true.", default=True)
    parser.add_argument('--grl-lambda', help="Gradient reversal parameter.", default=1, type=int)
    parser.add_argument('--attack-size', help="Number of adversarial points for each attack.", default=25, type=int)
    parser.add_argument('--reset-attack', help="Reuse the same model if False.", default=False, type=bool)
    parser.add_argument('--dataset', help="The data set to use; values: compas or adult", default="compas", type=str)
    parser.add_argument('--save-dir', help="Save history and setup if specified.", default=None)
    args = parser.parse_args()
    main(args)
