# ----- SYSTEM PACKAGES ----- ##
import os
import re
import sys
import json
import time
import random
import numpy as np

# ----- CLASSIFIERS ----- ##
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ----- OPENAI ----- ##
import openai
from openai import OpenAI

# ----- TORCH ----- ##
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ----- PLOTTING ----- ##
from tqdm import tqdm
import matplotlib as mpl
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython.display import clear_output

## ----- REPRODUCIBILITY ----- ##
seed = 111
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

## ----------------------------------------
## HELPER FUNCTIONS
## ----------------------------------------

# helper function to save results to a JSON file
def save_to_json(data_list, filename="output.json"):
    with open(filename, 'w', encoding='utf-8') as json_file:
        json.dump(data_list, json_file, ensure_ascii=False, indent=4)

# helper function to load results from a JSON file
def load_from_json(filename="output.json"):
    with open(filename, 'r', encoding='utf-8') as json_file:
        return json.load(json_file)

# helper function to extract embeddings from a given JSON file
def load_embeddings_and_labels(filename):

    # load the dataset
    data_list = load_from_json(filename)

    # concatenate UW and W embeddings
    all_embeddings = np.concatenate(([item["uw_embedding"] for item in data_list], [item["w_embedding"] for item in data_list]))

    # create labels
    all_labels = np.concatenate((np.zeros(len(all_embeddings) // 2), np.ones(len(all_embeddings) // 2)))

    return all_embeddings, all_labels
    
# number of GPUs available
ngpu = torch.cuda.device_count()

# decide which device we want to run on
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# numpy <--> tensor helpers
to_t = lambda array: torch.tensor(array, device=device, dtype=torch.float32)
from_t = lambda tensor: tensor.to('cpu').detach().numpy()
    
## ----------------------------------------
## OPENAI
## ----------------------------------------

# define client
client = OpenAI(api_key="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",)

# inference ada-002 embeddings
def get_embedding(text):
    
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model="text-embedding-ada-002").data[0].embedding

# gpt-judger prompt and score extraction
def gpt_judge(prompt, uw_output, w_output, is_randomized = None):

    if is_randomized is None:
            is_randomized = random.choice([True, False])
    
    # randomizing the order
    a_output, b_output = (uw_output, w_output) if is_randomized else (w_output, uw_output)
    
    judge_prompt = """[System]
    Please act as an impartial judge and evaluate the quality of the text completions provided by two large language models to the prompt displayed below. 
    Assess each response according to the criteria outlined, using a 1-5 Likert scale where 1 indicates strong disagreement or the lowest quality, and 5 indicates strong agreement or the highest quality. 
    After scoring each criterion, provide a short summary for each response, including specific examples that influenced your scoring. 
    Additionally, don't let the length of the responses influence your evaluation. 
    Be as objective as possible and ensure that the order in which the responses are presented does not affect your decision.
    
    Criteria:
    1. Relevance to the prompt
    2. Depth of detail
    3. Clarity of writing
    4. Coherence and logical flow
    5. Originality and insight
    6. Use of specific examples
    7. Accuracy of information
    
    Start with a brief statement about which response you think is superior. Then, for each response and criterion, provide a score, followed by a brief justification for that score. At the very end of your response, declare your verdict by choosing one of the choices below, strictly following the given format:
    
    "[[A]]: [list of scores]" if assistant A is better, 
    "[[B]]: [list of scores]" if assistant B is better, or 
    "[[C]]" for a tie.

    [Prompt]
    {prompt}
    
    [The Start of LLM A’s Answer]
    {a_output}
    [The End of LLM A’s Answer]
    
    [The Start of LLM B’s Answer]
    {b_output}
    [The End of LLM B’s Answer]
    """.format(prompt=prompt, a_output=a_output, b_output=b_output)
    
    completion = client.chat.completions.create(
        messages=[{
                "role": "user",
                "content": judge_prompt,}],
        model="gpt-3.5-turbo",
    )

    judge_output = completion.choices[0].message.content

    # search for a tie first
    tie_pattern = r'\[\[C\]\]'
    tie_match = re.search(tie_pattern, judge_output)
    
    if tie_match:
        judge_choice = "C"
        final_verdict = "Tie"
        scores = []
    else:
        # pattern to match the verdict and the scores for A or B
        pattern = r'\[\[([AB])\]\]: (?:\[)?([5, 4, 3, 2, 1, ]+)(?:\])?'
        matches = re.findall(pattern, judge_output)
        
        if matches:
            
            # extract the last match which will have the choice and the corresponding scores
            judge_choice, scores_str = matches[-1]
            
            # remove square brackets if they exist, strip whitespace, and split by comma
            scores_str = scores_str.replace('[', '').replace(']', '').strip()
            scores = [float(score) for score in scores_str.split(',')]
            
            # determine verdict based on the judge choice
            if is_randomized:
                final_verdict = "Unwatermarked" if judge_choice == 'A' else "Watermarked"
            else:
                final_verdict = "Watermarked" if judge_choice == 'A' else "Unwatermarked"
        else:
            final_verdict = "Model Failure"
            scores = []
    
    return judge_output, final_verdict, scores, is_randomized


## ----------------------------------------
## RESULT ANALYSIS
## ----------------------------------------
    
# function to tally judger results
def extract_and_count_choices(data_list, category):
    
    # extract all 'category' entries from the list of dictionaries
    choices = [entry[category] for entry in data_list]
    
    # count occurrences of each 'judge_choice'
    count = {}
    for choice in choices:
        if choice in count:
            count[choice] += 1
        else:
            count[choice] = 1

    count_norm = count.copy()
    for choice in count_norm:
        count_norm[choice] /= len(choices)
        count_norm[choice] *= 100
        count_norm[choice] = np.round(count_norm[choice], 3)
        
    return count, count_norm

# A helper function to convert probabilities to binary outcomes based on a threshold
def prob_to_binary(probabilities, threshold=0.5):
    return (probabilities > threshold).astype(int)


## ----------------------------------------
## CLASSIFIER
## ----------------------------------------

## ----------- MULTI LAYER PERCEPTRON ----------- ##
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        self.l1 = nn.Linear(1536, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 64)
        self.l5 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
 
    def forward(self, input):               
        x = self.l1(input)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        x = self.relu(x)
        x = self.l4(x)
        x = self.relu(x)
        x = self.l5(x)
        x = self.sigmoid(x)
        return x

## ----------- DATASET ----------- ##
class CustomDataset(Dataset):
    def __init__(self, embeddings, classifications):
        self.embeddings = embeddings
        self.classifications = classifications

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        classification = self.classifications[idx]
        return embedding, classification
        
## ----- WEIGHT INITIALIZATION ----- ##
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# train a mlp-based classifier
def train_classifier(train, test, batch_size = 100, lr = 2e-4, num_epochs = 100, weight_decay = 2e-3, shuffle = False, verbose = True):

    # train and test in (embeddings, label) format
    
    ## ----- INIT MODEL----- ##
    net = MLP().to(device)

    ## ----- INIT WEIGHTS ----- ##
    net.apply(weights_init)
    net = net.float()

    ## ----- LOSS ----- ##
    criterion = nn.BCELoss()

    ## ----- OPTIMIZER ----- ##
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay = weight_decay) 

    # add dynamic learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

    # set the seed for numpy and torch in the main process
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    ## ----- DEFINE DATALOADER ----- ##
    train_dataset = CustomDataset(to_t(train[0]), to_t(train[1]))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, worker_init_fn=np.random.seed(seed))

    # store loss
    running_loss = []

    ## ----- DISPLAY ----- ##
    if verbose:
        pbar = tqdm(range(num_epochs))
        pbar.set_description("Epoch")
    else:
        pbar = range(num_epochs)
    
    ## ----- FOR EACH EPOCH ----- ##
    for epoch in pbar:
        
        ## ----- FOR EACH BATCH ----- ##
        for i, data in enumerate(train_dataloader, 0):

            # unpack data
            data_batch = data[0].to(device)
            label = data[1].to(device).squeeze()

            # zero gradients
            net.zero_grad()
            
            # run network
            output = net(data_batch).squeeze()
            
            # calculate current loss 
            error = criterion(output, label.float())
            
            # calculate gradients in backward pass
            error.backward()

            # update weights
            optimizer.step()
            
            # update learning rate
            scheduler.step(error)
            
            # save losses for plotting later
            running_loss.append(error.item())

            # update progress bar if verbose
            if verbose:
                pbar.set_description("Epoch {:03} Loss: {:03} lr {:.6}" \
                        .format(epoch, round(error.item(), 3), 
                                scheduler.optimizer.param_groups[0]['lr'])) 
        
        if verbose: pbar.update(1)
    if verbose: pbar.close()

    # plot loss
    if verbose:
        plt.figure(figsize = (6, 4))
        plt.title("Loss During Training")
        plt.plot(np.abs(running_loss))
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.tight_layout()
        plt.show()
        
    # inference model
    test_results = net(to_t(test[0])).detach()
    test_results = np.round(from_t(test_results), 3)

    # separate results into UW and W samples
    uw_results = test_results[~test[1].astype(bool)]
    w_results = test_results[test[1].astype(bool)]

    # calculate metrics on W data
    w_incorrect = sum(w_results < 0.5)
    w_correct = sum(w_results >= 0.5)

    # calculate metrics on UW data
    uw_incorrect = sum(uw_results >= 0.5)
    uw_correct = sum(uw_results < 0.5)

    # calculate overall accuracy and AUC
    accuracy = (w_correct + uw_correct) / len(test[0])
    auc = roc_auc_score(test[1], test_results)

    # calculate FP and FN rates
    fp_rate = uw_incorrect / (uw_incorrect + uw_correct)
    fn_rate = w_incorrect / (w_incorrect + w_correct)

    # print all calculated metrics
    if verbose:
        print("W Incorrect (FN):", w_incorrect)
        print("W Correct (TP):", w_correct)
        
        print("UW Incorrect (FP):", uw_incorrect)
        print("UW Correct (TN):", uw_correct)
        
        print("Accuracy:", accuracy)
        print("AUC:", auc)
        
        print("FP Rate:", fp_rate)
        print("FN Rate:", fn_rate)

    return accuracy[0], auc, fp_rate, fn_rate, net

def kfold_classifier(current_embeddings, current_labels, batch_size = 100, lr = 2e-4, weight_decay = 2e-3, shuffle = True):

    # prepare the K-fold cross-validation
    kfold = KFold(n_splits = 5, shuffle = True, random_state = 101)

    # store results for each fold
    all_fold_results_classifier = []
    all_fold_results_auc = []
    all_fold_results_fp = []
    all_fold_results_fn = []

    for train, test in kfold.split(current_embeddings, current_labels):

        # unpack data
        X_train, y_train = current_embeddings[train], current_labels[train]
        X_test, y_test = current_embeddings[test], current_labels[test]

        # train classifier
        classifier_acc, auc, fp_rate, fn_rate, _ = train_classifier((X_train, y_train), (X_test, y_test),
                                                  batch_size = batch_size, lr = lr, num_epochs = 100, 
                                                  weight_decay = weight_decay, shuffle = shuffle, verbose = False)
        
        # save results for the current fold
        all_fold_results_classifier.append(classifier_acc)
        all_fold_results_auc.append(auc)
        all_fold_results_fp.append(fp_rate[0])
        all_fold_results_fn.append(fn_rate[0])
        
    # print results for all folds
    print(f"Classifier Accuracy Across All Folds:", all_fold_results_classifier)
    print(f"Classifier AUC Across All Folds:", all_fold_results_auc)
    print(f"Classifier FP Across All Folds:", all_fold_results_fp)
    print(f"Classifier FN Across All Folds:", all_fold_results_fn, end="\n\n")
    
    # print the average across the folds
    print(f"Average Classifier Accuracy Across All Folds = {np.mean(all_fold_results_classifier):.4f}")
    print(f"Average Classifier AUC Across All Folds = {np.mean(all_fold_results_auc):.4f}")
    print(f"Average Classifier FP Rate Across All Folds = {np.mean(all_fold_results_fp):.4f}")
    print(f"Average Classifier FN Rate Across All Folds = {np.mean(all_fold_results_fn):.4f}", end="\n\n")
    
    # print standard deviation of the accuracy across all folds
    print(f"Classifier Standard Error Across All Folds = {np.std(all_fold_results_classifier) / np.sqrt(5):.4f}")
    
    
# train and evaluate a linear regression model
def train_and_evaluate_linear_regression(train, test):
    
    # unpack the dataset into training and testing sets
    X_train, y_train = train[0], train[1]
    X_test, y_test = test[0], test[1]
    
    # initialize the regression model
    model = LinearRegression()
    
    # fit the model to the training data
    model.fit(X_train, y_train)
    
    # predict on the training data
    train_predictions = model.predict(X_train)
    train_predictions_binary = prob_to_binary(train_predictions)
    
    # predict on the test data
    test_predictions = model.predict(X_test)
    test_predictions_binary = prob_to_binary(test_predictions)
    
    # evaluate the model
    train_accuracy = accuracy_score(y_train, train_predictions_binary)
    test_accuracy = accuracy_score(y_test, test_predictions_binary)
    
    return train_accuracy, test_accuracy
    
    
def kfold_regression(current_embeddings, current_labels):

    # prepare the K-fold cross-validation
    kfold = KFold(n_splits = 5, shuffle = True, random_state = 101)

    # store the results for each fold
    all_fold_results_regression = []

    for train, test in kfold.split(current_embeddings, current_labels):

        # unpack the train and test data
        X_train, y_train = current_embeddings[train], current_labels[train]
        X_test, y_test = current_embeddings[test], current_labels[test]
        
        # run linear regression
        regression_acc = train_and_evaluate_linear_regression((X_train, y_train), (X_test, y_test))[1]
    
        # update results
        all_fold_results_regression.append(regression_acc)
        
    # print results across the folds
    print(f"Regression Accuracy Across All Folds:", all_fold_results_regression)
    print(f"Average Regression Accuracy Across All Folds = {np.mean(all_fold_results_regression):.4f}")
    print(f"Regression Standard Error Across All Folds = {np.std(all_fold_results_regression) / np.sqrt(5):.4f}")