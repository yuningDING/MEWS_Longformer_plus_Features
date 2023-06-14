import argparse
import ast
import gc
import os
import random
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, \
    cohen_kappa_score
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, AutoConfig, AutoModel
from stream_fork import StreamFork

pd.options.mode.chained_assignment = None


# set parameters

class Parameters:
    TARGET_LIST = [0, 1, 2, 3, 4, 5, 6]
    # longformer = 'allenai/longformer-large-4096'
    # max_len = 1024
    # batch_size = 1
    # epochs = 1
    # learning_rate = 1e-5
    # random_seed = 42
    # folds = 10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", type=str, default="AD", choices=["AD", "TE"])
    parser.add_argument("-sr", "--score_rubric", type=str, default="Spr_fs_facets_rounded",
                        choices=["Spr_fs_facets_rounded", "Str_fs_facets_rounded", "Inh_fs_facets_rounded"])
    parser.add_argument("-lf", "--longformer", type=str, default='allenai/longformer-large-4096')
    parser.add_argument("-ml", "--max_len", type=int, default=512)
    parser.add_argument("-bs", "--batch_size", type=int, default=1)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5)
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-rs", "--random_seed", type=int, default=42)
    parser.add_argument("-f", "--folds", type=int, default=10)

    parser.parse_args(namespace=Parameters)


parse_args()

# setup output streams
h_params = "__".join(
    filter(
        lambda x: not (x.startswith("_") or "[" in x),
        map(
            lambda x: f"{x[0]}_{str(x[1]).replace('/', '-')}",
            vars(Parameters).items()
        )
    )
)
date_time_str = datetime.now().strftime("%Y%m%d%H%M")

verbose_outputs_file = open(f"verbose_results_{date_time_str}_{h_params}.txt", "w")
verbose_os = StreamFork(sys.stdout, verbose_outputs_file)
qwk_results_file = open(f"qwk_results_{date_time_str}_{h_params}.txt", "w")

# set random seeds

random.seed(Parameters.random_seed)
np.random.seed(Parameters.random_seed)
torch.manual_seed(Parameters.random_seed)

data_dir = "/kaggle/input" if os.environ.get('KAGGLE_KERNEL_RUN_TYPE', False) else "data"


# Step 1: Pre-processing of Data and CTAP Features
# The file Labels_texts_CTAPfeatures.xlsx is extracted into 2 dataframes (prompt AD and TE) with the following columns:
# id text Spr_fs_facets_rounded Str_fs_facets_rounded Inh_fs_facets_rounded ctap
# Spr_fs_facets_rounded, Str_fs_facets_rounded, Inh_fs_facets_rounded are the three traits, that we try to learn and
# predict. Each trait contains 7 classes.
# ctap contains all the CTAP Features in arrays.

def detect_multicollinearity(df, threshold):
    """
    Detects multicollinearity in a DataFrame and returns a list of non-multicollinear variables.
    :param df: DataFrame
    :param threshold: correlation threshold above which to detect multicollinearity
    :return: list of non-multicollinear variable names
    """
    colnames = list(df.columns)
    var_list = []
    cols = colnames[:-1]

    for n in cols:
        colnames.remove(n)
        for i in colnames:
            if i == n:
                pass
            else:
                c1 = df[[n, i]].corr().min()[0]
                if c1 >= threshold:
                    var_list.append(i)
                else:
                    pass

    colnames = list(df.columns)
    cols = []

    for c in colnames:
        if c in var_list:
            pass
        else:
            cols.append(c)

    return (cols)


def detect_na_columns(df):
    """
    Detects columns containing missing values (NAs) in a DataFrame and returns a list of column names.
    :param df: DataFrame
    :return: list of column names containing NAs
    """
    na_cols = df.isna().any()
    na_cols = na_cols[na_cols].index.tolist()
    return na_cols


def get_col_na_count(df, column):
    return df[column].isna().value_counts().loc[True]


def get_data_with_ctap(xlsx_file, prompt):
    df = pd.read_excel(xlsx_file)
    prompt_df = df[df["task"] == prompt].reset_index()

    # drop columns which contain more than 10 nas
    na_cols = detect_na_columns(prompt_df)
    cols_with_many_nas = [col for col in na_cols if get_col_na_count(prompt_df, col) > 10]
    prompt_df = prompt_df.drop(columns=cols_with_many_nas)

    # drop rows with na
    print("pre drop rows count: ", prompt_df.shape[0])
    prompt_df.dropna(axis="index", how="any", inplace=True)
    print("post drop rows count: ", prompt_df.shape[0])

    data = prompt_df[['id', 'text', 'Spr_fs_facets_rounded', 'Str_fs_facets_rounded', 'Inh_fs_facets_rounded']]

    # QUESTION: does removing multicollinearity make a difference?
    features = prompt_df[detect_multicollinearity(prompt_df[prompt_df.columns[21:]], threshold=0.90)].to_numpy()

    # features = prompt_df[prompt_df.columns[21:]].to_numpy()
    data['ctap'] = features.tolist()

    return data


df_AD = pd.read_csv(f"{data_dir}/mews-essays/CTAP_features_AD.csv")
df_AD["ctap"] = [ast.literal_eval(feature_vec) for feature_vec in df_AD["ctap"].values]
df_TE = pd.read_csv(f"{data_dir}/mews-essays/CTAP_features_TE.csv")
df_TE["ctap"] = [ast.literal_eval(feature_vec) for feature_vec in df_TE["ctap"].values]

num_features_AD = len(df_AD['ctap'].iloc[0])
num_features_TE = len(df_TE['ctap'].iloc[0])

print("Number of CTAP Features: AD-" + str(num_features_AD) + ", TE-" + str(num_features_TE))


# STEP 2: Split of Datasets
# For each dataframe, we first split 10% data as validation set. Then split the rest data into 5 Folds.

def get_k_folds(k, random_state, id_set):
    k_folds_train = []
    k_folds_test = []
    kf = KFold(n_splits=k, random_state=random_state, shuffle=True)
    for i, (train_index, test_index) in enumerate(kf.split(id_set)):
        # print(f"Fold {i}:")
        # print(f"  Amount Train: {len(train_index)}")
        # print(f"  Amount Test:  {len(test_index)}")
        k_folds_train.append([id_set[x] for x in train_index])
        k_folds_test.append([id_set[x] for x in test_index])
    return k_folds_train, k_folds_test


# %%
def validation_train_test_split(df):
    validate_ids = random.sample(list(df['id'].unique()), int(len(df['id'].unique()) / 10))
    validate = df[df['id'].isin(validate_ids)]
    rest_ids = [item for item in list(df['id'].unique()) if item not in validate_ids]
    train_ids, test_ids = get_k_folds(Parameters.folds, Parameters.random_seed, rest_ids)
    train = []
    test = []
    for i in range(Parameters.folds):
        train.append(df[df['id'].isin(train_ids[i])])
        # print(len(train[i]))
        test.append(df[df['id'].isin(test_ids[i])])
        # print(len(test[i]))

    return validate, train, test


validate_AD, train_AD, test_AD = validation_train_test_split(df_AD)
validate_TE, train_TE, test_TE = validation_train_test_split(df_TE)


# Step 3: Datasets and Model

class MEWSDataset(Dataset):
    def __init__(self, data, max_len, tokenizer, target, extra_feature=None):
        self.data = data
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.text = data['text'].values
        self.targets = data[Parameters.TARGET_LIST].values
        self.essay_id = data['id'].values
        if extra_feature != None:
            self.extra_feature = data[extra_feature].values
        else:
            self.extra_feature = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        inputs = self.tokenizer.encode_plus(self.text[index].lower(),
                                            truncation=True,
                                            padding='max_length',
                                            add_special_tokens=True,
                                            return_attention_mask=True,
                                            return_token_type_ids=True,
                                            max_length=self.max_len,
                                            return_tensors='pt')

        input_ids = inputs['input_ids'].flatten()
        attention_mask = inputs['attention_mask'].flatten()
        token_type_ids = inputs['token_type_ids'].flatten()
        targets = torch.FloatTensor(self.targets[index])
        # print(targets)

        if len(self.extra_feature) > 0:
            return {'input_ids': input_ids, 'attention_mask': attention_mask,
                    'extra_data': torch.FloatTensor(self.extra_feature[index]), 'token_type_ids': token_type_ids,
                    'targets': targets}
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids,
                'targets': targets}


class LongformerWithCustomFeatureModel(torch.nn.Module):
    def __init__(self, model_name, num_extra_dims, num_labels):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
        num_hidden_size = self.transformer.config.hidden_size
        self.classifier = torch.nn.Linear(num_hidden_size + num_extra_dims, num_labels)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, num_labels)

    def forward(self, input_ids, extra_data, attention_mask=None):
        hidden_states = self.transformer(input_ids=input_ids,
                                         attention_mask=attention_mask)  # [batch size, sequence length, hidden size]
        cls_embeds = hidden_states.last_hidden_state[:, 0, :]  # [batch size, hidden size]
        concat = torch.cat((cls_embeds, extra_data), dim=-1)  # [batch size, hidden size+num extra dims]
        output = self.classifier(concat)  # [batch size, num labels]
        return output


class LongformerModel(torch.nn.Module):
    def __init__(self):
        super(LongformerModel, self).__init__()
        self.model = LongformerForSequenceClassification.from_pretrained(Parameters.longformer, return_dict=True,
                                                                         num_labels=len(Parameters.TARGET_LIST))

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits

        # output_label = output.logits.argmax().item()

        return output


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def get_optimizer(model, learning_rate):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    return optimizer


# %%
def build_model_tokenizer(withCustomFeature, num_extra_dims, model_path=None):
    # Tokenizer
    tokenizer = LongformerTokenizerFast.from_pretrained(Parameters.longformer)

    # Modell
    if withCustomFeature:
        model = LongformerWithCustomFeatureModel(Parameters.longformer, num_extra_dims, len(Parameters.TARGET_LIST))
    else:
        model = LongformerModel()

    if model_path is not None:
        model.load_state_dict(torch.load(model_path))

    return tokenizer, model


# %%
def train_model(n_epochs,
                train_loader,
                val_loader,
                test_loader,
                model, lr,
                device, extra_data=None):
    optimizer = get_optimizer(model, lr)
    model.to(device)
    for epoch in range(n_epochs):
        train_loss = 0
        val_loss = 0
        model.train()
        print(f' Epoch: {epoch + 1} - Train Set '.center(50, '='),
              file=verbose_os)
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device, dtype=torch.long)
            attention_mask = batch['attention_mask'].to(device, dtype=torch.long)
            token_type_ids = batch['token_type_ids'].to(device, dtype=torch.long)
            targets = batch['targets'].to(device, dtype=torch.float)

            if extra_data is None:
                outputs = model(input_ids, attention_mask, token_type_ids)
            else:
                extra_data = batch[extra_data].to(device, dtype=torch.long)
                outputs = model(input_ids, extra_data, attention_mask)
            # print(outputs)
            # print(targets)
            optimizer.zero_grad()
            loss = loss_fn(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.item() - train_loss))
            del input_ids, attention_mask, token_type_ids, targets, outputs
            gc.collect()

        print(f' Epoch: {epoch + 1} - Validation Set '.center(50, '='),
              file=verbose_os)
        model.eval()
        val_targets = []
        val_outputs = []
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                input_ids = data['input_ids'].to(device, dtype=torch.long)
                attention_mask = data['attention_mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)
                if extra_data is None:
                    outputs = model(input_ids, attention_mask, token_type_ids)
                else:
                    extra_data = data[extra_data].to(device, dtype=torch.long)
                    outputs = model(input_ids, extra_data, attention_mask)
                loss = loss_fn(outputs, targets)
                val_loss = val_loss + ((1 / (batch_idx + 1)) * (loss.item() - val_loss))
                val_targets.extend(targets.cpu().detach().numpy().tolist())
                val_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
                del input_ids, attention_mask, token_type_ids, targets, outputs
                gc.collect()
            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)
            print('Epoch: {} \tAvgerage Training Loss: {:.6f} \tAverage Validation Loss: {:.6f} \n'.format(
                epoch + 1,
                train_loss,
                val_loss
            ), file=verbose_os)
        val_outputs_labels = np.array([np.argmax(a) for a in val_outputs])
        val_targets_labels = np.array([np.argmax(a) for a in val_targets])
        val_qwk = cohen_kappa_score(val_targets_labels, val_outputs_labels, weights='quadratic')
        print(f"Validation QWK: {round(val_qwk, 4)}", file=verbose_os)

        print('Test', file=verbose_os)
        model.eval()
        test_targets = []
        test_outputs = []
        test_loss = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                input_ids = data['input_ids'].to(device, dtype=torch.long)
                attention_mask = data['attention_mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)
                if extra_data is None:
                    outputs = model(input_ids, attention_mask, token_type_ids)
                else:
                    extra_data = data[extra_data].to(device, dtype=torch.long)
                    outputs = model(input_ids, extra_data, attention_mask)
                loss = loss_fn(outputs, targets)
                test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.item() - test_loss))
                test_targets.extend(targets.cpu().detach().numpy().tolist())
                test_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        test_outputs_labels = np.array([np.argmax(a) for a in test_outputs])
        test_targets_labels = np.array([np.argmax(a) for a in test_targets])
        accuracy = accuracy_score(test_targets_labels, test_outputs_labels)
        recall_micro = recall_score(test_targets_labels, test_outputs_labels, average='micro')
        recall_macro = recall_score(test_targets_labels, test_outputs_labels, average='macro')
        f1_score_micro = f1_score(test_targets_labels, test_outputs_labels, average='micro')
        f1_score_macro = f1_score(test_targets_labels, test_outputs_labels, average='macro')
        qwk = cohen_kappa_score(test_targets_labels, test_outputs_labels, weights='quadratic')
        print(f"Test Loss: {round(test_loss, 4)}", file=verbose_os)
        print(f"Accuracy Score: {round(accuracy, 4)}", file=verbose_os)
        print(f"Recall (Micro): {round(recall_micro, 4)}", file=verbose_os)
        print(f"Recall (Macro): {round(recall_macro, 4)}", file=verbose_os)
        print(f"F1 Score (Micro): {round(f1_score_micro, 4)}", file=verbose_os)
        print(f"F1 Score (Macro): {round(f1_score_macro, 4)} \n", file=verbose_os)
        print(f"QWK: {round(qwk, 4)}", file=StreamFork(verbose_os, qwk_results_file))
        cm = confusion_matrix(test_targets_labels, test_outputs_labels)
        print("Confusion Matrix:", file=verbose_os)
        print(cm, file=verbose_os)

    return model


# STEP 4: Training Pipeline


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)


def transfer_targets(df, target):
    for col in Parameters.TARGET_LIST:
        df[col] = np.where(df[target] == col, 1, 0)
    return df


# Pipeline 1 - Baseline Longformer without extra features

tokenizer, model = build_model_tokenizer(withCustomFeature=False, num_extra_dims=0)

valid_dataset = MEWSDataset(transfer_targets(validate_AD, Parameters.score_rubric), max_len=Parameters.max_len,
                            tokenizer=tokenizer, target=Parameters.score_rubric)
val_data_loader = DataLoader(valid_dataset, shuffle=False, batch_size=Parameters.batch_size)

# TODO: Run 10 Folds, 10 Epochs
for i in range(Parameters.folds):
    train_dataset = MEWSDataset(transfer_targets(train_AD[i], Parameters.score_rubric), max_len=Parameters.max_len,
                                tokenizer=tokenizer, target=Parameters.score_rubric)
    train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=Parameters.batch_size)
    test_dataset = MEWSDataset(transfer_targets(test_AD[i], Parameters.score_rubric), max_len=Parameters.max_len,
                               tokenizer=tokenizer, target=Parameters.score_rubric)
    test_data_loader = DataLoader(test_dataset, shuffle=False, batch_size=Parameters.batch_size)
    model = train_model(n_epochs=Parameters.epochs, train_loader=train_data_loader, val_loader=val_data_loader,
                        test_loader=test_data_loader, model=model, lr=Parameters.learning_rate, device=device)

    # TODO: Run three targets in ['Spr_fs_facets_rounded', 'Str_fs_facets_rounded', 'Inh_fs_facets_rounded']
    # TODO: Run two prompts in ['AD', 'TE']
# %%

# Pipeline 2 - Longformer with extra feature
# tokenizer, model = build_model_tokenizer(withCustomFeature=True, num_extra_dims=num_features_AD)

# valid_dataset = MEWSDataset(validate_AD, max_len=Parameters.max_len,tokenizer=tokenizer,target=Parameters.score_rubric, extra_feature='ctap')
# val_data_loader = DataLoader(valid_dataset,shuffle=False,batch_size=Parameters.batch_size)
# for i in range(Parameters.folds):
#    train_dataset = MEWSDataset(train_AD[i], max_len=Parameters.max_len,tokenizer=tokenizer,target=Parameters.score_rubric, extra_feature='ctap')
#    train_data_loader = DataLoader(train_dataset,shuffle=True,batch_size=Parameters.batch_size)
#    test_dataset = MEWSDataset(test_AD[i], max_len=Parameters.max_len,tokenizer=tokenizer,target=Parameters.score_rubric, extra_feature='ctap')
#    test_data_loader = DataLoader(test_dataset,shuffle=False,batch_size=Parameters.batch_size)
#    model = train_model(n_epochs=Parameters.epochs, train_loader=train_data_loader, val_loader=val_data_loader,test_loader=test_data_loader,model=model, targets=Parameters.score_rubric, lr=Parameters.learning_rate, extra_data='ctap', device=device)
# TODO: Run two prompts in ['AD', 'TE']
# TODO: Run three targets in ['Spr_fs_facets_rounded', 'Str_fs_facets_rounded', 'Inh_fs_facets_rounded']


# close output streams

verbose_outputs_file.close()
qwk_results_file.close()
