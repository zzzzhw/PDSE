#! /usr/bin/env python

from distutils import core
import os

import datetime as dt
import logging
import numpy as np
import time

import pandas as pd
import torch
import xgboost as xgb

from collections import Counter, defaultdict
from dateutil.relativedelta import relativedelta
from pprint import pformat
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import LinearSVC

# local imports
import data
import utils
from common import get_model_stats
from model import CAE, Enc
from model import MLPClassifier
from model import SimpleEncClassifier
from model import ResClassifier
from joblib import dump
from joblib import load
from selector_cadeood import OODSelector
from selector_pseudo_loss import LocalPseudoLossSelector
from selector_simple import UncertainPredScoreSelector, MultiUncertainPredScoreSelector
from selector_transcend import TranscendSelector
from utils import save_model
from train import train_encoder, train_classifier
from xgboost_wrapper import xgboost_wrapper
from torch.utils.data import DataLoader
import json

# 声明全局变量
f1_scores, f1_scores_retrain = [], []  # 用于存储 F1 分数的全局列表
diffs, diffs_retrain = [], []  # 用于存储月份差异的全局列表
def eval_classifier(args, classifier, cur_month_str, X, y_binary, y_family, train_families, \
                        fout, fout_retrain, fam_out, stat_out, retrain=False, start_month='2012-12', gpu=False, multi=False):
    if args.classifier == 'res':
        classifier.eval()
        test_loader = DataLoader(X, batch_size=5000, shuffle=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        classifier.to(device)
        if torch.cuda.is_available():
            with torch.no_grad():
                y_pred = []
                for features in test_loader:
                    features = features.float()
                    features = features.to(device)
                    # 将featrues转换为32位
                    #features = features.cuda()
                    features = features.reshape(-1, 34, 34)
                    features = features.unsqueeze(0)
                    features = features.permute(1, 0, 2, 3)
                    y_hat = classifier.predict(features)
                    # 从计算图抽出。
                    y_hat = y_hat.detach().cpu().numpy().tolist()
                    y_pred.extend(y_hat)
        y_pred = np.array(y_pred)
    else:
        if gpu == True:
            X_tensor = torch.from_numpy(X).float()
            if torch.cuda.is_available():
                X_tensor = X_tensor.cuda()
                y_pred = classifier.cuda().predict(X_tensor)
                y_pred = y_pred.cpu().detach().numpy()
            else:
                y_pred = classifier.predict(X_tensor).numpy()
        else:
            y_pred = classifier.predict(X)
    
    # logging.info(f'y_pred[0]: {y_pred[0]}')
    # logging.info(f'y_binary[0]: {y_binary[0]}')
    if args.multi_class == True:
        # process multi-class y_pred to binary
        # if y_pred is 0, it is 0, otherwise it is 1
        y_pred_bin = np.where(y_pred == 0, 0, 1)
    else:
        y_pred_bin = y_pred

    start_date = dt.datetime.strptime(start_month, "%Y-%m")
    end_date = dt.datetime.strptime(cur_month_str, "%Y-%m")

    # 计算年份和月份的差异
    year_diff = end_date.year - start_date.year
    month_diff = end_date.month - start_date.month

    # 总的月份差异
    diff = year_diff * 12 + month_diff
    tpr, tnr, fpr, fnr, acc, precision, f1 = get_model_stats(y_binary, y_pred_bin, multi_class = multi)
    if retrain:
        f1_scores_retrain.append(f1)
        diffs_retrain.append(diff)
        aut_f1_retrain = np.trapz(f1_scores_retrain, diffs_retrain)
        fout_retrain.write('%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n' % \
                           (cur_month_str, tpr, tnr, fpr, fnr, acc, precision, f1, aut_f1_retrain))
        fout_retrain.flush()
        return y_pred
    else:
        f1_scores.append(f1)
        diffs.append(diff)
        aut_f1 = np.trapz(f1_scores, diffs)
        fout.write('%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\n' % \
                   (cur_month_str, tpr, tnr, fpr, fnr, acc, precision, f1, aut_f1))
        fout.flush()
        if multi == False:
            tn, fp, fn, tp = confusion_matrix(y_binary, y_pred_bin).ravel()
            stat_out.write('%s\t%d\t%d\t%d\t%d\t%d\n' % \
                           (cur_month_str, X.shape[0], tp, tn, fp, fn))
            stat_out.flush()

        # check FNR within different families.
        family_cnt = defaultdict(lambda: 0)
        for idx, family in enumerate(y_family):
            family_cnt[family] += 1
        neg_by_fam = defaultdict(lambda: 0)
        family_to_idx = defaultdict(list)
        # y_family can be all_train_family since we only care abou False Negatives
        fn_indices = np.where((y_binary != y_pred_bin) & (y_binary != 0))[0]
        for idx in fn_indices:
            family = y_family[idx]
            neg_by_fam[family] += 1
            family_to_idx[family].append(idx)
        for family, neg_cnt in neg_by_fam.items():
            new = family not in train_families
            fam_total = family_cnt[family]
            fam_rate = neg_cnt / float(fam_total)
            # 记录样本索引
            indexsingle = family_to_idx[family]
            fam_out.write('%s\t%s\t%s\t%s\t%d\t%s\n' % (cur_month_str, new, family, fam_rate, neg_cnt, indexsingle))
            fam_out.flush()
        return y_pred, neg_by_fam, family_to_idx

def main():

    """
    Step (0): Init log path and parse args.
    """
    args = utils.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.ids:
        ids_compatible = (
            (args.encoder == 'simple-enc-mlp' and args.loss_func == 'hi-dist-xent')
            or (args.encoder == 'cae' and args.loss_func == 'triplet-mse')
        )
        if not ids_compatible:
            raise ValueError(
                'IDS requires HCL (--encoder simple-enc-mlp, --loss_func hi-dist-xent) '
                'or CADE (--encoder cae, --loss_func triplet-mse)'
            )
        if not args.encoder_retrain:
            raise ValueError('IDS requires --encoder_retrain for monthly model updates')
        if not 0.0 <= args.ids_ema_decay < 1.0:
            raise ValueError('--ids-ema-decay must be in [0, 1)')
        if args.ids_lambda <= 0 or args.ids_batch_size < 3:
            raise ValueError('IDS requires positive perturbation scale and batch size >= 3')

    start_epoch, end_epoch, step = args.lr_decay_epochs.split(',')
    args.lr_decay_epochs = list([range(int(start_epoch), int(end_epoch), int(step))])

    log_file_path = args.log_path
    if args.verbose == False:
        logging.basicConfig(filename=log_file_path,
                            filemode='a',
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            level=logging.INFO,
                            )
    else:
        logging.basicConfig(filename=log_file_path,
                            filemode='a',
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            level=logging.DEBUG,
                            )
    
    logging.info('Running with configuration:\n' + pformat(vars(args)))

    """
    Step (1): Prepare the training dataset. Load the feature vectors and labels.
    """

    logging.info(f'Loading {args.data} training dataset')
    if args.data.startswith('tesseract') or \
        args.data.startswith('gen_tesseract') or \
        args.data.startswith('fam_tesseract') or \
        args.data.startswith('emberv2'):
        X_train, y_train, all_train_family = data.load_range_dataset_w_benign(args, args.data, args.train_start, args.train_end)
    else:
        X_train, y_train, y_train_family = data.load_range_dataset_w_benign(args, args.data, args.train_start, args.train_end)
        # all_train_family has 'benign'
        ben_len = X_train.shape[0] - y_train_family.shape[0]
        y_ben_family = np.full(ben_len, 'benign')
        all_train_family = np.concatenate((y_train_family, y_ben_family), axis=0)
            
    train_families = set(all_train_family)
    all_train_family_original = all_train_family
    
    # count label distribution
    counted_labels = Counter(y_train)
    logging.info(f'Loaded X_train: {X_train.shape}, {y_train.shape}')
    logging.info(f'y_train labels: {np.unique(y_train)}')
    logging.info(f'y_train: {Counter(y_train)}')

    # the index mapping for the first training set
    new_y_mapping = {}
    for _, label in enumerate(np.unique(y_train)):
        new_y_mapping[label] = label

    """
    Step (2): Variable names and file names.
    """
    # some commonly used variables.
    if args.train_start != args.train_end:
        train_dataset_name = f'{args.train_start}to{args.train_end}'
    else:
        train_dataset_name = f'{args.train_start}'

    if args.model_dir is None:
        SAVED_MODEL_FOLDER = f'models/{args.method}/{args.count}'
    else:
        SAVED_MODEL_FOLDER = args.model_dir
    # only based on malicious training samples
    NUM_FEATURES = X_train.shape[1]
    NUM_CLASSES = len(np.unique(y_train))

    logging.info(f'Number of features: {NUM_FEATURES}; Number of classes: {NUM_CLASSES}')

    # convert y_train to y_train_binary
    y_train_binary = np.array([1 if item != 0 else 0 for item in y_train])
    BIN_NUM_CLASSES = 2
    #class_weight = None

    """
    Step (3): Train the encoder model.
    `encoder` needs to have the same APIs.
    If they don't have the required API, we could use a wrapper.
    """
    train_encoder_func = train_encoder
    # set up the encoder model
    if args.encoder == None:
        # We will not use an encoder. The input features are used directly.
        logging.info('Not using an encoder. Using the input features directly.')
    elif args.encoder == 'mlp':
        # assert args.encoder == args.classifier, "mlp encoder is from mlp classifier"
        if args.multi_class == True:
            output_dim = len(np.unique(y_train))
        else:
            output_dim = BIN_NUM_CLASSES
        mlp_dims = utils.get_model_dims('MLP', NUM_FEATURES, args.mlp_hidden, output_dim)
        enc_dims = mlp_dims[:-1]
        encoder = MLPClassifier(mlp_dims)
        # not used
        # encoder_name = 'MLP'
    elif args.encoder == 'simple-enc-mlp':
        # Enc + MLP model 
        enc_dims = utils.get_model_dims('Encoder', NUM_FEATURES,
                            args.enc_hidden, NUM_CLASSES)
        mlp_dims = utils.get_model_dims('MLP', enc_dims[-1], args.mlp_hidden, BIN_NUM_CLASSES)
        encoder = SimpleEncClassifier(enc_dims, mlp_dims)
        encoder_name = 'simple_enc_classifier'
    elif args.encoder == 'cae':
        # CAE + MLP model 
        enc_dims = utils.get_model_dims('Encoder', NUM_FEATURES,
                            args.enc_hidden, NUM_CLASSES)
        encoder = CAE(enc_dims)
        encoder_name = 'cae'
    elif args.encoder == 'enc':
        # CAE + MLP model 
        enc_dims = utils.get_model_dims('Encoder', NUM_FEATURES,
                            args.enc_hidden, NUM_CLASSES)
        encoder = Enc(enc_dims)
        encoder_name = 'enc'
    else:
        raise Exception(f'The encoder {args.encoder} is not supported yet.')

    MODEL_DIR = os.path.join(SAVED_MODEL_FOLDER, train_dataset_name)
    utils.create_folder(MODEL_DIR)
    if args.encoder == 'mlp':
        # set Adam optimizer
        mlp_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.mlp_lr)

        # ENC_MODEL_PATH save model name
        mlp_dims_str = str(mlp_dims).replace(' ', '').replace(',', '-').replace('[', '').replace(']', '') # remove extra symbols
        ENC_MODEL_PATH = os.path.join(MODEL_DIR, f'MLP_{mlp_dims_str}_feat_{args.cls_feat}' + \
                                    f'_dropout{args.mlp_dropout}' + \
                                    f'_{args.optimizer}_{args.scheduler}' + \
                                    f'_lr{args.mlp_lr}' + \
                                    f'_b{args.mlp_batch_size}_e{args.mlp_epochs}_mdate{args.mdate}.pth')       
        logging.info(f'Initial classifier and encoder model: ENC_MODEL_PATH {ENC_MODEL_PATH}')
    elif args.encoder != None:
        if args.optimizer == 'adam':
            # Adam optimizer
            optimizer_func = torch.optim.Adam
            optimizer = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate)
        elif args.optimizer == 'sgd':
            # SGD optimizer
            optimizer_func = torch.optim.SGD
            optimizer = torch.optim.SGD(encoder.parameters(), lr=args.learning_rate)
        else:
            raise Exception(f'The optimizer {args.optimizer} is not supported yet.')
        
        # ENC_MODEL_PATH save model name
        enc_dims_str = str(enc_dims).replace(' ', '').replace(',', '-').replace('[', '').replace(']', '') # remove extra symbols
        ENC_MODEL_PATH = os.path.join(MODEL_DIR, f'{encoder_name}_{enc_dims_str}_{args.loss_func}' + \
                                    f'_xent{args.xent_lambda}' + \
                                    f'_mselambda{args.mse_lambda}' + \
                                    f'_caelambda{args.cae_lambda}' + \
                                    f'_{args.optimizer}_{args.scheduler}' + \
                                    f'_lr{args.learning_rate}_decay{args.lr_decay_rate}' + \
                                    f'_{args.sampler}_b{args.bsize}_e{args.epochs}_mdate{args.mdate}.pth')
        logging.info(f'Initial encoder model: ENC_MODEL_PATH {ENC_MODEL_PATH}')
    
    X_train_final = X_train
    y_train_final = y_train
    y_train_binary_final = y_train_binary
    upsample_values = None
    
    logging.info(f'upsample_values {upsample_values}')
    logging.info(f'X_train_final.shape: {X_train_final.shape}')
    logging.info(f'y_train_final.shape: {y_train_final.shape}')
    logging.info(f'y_train_binary_final.shape: {y_train_binary_final.shape}')
    logging.info(f'y_train_final labels: {np.unique(y_train_final)}')
    logging.info(f'y_train_final: {Counter(y_train_final)}')

    # if we are training our own model
    # make all singleton families the same as "unknown"
    if args.encoder != None and args.encoder.startswith('simple-enc-mlp') == True:
        counted_y_train = Counter(y_train)
        singleton_families = [family for family, count in counted_y_train.items() if count == 1]
        logging.info(f'Singleton families: {singleton_families}')
        logging.info(f'Number of singleton families: {len(singleton_families)}')

        # unknown_idx = y_train[np.where(all_train_family == 'unknown')[0][0]]
        # # make all singleton families the same as "unknown"
        # y_train_final = np.array([y_train[i] if family not in singleton_families else unknown_idx for i, family in enumerate(y_train)])
        # logging.info(f'After merging singleton families: X_train.shape, {X_train.shape}, y_train_final.shape, {y_train_final.shape}')
        # logging.info(f'After merging singleton families: {Counter(y_train_final)}')
        
        X_train_final = np.array([X_train[i] for i, family in enumerate(y_train) if family not in singleton_families])
        y_train_final = np.array([y_train[i] for i, family in enumerate(y_train) if family not in singleton_families])
        y_train_binary_final = np.array([y_train_binary[i] for i, family in enumerate(y_train) if family not in singleton_families])
        # y_train_final = y_train
        # y_train_binary_final = y_train_binary
        all_train_family = np.array([all_train_family[i] for i, family in enumerate(y_train) if family not in singleton_families])
        logging.info(f'After removing singleton families: X_train_final.shape, {X_train_final.shape}, y_train_final.shape, {y_train_final.shape}')
        logging.info(f'After removing singleton families: {Counter(y_train_final)}')


    # train the encoder model if it does not already exist.
    # train mlp encoder in the classifier training step
    if args.encoder in ['cae', 'enc', 'simple-enc-mlp']:
        if args.retrain_first == True or not os.path.exists(ENC_MODEL_PATH):
            s1 = time.time()
            train_encoder_func(args, encoder, X_train_final, y_train_final, y_train_binary_final, \
                            optimizer, args.epochs, ENC_MODEL_PATH, adjust = True, save_best_loss = False, \
                            save_snapshot = args.snapshot)
            e1 = time.time()
            logging.info(f'Training Encoder model time: {(e1 - s1):.3f} seconds')
            
            # logging.info(f'Loading the best model {ENC_MODEL_PATH}...')
            # state_dict = torch.load(ENC_MODEL_PATH)
            # encoder.load_state_dict(state_dict['model'])

            logging.info('Saving the model...')
            save_model(encoder, optimizer, args, args.epochs, ENC_MODEL_PATH)
            logging.info(f'Training Encoder model finished: {ENC_MODEL_PATH}')
        else:
            logging.info('Loading the model...')
            state_dict = torch.load(ENC_MODEL_PATH)
            encoder.load_state_dict(state_dict['model'])
    elif args.encoder == 'mlp':
        train_classifier(args, encoder, X_train_final, y_train_final, y_train_binary_final, \
                        mlp_optimizer, args.mlp_epochs, ENC_MODEL_PATH, \
                        save_best_loss = False, multi = args.multi_class)
        logging.info('Saving the model...')
        save_model(encoder, mlp_optimizer, args, args.epochs, ENC_MODEL_PATH)
        logging.info(f'Training Encoder model finished: {ENC_MODEL_PATH}')

    """
    Select the classifier model.
    """
    # prepare X_feat and X_feat_tensor if they are embeddings
    if args.cls_feat == 'encoded':
        X_train_tensor = torch.from_numpy(X_train).float()
        if torch.cuda.is_available():
            X_train_tensor = X_train_tensor.cuda()
            X_feat_tensor = encoder.cuda().encode(X_train_tensor)
            X_train_feat = X_feat_tensor.cpu().detach().numpy()
        else:
            X_train_feat = encoder.encode(X_train_tensor).numpy()
    else:
        # args.cls_feat == 'input'
        X_train_feat = X_train

    if args.classifier in ['simple-enc-mlp'] or args.classifier == args.encoder:
        # we have already trained it as the sample selection model.
        classifier = encoder
        CLS_MODEL_PATH = ENC_MODEL_PATH
        cls_gpu = True
    elif args.classifier == 'svm':
        if args.encoder != 'mlp' and args.multi_class == True:
            classifier = CalibratedClassifierCV(LinearSVC(random_state=0, max_iter=10000, C=args.svm_c)).fit(X_train_feat, y_train)
            MODEL_DIR = os.path.join(SAVED_MODEL_FOLDER, train_dataset_name)
            CLS_MODEL_PATH = os.path.join(MODEL_DIR, f'svm_classifier_multiclass_feat_{args.cls_feat}_c{args.svm_c}_{args.mdate}.joblib')
            logging.info(f'Saving linear SVM model to {CLS_MODEL_PATH}...')
        else:
            ### Train a binary-class linear classifier
            classifier = CalibratedClassifierCV(LinearSVC(random_state=0, max_iter=10000, C=args.svm_c)).fit(X_train_feat, y_train_binary)
            MODEL_DIR = os.path.join(SAVED_MODEL_FOLDER, train_dataset_name)
            CLS_MODEL_PATH = os.path.join(MODEL_DIR, f'svm_classifier_{args.cls_feat}_c{args.svm_c}_{args.mdate}.joblib')
            logging.info(f'Saving linear SVM model to {CLS_MODEL_PATH}...')
        dump(classifier, CLS_MODEL_PATH)
        cls_gpu = False
    elif args.classifier == 'gbdt':
        # assume binary
        dtrain = xgb.DMatrix(X_train_feat, label=y_train_binary)
        param = {'max_depth': args.max_depth, 'eta': args.eta, 'eval_metric': 'error'}
        evallist = [(dtrain, 'train'), ]
        xgbmodel = xgb.train(param, dtrain, num_boost_round = args.num_round, \
                            evals = evallist)
        classifier = xgboost_wrapper(xgbmodel, binary = True)
        CLS_MODEL_PATH = os.path.join(MODEL_DIR, f'xgb_{args.cls_feat}_maxdepth{args.max_depth}_round{args.num_round}_eta{args.eta}_{args.mdate}.json')
        logging.info(f'Saving XGBoost model to {CLS_MODEL_PATH}...')
        xgbmodel.save_model(CLS_MODEL_PATH)
        cls_gpu = False
    elif args.classifier == 'mlp':
        if args.encoder == 'mlp':
            classifier = encoder
            CLS_MODEL_PATH = ENC_MODEL_PATH
        else:
            if args.multi_class == True:
                output_dim = NUM_CLASSES
            else:
                output_dim = BIN_NUM_CLASSES
            if args.cls_feat == 'encoded':
                mlp_dims = utils.get_model_dims('MLP', enc_dims[-1], args.mlp_hidden, output_dim)
            else:
                mlp_dims = utils.get_model_dims('MLP', NUM_FEATURES, args.mlp_hidden, output_dim)
            classifier = MLPClassifier(mlp_dims)
            
            # set Adam optimizer
            mlp_optimizer = torch.optim.Adam(classifier.parameters(), lr=args.mlp_lr)

            MODEL_DIR = os.path.join(SAVED_MODEL_FOLDER, train_dataset_name)
            utils.create_folder(MODEL_DIR)
            mlp_dims_str = str(mlp_dims).replace(' ', '').replace(',', '-').replace('[', '').replace(']', '') # remove extra symbols

            CLS_MODEL_PATH = os.path.join(MODEL_DIR, f'MLP_{mlp_dims_str}_feat_{args.cls_feat}' + \
                                        f'_dropout{args.mlp_dropout}' + \
                                        f'_lr{args.mlp_lr}' + \
                                        f'_b{args.mlp_batch_size}_e{args.mlp_epochs}_mdate{args.mdate}.pth')
            if args.cls_feat == 'encoded':
                # 使用 os.path.basename 提取文件名，避免分隔符问题
                enc_model_filename = os.path.basename(ENC_MODEL_PATH)
                # 去掉 .pth 并拼接文件名
                CLS_MODEL_PATH = os.path.splitext(CLS_MODEL_PATH)[0] + '_' + enc_model_filename
            
            logging.info(f'Initial MLP Classifier model: CLS_MODEL_PATH {CLS_MODEL_PATH}')
        cls_gpu = True
    elif args.classifier == 'res':
        if args.encoder == 'mlp':
            classifier = encoder
            CLS_MODEL_PATH = ENC_MODEL_PATH
        else:
            if args.multi_class == True:
                output_dim = NUM_CLASSES
            else:
                output_dim = BIN_NUM_CLASSES
            classifier = ResClassifier(2)

            # set Adam optimizer
            mlp_optimizer = torch.optim.Adam(classifier.parameters(), lr=args.mlp_lr)

            MODEL_DIR = os.path.join(SAVED_MODEL_FOLDER, train_dataset_name)
            utils.create_folder(MODEL_DIR)
            mlp_dims = 521
            mlp_dims_str = str(mlp_dims).replace(' ', '').replace(',', '-').replace('[', '').replace(']',
                                                                                                     '')  # remove extra symbols

            CLS_MODEL_PATH = os.path.join(MODEL_DIR, f'MLP_{mlp_dims_str}_feat_{args.cls_feat}' + \
                                          f'_dropout{args.mlp_dropout}' + \
                                          f'_lr{args.mlp_lr}' + \
                                          f'_b{args.mlp_batch_size}_e{args.mlp_epochs}_mdate{args.mdate}.pth')
            if args.cls_feat == 'encoded':
                CLS_MODEL_PATH = CLS_MODEL_PATH.split('.pth')[0] + '_' + ENC_MODEL_PATH.split('/')[-1]

            logging.info(f'Initial MLP Classifier model: CLS_MODEL_PATH {CLS_MODEL_PATH}')
        cls_gpu = True
    else:
        raise Exception(f'The classifier {args.classifier} is not supported yet.')

    if args.classifier not in ['svm', 'gbdt'] and (args.classifier != args.encoder or (args.classifier == 'mlp' and args.encoder == 'mlp')):
        if args.cls_retrain == 1 or not os.path.exists(CLS_MODEL_PATH):
            s1 = time.time()
            train_classifier(args, classifier, X_train_feat, y_train, \
                            y_train_binary, mlp_optimizer, args.mlp_epochs, \
                            CLS_MODEL_PATH, save_best_loss = False, multi = args.multi_class)
            e1 = time.time()
            logging.info(f'Training Classifier model time: {(e1 - s1):.3f} seconds')
            # logging.info(f'Loading the best model {CLS_MODEL_PATH}...')
            # state_dict = torch.load(CLS_MODEL_PATH)
            # classifier.load_state_dict(state_dict['model'])
            logging.info('Saving the model...')
            save_model(classifier, mlp_optimizer, args, args.mlp_epochs, CLS_MODEL_PATH)
            logging.info(f'Training Classifier model finished: {CLS_MODEL_PATH}')
        else:
            # load the existing model
            logging.info('Loading the Classifier model...')
            state_dict = torch.load(CLS_MODEL_PATH)
            classifier.load_state_dict(state_dict['model'])

    # save training acc
    fout = open(args.result, 'w')
    fout.write('date\tTPR\tTNR\tFPR\tFNR\tACC\tPREC\tF1\tAUT(F1)\tF1-D\n')
    fout_retrain = open(args.result.split('.csv')[0] + '_retrain.csv', 'w')
    fout_retrain.write('date\tTPR\tTNR\tFPR\tFNR\tACC\tPREC\tF1\tAUT(F1)\n')
    fam_out = open(args.result.split('.csv')[0] + '_family.csv', 'w')
    fam_out.write('Month\tNew\tFamily\tFNR\tCnt\tIndex\n')
    stat_out = open(args.result.split('.csv')[0] + '_stat.csv', 'w')
    stat_out.write('date\tTotal\tTP\tTN\tFP\tFN\n')
    lifetime_path = args.result.split('.csv')[0] + '_pred.csv'
    if args.classifier != 'res':
        encoder.eval()
    if args.classifier != 'svm':
        classifier.eval()
    eval_classifier(args, classifier, args.train_end, X_train_feat, y_train_binary, all_train_family_original, \
                    train_families, fout, fout_retrain, fam_out, stat_out, retrain=True, gpu=cls_gpu,
                    multi=args.eval_multi)
    eval_classifier(args, classifier, args.train_end, X_train_feat, y_train_binary, all_train_family_original,
                    train_families, \
                    fout, fout_retrain, fam_out, stat_out, gpu=cls_gpu, multi=args.eval_multi)

    sample_out = open(args.result.split('.csv')[0]+'_sample.csv', 'w')
    sample_out.write('date\tCount\tIndex\tTrue\tPred\tFamily\tScore\n')
    sample_out.flush()
    sample_explanation = open(args.result.split('.csv')[0]+'_sample_explanation.csv', 'w')
    sample_explanation.write('date\tCorrect\tWrong\tBenign\tMal\tNew_fam_cnt\tNew_fam\tUnique_fam\n')
    sample_explanation.flush()
    # sample_score_out = open(args.result.split('.csv')[0]+'_sample_scores.csv', 'w')
    # sample_score_out.write('date\tIndex\tWrong\tFamily\tScore\tDistance\tPred\n')
    # sample_score_out.flush()
    
    """
    Set up the selector.
    """
    if args.al == True:
        strategy = 'strategy'
        if args.rand == True:
            strategy += '_rand'
        if args.unc == True:
            strategy += '_unc'
            if args.multi_class == True:
                strategy += '_multi'
                selector = MultiUncertainPredScoreSelector(classifier)
            else:
                selector = UncertainPredScoreSelector(classifier)
        if args.ood == True:
            strategy += '_ood'
            selector = OODSelector(encoder)
        if args.transcend == True:
            strategy += '_transcend'
            if args.criteria == 'cred':
                crit = 'cred'
            elif args.criteria == 'conf':
                crit = 'conf'
            else:
                # args.criteria == 'cred+conf'
                crit = 'cred+conf'
            selector = TranscendSelector(encoder, crit=crit)
        if args.local_pseudo_loss == True:
            strategy += '_local_pseudo_loss'
            strategy += f'_{args.reduce}'
            selector = LocalPseudoLossSelector(encoder)
        if args.encoder_retrain == True:
            strategy += '_encretrain'

        # cold or warm setup
        if args.cold_start == True:
            strategy += '_cold'
        else:
            strategy += f'_warm_{args.al_optimizer}_wlr{args.al_epochs}_we{args.warm_learning_rate}'
        strategy += f'_count{args.count}'
    
    """
    Step (5): Go over each month in the test_valuation range.
    """
    # saved_train_feature_file = os.path.join('data', args.data, f'{train_dataset_name}_selected_training_features.json')
    start = dt.datetime.strptime(args.test_start, '%Y-%m')
    end = dt.datetime.strptime(args.test_end, '%Y-%m')
    cur_month = start

    month_loop_cnt = 0
    prev_train_size = X_train.shape[0]
    cur_sample_indices = []

    ### 为保存更新数据集而做的设置 ###
    all_train_family = all_train_family_original
    init_date = np.full(X_train.shape[0], '2012')
    all_date = init_date
    X_test_all = np.empty((0, NUM_FEATURES), dtype=X_train.dtype)
    index_map = []  # 用于记录索引与原始行的映射关系
    f1_ds = [] # 存储每个月的F1-D

    while cur_month <= end:
        """
        Step (6): Load test_valuation data.
        """
        cur_month_str = cur_month.strftime('%Y-%m')

        MODEL_DIR = os.path.join(SAVED_MODEL_FOLDER, train_dataset_name)
        NEW_ENC_MODEL_PATH = os.path.join(MODEL_DIR, f'encoder_{args.encoder}_{args.method}_{cur_month_str}_retrain.pth')
        NEW_CLS_MODEL_PATH = os.path.join(MODEL_DIR, f'classifier_{args.classifier}_{args.method}_{cur_month_str}_retrain.pth')
        
        if args.data.startswith('tesseract'):
            X_test, y_test, all_test_family = data.load_range_dataset_w_benign(args, args.data, cur_month_str, cur_month_str)
        else:
            X_test, y_test, y_test_family = data.load_range_dataset_w_benign(args, args.data, cur_month_str, cur_month_str)
            # all_test_family has 'benign'
            ben_test_len = X_test.shape[0] - y_test_family.shape[0]
            y_ben_test_family = np.full(ben_test_len, 'benign')
            all_test_family = np.concatenate((y_test_family, y_ben_test_family), axis=0)
        
        logging.info(f'X_test.shape {X_test.shape}')
        logging.info(f'y_test.shape {y_test.shape}')
        logging.info(f'y_test_family.shape {y_test_family.shape}')

        y_test_binary = np.array([1 if item != 0 else 0 for item in y_test])

        # compute the embedding once
        # this could be used to retrain the classifier
        X_test_tensor = torch.from_numpy(X_test).float()
        if args.encoder != None:
            if torch.cuda.is_available():
                X_test_feat_tensor = encoder.cuda().encode(X_test_tensor.cuda())
                X_test_encoded = X_test_feat_tensor.cpu().detach().numpy()
            else:
                X_test_encoded = encoder.encode(X_test_tensor).numpy()
        
        if args.cls_feat == 'encoded':
            X_test_feat = X_test_encoded
        else:
            X_test_feat = X_test

        # Only month_loop_cnt == 0 will we update the accum data with new month data
        if args.accumulate_data == True and month_loop_cnt == 0:
            if cur_month_str == '2013-01':
                X_test_accum = X_test
                y_test_accum = y_test
                all_test_family_accum = all_test_family
                X_test_accum_feat = X_test_feat # for the classifier
            else:
                X_test_accum = np.concatenate((X_test_accum, X_test), axis=0)
                y_test_accum = np.concatenate((y_test_accum, y_test), axis=0)
                all_test_family_accum = np.concatenate((all_test_family_accum, all_test_family), axis=0)
                X_test_accum_feat = np.concatenate((X_test_accum_feat, X_test_feat), axis=0) # for the classifier
        elif month_loop_cnt == 0:
            X_test_accum = X_test
            y_test_accum = y_test
            all_test_family_accum = all_test_family
            X_test_accum_feat = X_test_feat # for the classifier
        
        y_test_binary_accum = np.array([1 if item != 0 else 0 for item in y_test_accum])
        
        """
        Evaluate the test_valuation performance.
        """
        logging.info(f'Testing on {cur_month_str}')
        # 设置计算F1-D的基准时间点
        if args.F1D < 10:
            threshold_date = dt.datetime.strptime(f'2013-0{args.F1D}', '%Y-%m')
        else:
            threshold_date = dt.datetime.strptime(f'2013-{args.F1D}', '%Y-%m')
        # 判断并执行程序
        if cur_month > threshold_date:
            month = cur_month - relativedelta(months=args.F1D)
            month_str = month.strftime('%Y-%m')
            if args.data.startswith('tesseract'):
                X, y, family = data.load_range_dataset_w_benign(args, args.data, month_str, month_str)
            else:
                X, y, family = data.load_range_dataset_w_benign(args, args.data, month_str, month_str)
            y_binary = np.array([1 if item != 0 else 0 for item in y])

            if args.classifier == 'res':
                test_loader = DataLoader(X, batch_size=5000, shuffle=False)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                classifier.to(device)
                if torch.cuda.is_available():
                    with torch.no_grad():
                        y_pred_past = []
                        for features in test_loader:
                            features = features.float()
                            features = features.to(device)
                            # 将featrues转换为32位
                            # features = features.cuda()
                            features = features.reshape(-1, 34, 34)
                            features = features.unsqueeze(0)
                            features = features.permute(1, 0, 2, 3)
                            y_hat = classifier.predict(features)
                            # 从计算图抽出。
                            y_hat = y_hat.detach().cpu().numpy().tolist()
                            y_pred_past.extend(y_hat)
                y_pred_past = np.array(y_pred_past)
            else:
                X_tensor_past = torch.from_numpy(X).float()
                if args.encoder != None:
                    if torch.cuda.is_available():
                        X_tensor_past_feat = encoder.cuda().encode(X_tensor_past.cuda())
                        X_test_past_encoded = X_tensor_past_feat.cpu().detach().numpy()
                    else:
                        X_test_past_encoded = encoder.encode(X_tensor_past).numpy()

                if args.cls_feat == 'encoded':
                    X_test_past_feat = X_test_past_encoded
                else:
                    X_test_past_feat = X

                if cls_gpu == True:
                    X_tensor_past = torch.from_numpy(X_test_past_feat).float()
                    if torch.cuda.is_available():
                        X_tensor_past = X_tensor_past.cuda()
                        y_pred_past = classifier.cuda().predict(X_tensor_past)
                        y_pred_past = y_pred_past.cpu().detach().numpy()
                    else:
                        y_pred_past = classifier.predict(X_tensor_past).numpy()
                else:
                    y_pred_past = classifier.predict(X_test_past_feat)
            if args.multi_class == True:
                # process multi-class y_pred to binary
                # if y_pred is 0, it is 0, otherwise it is 1
                y_pred_past_bin = np.where(y_pred_past == 0, 0, 1)
            else:
                y_pred_past_bin = y_pred_past
            tpr, tnr, fpr, fnr, acc, precision, f1_d = get_model_stats(y_binary, y_pred_past_bin,
                                                                       multi_class=args.eval_multi)
            f1_ds.append(f1_d)

        '''
        测试集当月测试集在当月下的性能表现
        '''
        if args.classifier != 'res':
            encoder.eval()
        if args.classifier != 'svm':
            classifier.eval()
        y_test_pred, neg_by_fam, family_to_idx = eval_classifier(args, classifier, cur_month_str, X_test_feat,
                                                                 y_test_binary, all_test_family, train_families,
                                                                 fout, fout_retrain, fam_out, stat_out, gpu=cls_gpu,
                                                                 multi=args.eval_multi)

        logging.info(f'测试新型恶意家族生存状态')
        '''
        测试新型家族恶意样本的生存周期
        '''
        # 读取所有新型恶意家族样本

        family_path = args.result.split('.csv')[0]+'_family.csv'
        family_df = pd.read_csv(family_path, sep="\t")

        # 筛选出 New 列值为 True 的行
        # new_data = family_df[family_df['New'] == True]
        # 筛选 Month 列中与 cur_month_str 相同的行
        filtered_data = family_df[family_df['Month'] == cur_month_str]
        # 遍历筛选后的数据行
        for _, row in filtered_data.iterrows():
            indices = eval(row['Index'])  # 将 Index 转换为列表
            extracted_data = X_test[indices, :]  # 提取对应的样本数据
            X_test_all = np.vstack((X_test_all, extracted_data))  # 将数据拼接到 X_test_all
            for i in indices:
                index_map.append({
                    'cur_month_str': cur_month_str,
                    'Month': row['Month'],
                    'Index': i
                })
        logging.info(f'新型恶意家族样本数量{len(index_map)},{X_test_all.shape}')

        # 更改标签与家族信息，同时更改评估函数，让该函数在评估新型家族恶意样本时，不在文件中记录
        if args.classifier == 'res':
            classifier.eval()
            test_loader = DataLoader(X_test_all, batch_size=5000, shuffle=False)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            classifier.to(device)
            if torch.cuda.is_available():
                with torch.no_grad():
                    y_pred_all = []
                    for features in test_loader:
                        features = features.float()
                        features = features.to(device)
                        # 将featrues转换为32位
                        # features = features.cuda()
                        features = features.reshape(-1, 34, 34)
                        features = features.unsqueeze(0)
                        features = features.permute(1, 0, 2, 3)
                        y_hat = classifier.predict(features)
                        # 从计算图抽出。
                        y_hat = y_hat.detach().cpu().numpy().tolist()
                        y_pred_all.extend(y_hat)
            y_pred_all = np.array(y_pred_all)
        else:
            X_tensor_all = torch.from_numpy(X_test_all).float()
            if args.encoder != None:
                if torch.cuda.is_available():
                    X_tensor_all_feat = encoder.cuda().encode(X_tensor_all.cuda())
                    X_test_all_encoded = X_tensor_all_feat.cpu().detach().numpy()
                else:
                    X_test_all_encoded = encoder.encode(X_tensor_all).numpy()

            if args.cls_feat == 'encoded':
                X_test_all_feat = X_test_all_encoded
            else:
                X_test_all_feat = X_test_all

            if cls_gpu == True:
                X_tensor_all = torch.from_numpy(X_test_all_feat).float()
                if torch.cuda.is_available():
                    X_tensor_all = X_tensor_all.cuda()
                    y_pred_all = classifier.cuda().predict(X_tensor_all)
                    y_pred_all = y_pred_all.cpu().detach().numpy()
                else:
                    y_pred_all = classifier.predict(X_tensor_all).numpy()
            else:
                y_pred_all = classifier.predict(X_test_all_feat)

        logging.info(f'预测标签数量{len(y_pred_all)}')

        # 准备结果并处理评估结果为 0 的数据
        # 首先将原始数据写入文件
        results = []
        for i, pred in enumerate(y_pred_all):
            results.append({
                'cur_month_str': cur_month_str,
                'Month': index_map[i]['Month'],
                'Index': index_map[i]['Index'],
                'pred': pred
            })

        # 判断文件是否存在
        if not os.path.exists(lifetime_path):
            # 如果文件不存在，写入时包含表头
            pd.DataFrame(results).to_csv(lifetime_path, sep='\t', index=False)
        else:
            # 如果文件已存在，追加写入，并且不包含表头
            pd.DataFrame(results).to_csv(lifetime_path, sep='\t', index=False, mode='a', header=False)

        # 然后提取预测值为 0 的数据
        remaining_X_test_all = []
        remaining_index_map = []
        for i, pred in enumerate(y_pred_all):
            if pred == 0:
                remaining_X_test_all.append(X_test_all[i])
                remaining_index_map.append(index_map[i])

        # 更新 X_test_all
        X_test_all = np.asarray(remaining_X_test_all, dtype=X_train.dtype).reshape(
            -1, NUM_FEATURES
        )
        index_map = remaining_index_map

        if args.accumulate_data == True and month_loop_cnt == 0:
            if cur_month_str == '2013-01':
                y_test_pred_accum = y_test_pred
            else:
                y_test_pred_accum = np.concatenate((y_test_pred_accum, y_test_pred), axis=0)
        elif month_loop_cnt == 0:
            y_test_pred_accum = y_test_pred
        """
        Step (7): Pick samples. Expand the training set.
        """
        if args.al == True and cur_month != end:
            # predict_proba returns ndarray of shape (n_samples, n_classes)
            if cls_gpu == True:
                X_test_accum_feat_tensor = torch.from_numpy(X_test_accum_feat).float()
                if args.classifier == 'res':
                    if torch.cuda.is_available():
                        X_test_accum_feat_tensor = X_test_accum_feat_tensor.reshape(-1, 34, 34)
                        X_test_accum_feat_tensor = X_test_accum_feat_tensor.unsqueeze(0)
                        X_test_accum_feat_tensor = X_test_accum_feat_tensor.permute(1, 0, 2, 3)
                        pred_scores_accum = classifier.cuda().predict_proba(X_test_accum_feat_tensor.cuda())
                        pred_scores_accum = pred_scores_accum.cpu().detach().numpy()
                    else:
                        pred_scores_accum = classifier.predict_proba(X_test_accum_feat_tensor)
                else:
                    if torch.cuda.is_available():
                        pred_scores_accum = classifier.cuda().predict_proba(X_test_accum_feat_tensor.cuda())
                        pred_scores_accum = pred_scores_accum.cpu().detach().numpy()
                    else:
                        pred_scores_accum = classifier.predict_proba(X_test_accum_feat_tensor)
            else:
                pred_scores_accum = classifier.predict_proba(X_test_accum_feat)
            test_offset = prev_train_size

            cluster_fname = args.result.split('.csv')[0] + '_clusters.csv'
            test_distances = {}

            if args.ood == True:
                sample_indices, sample_scores = selector.select_samples(X_train, y_train,
                                                                        X_test_accum,
                                                                        args.count)
            elif args.transcend == True:
                sample_indices, sample_scores = selector.select_samples(X_train, y_train,
                                                                        X_test_accum,
                                                                        args.count)
            elif args.unc == True:
                # Uncertainty sampling
                sample_indices, sample_scores = selector.select_samples(args, X_test_feat, y_test_pred_accum, args.count)
            elif args.local_pseudo_loss == True:
                total_epochs = 10
                sample_indices, sample_scores = selector.select_samples(args,
                                                                        X_train, y_train, y_train_binary,
                                                                        X_test_accum, y_test_pred_accum,
                                                                        total_epochs,
                                                                        test_offset,
                                                                        all_test_family_accum,
                                                                        args.count)
            else:
                raise ValueError('Unknown sampling method')
            #更新训练集中的家族列表
            selected_family = set(all_test_family_accum[sample_indices])
            train_families.update(selected_family)
                    
            """
            Step (8): expand the training set: X_train, y_train, etc.
            """
            # print out information about picked samples
            # $date-total, sample_cnt
            # month, idx, true label, predicted label, family label, OOD score
            cnt = 0
            for idx in sample_indices:
                try:
                    fam_label = all_test_family_accum[idx]
                except IndexError:
                    fam_label = 'benign'
                pred_label = int(y_test_pred_accum[idx])
                if args.classifier == 'gbdt':
                    sample_out.write('%s\t%d\t%d\t%s\t%.4f\t%s\t%.4f\n' % \
                                (cur_month_str, cnt, idx, y_test_binary_accum[idx], pred_scores_accum[idx], fam_label, sample_scores[idx]))
                else:
                    sample_out.write('%s\t%d\t%d\t%s\t%.4f\t%s\t%.4f\n' % \
                                (cur_month_str, cnt, idx, y_test_binary_accum[idx], pred_scores_accum[idx][pred_label], fam_label, sample_scores[idx]))
                cnt += 1
            sample_out.flush()
            # More detailed distribution of samples selected
            correct_pred = 0
            wrong_pred = 0
            fam_dict = defaultdict(lambda: 0)
            logging.info(f'y_test_binary_accum.shape, {y_test_binary_accum.shape}')
            logging.info(f'pred_scores_accum.shape, {pred_scores_accum.shape}')
            for idx in sample_indices:
                try:
                    fam_label = all_test_family_accum[idx]
                except IndexError:
                    fam_label = 'benign'
                true_label = y_test_binary_accum[idx]
                pred_label = int(y_test_pred_accum[idx])
                logging.info(f'{idx}, {fam_label}, {true_label}, {pred_label}, {pred_scores_accum[idx][pred_label]}')
                # correct/wrong predictions
                if true_label == pred_label:
                    correct_pred += 1
                else:
                    wrong_pred += 1
                # family count
                fam_dict[fam_label] += 1
            benign_num = fam_dict['benign']
            mal_num = cnt - benign_num
            new_families_lst = list(set(fam_dict.keys()) - set(all_train_family.flatten()))
            uniq_families_lst = list(fam_dict.keys())
            uniq_families = ",".join(uniq_families_lst)
            new_fam_cnt = 0
            for fam in new_families_lst:
                new_fam_cnt += fam_dict[fam]
            new_families_selected = ",".join(new_families_lst)
            sample_explanation.write('%s\t%d\t%d\t%d\t%d\t%d\t%s\t%s\n' % \
                (cur_month_str, correct_pred, wrong_pred, benign_num, mal_num, new_fam_cnt, new_families_selected, uniq_families))
            sample_explanation.flush()

            # add X_test[sample_indices] to training set
            X_train = np.concatenate((X_train, X_test_accum[sample_indices]), axis=0)
            y_train_binary = np.concatenate((y_train_binary, y_test_binary_accum[sample_indices]), axis=0)
            original_y = y_test_accum[sample_indices]
            # make new label start from max(y_train) + 1
            new_y = np.copy(original_y)
            new_label = max(y_train) + 1
            for idx, label in enumerate(original_y):
                if new_y_mapping.get(label, None) != None:
                    new_y[idx] = new_y_mapping[label]
                else:
                    new_y_mapping[label] = new_label
                    new_y[idx] = new_label
                    new_label += 1
            y_train = np.concatenate((y_train, new_y), axis=0)
            # y_train = np.concatenate((y_train, y_test_accum[sample_indices]), axis=0)
            logging.info(f'y_test_accum[sample_indices] {y_test_accum[sample_indices]}')
            logging.info(f'new_y {new_y}')

            all_train_family = np.concatenate((all_train_family, all_test_family_accum[sample_indices]), axis=0)

            # Remove selected samples from test_valuation data
            X_test_accum = np.delete(X_test_accum, sample_indices, axis=0)
            X_test_accum_feat = np.delete(X_test_accum_feat, sample_indices, axis=0)
            y_test_accum = np.delete(y_test_accum, sample_indices, axis=0)
            all_test_family_accum = np.delete(all_test_family_accum, sample_indices, axis=0)
            y_test_pred_accum = np.delete(y_test_pred_accum, sample_indices, axis=0)

            X_train_final = X_train
            y_train_final = y_train
            y_train_binary_final = y_train_binary
            upsample_values = None

            logging.info(f'upsample_values {upsample_values}')
            logging.info(f'X_train_final.shape: {X_train_final.shape}')
            logging.info(f'y_train_final.shape: {y_train_final.shape}')
            logging.info(f'y_train_binary_final.shape: {y_train_binary_final.shape}')
            logging.info(f'y_train_final labels: {np.unique(y_train_final)}')
            logging.info(f'y_train_final: {Counter(y_train_final)}')

            # # if we are training our own model
            # # make all singleton families the same as "unknown"
            # if args.encoder != None and args.encoder.startswith('simple-enc-mlp') == True:
            #     counted_y_train = Counter(y_train)
            #     singleton_families = [family for family, count in counted_y_train.items() if count == 1]
            #     logging.info(f'Singleton families: {singleton_families}')
            #     logging.info(f'Number of singleton families: {len(singleton_families)}')
            #     unknown_idx = y_train[np.where(all_train_family == 'unknown')[0][0]]
            #     # make all singleton families the same as "unknown"
            #     y_train_final = np.array([y_train[i] if family not in singleton_families else unknown_idx for i, family in enumerate(y_train)])
            #     logging.info(f'After merging singleton families: X_train.shape, {X_train.shape}, y_train.shape, {y_train.shape}')
            #     logging.info(f'After merging singleton families: {Counter(y_train_final)}')

            """
            Step (9): Retrain the sample selection model, e.g., Enc + MLP.
            """
            # Training the encoder model again
            logging.info(f'{args.encoder_retrain}')
            if args.encoder_retrain == True:
                # whether we use the same optimizer or according to al_optimizer
                if args.al_optimizer == None:
                    # use the same optimizer as the first model
                    logging.info(f'Active learning using optimizer {args.optimizer}')
                    pass
                elif args.al_optimizer == 'adam':
                    # Adam optimizer
                    optimizer_func = torch.optim.Adam
                    logging.info(f'Active learning using optimizer {args.al_optimizer}')
                elif args.al_optimizer == 'sgd':
                    # SGD optimizer
                    optimizer_func = torch.optim.SGD
                    logging.info(f'Active learning using optimizer {args.al_optimizer}')

                if args.cold_start == True:
                    # re-initialize the encoder
                    if args.encoder == 'simple-enc-mlp':
                        # Enc + MLP model
                        enc_dims = utils.get_model_dims('Encoder', NUM_FEATURES,
                                            args.enc_hidden, NUM_CLASSES)
                        mlp_dims = utils.get_model_dims('MLP', enc_dims[-1], args.mlp_hidden, BIN_NUM_CLASSES)
                        enc_classifier = SimpleEncClassifier(enc_dims, mlp_dims)

                        # original learning rate for cold start
                        optimizer = optimizer_func(enc_classifier.parameters(), lr=args.learning_rate)

                        MODEL_DIR = os.path.join(SAVED_MODEL_FOLDER, train_dataset_name)
                        utils.create_folder(MODEL_DIR)
                        enc_dims_str = str(enc_dims).replace(' ', '').replace(',', '-').replace('[', '').replace(']', '') # remove extra symbols

                        logging.info(f'Initial Simple Enc Classifier model: NEW_ENC_MODEL_PATH {NEW_ENC_MODEL_PATH}')

                        encoder = enc_classifier
                    elif args.encoder == 'cae':
                        enc_dims = utils.get_model_dims('Encoder', NUM_FEATURES,
                                            args.enc_hidden, NUM_CLASSES)
                        encoder = CAE(enc_dims)
                        encoder_name = 'cae'
                        # original learning rate for cold start
                        optimizer = optimizer_func(encoder.parameters(), lr=args.learning_rate)
                    elif args.encoder == 'enc':
                        enc_dims = utils.get_model_dims('Encoder', NUM_FEATURES,
                                            args.enc_hidden, NUM_CLASSES)
                        encoder = Enc(enc_dims)
                        encoder_name = 'enc'
                        # original learning rate for cold start
                        optimizer = optimizer_func(encoder.parameters(), lr=args.learning_rate)
                    else:
                        raise Exception(f"Re-initializing encoder {args.encoder} not implemented yet.")
                    al_total_epochs = args.epochs
                else:
                    # warm start learning rate, e.g., 0.1 * args.learning_rate
                    optimizer = optimizer_func(encoder.parameters(), lr=args.warm_learning_rate)
                    al_total_epochs = args.al_epochs

                # both cold start and warm start below
                if args.classifier != 'res':
                    encoder.train()
                if args.classifier != 'svm':
                    classifier.train()
                if args.encoder != None and args.encoder != 'mlp':
                    s2 = time.time()
                    logging.info('Training Encoder model...')
                    train_encoder_func(args, encoder, X_train_final, y_train_final, y_train_binary_final,
                                    optimizer, al_total_epochs, NEW_ENC_MODEL_PATH,
                                    weight = None,
                                    adjust = True, warm = not args.cold_start, save_best_loss = False,
                                    ids_exposure_count = len(sample_indices))
                    e2 = time.time()
                    logging.info(f'Training Encoder model time: {(e2 - s2):.3f} seconds')
                    save_model(encoder, optimizer, args, args.mlp_epochs, NEW_ENC_MODEL_PATH)
                elif args.encoder == 'mlp':
                    s2 = time.time()
                    if args.cold_start == True:
                        if args.multi_class == True:
                            output_dim = len(np.unique(y_train))
                        else:
                            output_dim = BIN_NUM_CLASSES
                        if args.cls_feat == 'encoded':
                            mlp_dims = utils.get_model_dims('MLP', enc_dims[-1], args.mlp_hidden, output_dim)
                        else:
                            mlp_dims = utils.get_model_dims('MLP', NUM_FEATURES, args.mlp_hidden, output_dim)
                        classifier = MLPClassifier(mlp_dims)
                        mlp_optimizer = torch.optim.Adam(classifier.parameters(), lr=args.mlp_lr)
                        mlp_total_epochs = args.mlp_epochs
                    else:
                        mlp_optimizer = torch.optim.Adam(classifier.parameters(), lr=args.mlp_warm_lr)
                        mlp_total_epochs = args.mlp_warm_epochs
                    logging.info('Training MLP Encoder model...')
                    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.mlp_lr)

                    train_classifier(args, encoder, X_train_final, y_train_final, y_train_binary_final, \
                                    optimizer, args.mlp_epochs, NEW_ENC_MODEL_PATH, \
                                    save_best_loss = False, multi = args.multi_class)
                    e2 = time.time()
                    logging.info(f'Training MLP Encoder model time: {(e2 - s2):.3f} seconds')
                    save_model(encoder, optimizer, args, args.mlp_epochs, NEW_ENC_MODEL_PATH)

            """
            Retrain the classifier if it's different from the encoder
            """
            # this is to retrain the classifier
            if args.cls_feat == 'encoded':
                X_train_tensor = torch.from_numpy(X_train).float()
                if torch.cuda.is_available():
                    X_train_tensor = X_train_tensor.cuda()
                    X_feat_tensor = encoder.cuda().encode(X_train_tensor)
                    X_train_feat = X_feat_tensor.cpu().detach().numpy()
                else:
                    X_train_feat = encoder.encode(X_train_tensor).numpy()
            else:
                # args.cls_feat == 'input'
                X_train_feat = X_train

            if args.classifier != args.encoder:
                if args.classifier == 'svm':
                    if args.encoder != 'mlp' and args.multi_class == True:
                        classifier.fit(X_train_feat, y_train)
                        logging.info(f'Saving linear SVM model to {NEW_CLS_MODEL_PATH}...')
                    else:
                        ### Train a linear classifier
                        classifier.fit(X_train_feat, y_train_binary)
                        logging.info(f'Saving linear SVM model to {NEW_CLS_MODEL_PATH}...')
                    dump(classifier, NEW_CLS_MODEL_PATH)
                elif args.classifier == 'gbdt':
                    # assume binary
                    dtrain = xgb.DMatrix(X_train_feat, label=y_train_binary)
                    param = {'max_depth': args.max_depth, 'eta': args.eta, 'eval_metric': 'error'}
                    evallist = [(dtrain, 'train'), ]
                    xgbmodel = xgb.train(param, dtrain, num_boost_round = args.num_round, \
                                        evals = evallist)
                    classifier = xgboost_wrapper(xgbmodel, binary = True)
                    logging.info(f'Saving XGBoost model to {NEW_CLS_MODEL_PATH}...')
                    xgbmodel.save_model(NEW_CLS_MODEL_PATH)
                elif (args.classifier == 'mlp' and args.encoder != 'mlp') or args.classifier == 'res':
                    s1 = time.time()
                    # Retraining from scratch with sample weights
                    if args.cold_start == True:
                        if args.multi_class == True:
                            output_dim = len(np.unique(y_train))
                        else:
                            output_dim = BIN_NUM_CLASSES
                        if args.cls_feat == 'encoded':
                            mlp_dims = utils.get_model_dims('MLP', enc_dims[-1], args.mlp_hidden, output_dim)
                        else:
                            mlp_dims = utils.get_model_dims('MLP', NUM_FEATURES, args.mlp_hidden, output_dim)
                        classifier = MLPClassifier(mlp_dims)
                        mlp_optimizer = torch.optim.Adam(classifier.parameters(), lr=args.mlp_lr)
                        mlp_total_epochs = args.mlp_epochs
                    else:
                        mlp_optimizer = torch.optim.Adam(classifier.parameters(), lr=args.mlp_warm_lr)
                        mlp_total_epochs = args.mlp_warm_epochs
                    logging.info('Training Classifier model...')
                    train_classifier(args, classifier, X_train_feat, y_train, y_train_binary, \
                                    mlp_optimizer, mlp_total_epochs, NEW_CLS_MODEL_PATH, \
                                    weight = None, save_best_loss = False, multi = args.multi_class)
                    e1 = time.time()
                    logging.info(f'Training Classifier model time: {(e1 - s1):.3f} seconds')
                    save_model(classifier, mlp_optimizer, args, args.mlp_epochs, NEW_CLS_MODEL_PATH)
        prev_train_size = X_train.shape[0]
        if args.classifier != 'res':
            encoder.eval()
        if args.classifier != 'svm':
            classifier.eval()
        eval_classifier(args, classifier, cur_month_str, X_test_feat, y_test_binary, all_test_family, train_families,
                        fout, fout_retrain, fam_out, stat_out, retrain=True, gpu=cls_gpu, multi=args.eval_multi)
        # increment to next month
        cur_month += relativedelta(months=1)
    
    """
    结束撰写结果文件
    """
    logging.info('概念漂移自适应完成，计算新型恶意家族生存周期')
    # 读取预测结果文件
    df = pd.read_csv(lifetime_path, sep='\t')

    # 根据 Month 和 Index 分组，统计列 'Value' 中 0 的个数
    zero_counts = df.groupby(['Month', 'Index'])['pred'].apply(lambda x: (x == 0).sum()).reset_index(name='long')

    date_index_long_path = args.result.split('.csv')[0] + '_lifetime.csv'
    zero_counts.to_csv(date_index_long_path, sep='\t', index=False)
    fout.close()
    fam_out.close()
    sample_out.close()
    stat_out.close()
    # sample_score_out.close()
    sample_explanation.close()
    # 读取文件
    with open(args.result, 'r') as fin:
        lines = fin.readlines()
    # 修改内容
    with open(args.result, 'w') as fout:
        for i, line in enumerate(lines):
            line = line.strip()  # 去除换行符
            if 1 < i < len(f1_ds) + 2:
                # 添加对应行的新数据
                fout.write(f"{line}\t{f1_ds[i - 2]:.4f}\n")
            elif i == 0:
                fout.write(f"{line}\n")
            else:
                fout.write(f"{line}\tnan\n")
    return

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    logging.info(f'time elapsed: {end - start} seconds')
