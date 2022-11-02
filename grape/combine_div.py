import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import pickle

from models.gnn_model import get_gnn
from models.prediction_model import MLPNet
from models.prediction_model import MLPNet_binary
from utils.plot_utils import plot_curve, plot_sample
from utils.utils import build_optimizer, objectview, get_known_mask, mask_edge
from uci.uci_data import create_node

num = 240
out = 'gd98_240_2001.csv'

def imp_pre_d(data, args, log_path, device=torch.device('cpu')):

    model = get_gnn(data, args).to(device)

    #### args.impute_hiddens = 64
    if args.impute_hiddens == '':
        impute_hiddens = []
    else:
        impute_hiddens = list(map(int,args.impute_hiddens.split('_')))
    #### args.concat_states = False
    #### args.node_dim = 64
    if args.concat_states:
        input_dim = args.node_dim * len(model.convs) * 2
    else:
        input_dim = args.node_dim * 2 
        #### input_dim = 64*2=128

    #### no args.ce_loss ; hasattr(args,'ce_loss') == False
    if hasattr(args,'ce_loss') and args.ce_loss:
        output_dim = len(data.class_values)
    else:
        output_dim = 1
    # print(output_dim)

############################################
##### input_dim = 128 ; output_dim = 1 #####
############################################

    #### impute_hiddens = 64
    #### args.impute_activation = 'relu'
    #### args.dropout = 0.
    impute_model = MLPNet(input_dim, output_dim,
                            hidden_layer_sizes=impute_hiddens,
                            hidden_activation=args.impute_activation,
                            output_activation=None,
                            dropout=args.dropout).to(device)

    if args.predict_hiddens == '':
        predict_hiddens = []
    n_row, n_col = data.df_X.shape
    #### model = get_gnn(data, args).to(device)
    #### impute_model = MLPNet(input_dim, output_dim,...
    predict_model = MLPNet_binary(n_col, 1,
                           hidden_layer_sizes=predict_hiddens,
                           dropout=args.dropout).to(device)

    trainable_parameters = list(model.parameters()) \
                           + list(impute_model.parameters()) \
                           + list(predict_model.parameters())

    # build optimizer
    scheduler, opt = build_optimizer(args, trainable_parameters)

    # train
    Train_loss = []
    Train_loss_pr = []
    Train_loss_im = []
    Test_rmse = []
    Test_acc = []
    Lr = []

    x = data.x.clone().detach().to(device)
    y = data.y.clone().detach().to(device)
    #### 不會跑下面的if, 會跑else
    #print(hasattr(args,'split_sample'))  #True
    #print(args.split_sample > 0.) #False

    #### 跑這裡
    edge_index = data.edge_index.clone().detach().to(device)
    edge_attr = data.edge_attr.clone().detach().to(device)
    all_train_edge_index = data.train_edge_index.clone().detach().to(device)
    all_train_edge_attr = data.train_edge_attr.clone().detach().to(device)
    all_train_labels = data.train_labels.clone().detach().to(device)
    test_input_edge_index = all_train_edge_index
    test_input_edge_attr = all_train_edge_attr
    test_edge_index = data.test_edge_index.clone().detach().to(device)
    test_edge_attr = data.test_edge_attr.clone().detach().to(device)
    test_labels = data.test_labels.clone().detach().to(device)
    
    all_train_y_mask = data.train_y_mask.clone().detach().to(device)
    train_y_mask = all_train_y_mask.clone().detach()
    test_y_mask = data.test_y_mask.clone().detach().to(device)

    # print(hasattr(data,'class_values'))   
    # hasattr(data,'class_values') == False
    # 會跑這個

    train_edge_index, train_edge_attr, train_labels =\
         all_train_edge_index, all_train_edge_attr, all_train_labels
    print("train edge num is {}, test edge num is input {}, output {}"\
            .format(
            train_edge_attr.shape[0],
            test_input_edge_attr.shape[0], test_edge_attr.shape[0]))
    
    obj = dict()
    obj['args'] = args
    obj['outputs'] = dict()
    for epoch in range(args.epochs):
     
        model.train()
        impute_model.train()
        
        known_mask = get_known_mask(args.known, int(train_edge_attr.shape[0] / 2)).to(device)
        double_known_mask = torch.cat((known_mask, known_mask), dim=0)
        known_edge_index, known_edge_attr = mask_edge(train_edge_index, train_edge_attr, double_known_mask, True)

        opt.zero_grad()
        x_embd = model(x, known_edge_attr, known_edge_index)

        ## imputation
        pred_im = impute_model([x_embd[train_edge_index[0]], x_embd[train_edge_index[1]]])
        pred_train_im = pred_im[:int(train_edge_attr.shape[0] / 2),0]
        # pred_train_im.size() : 8585

        label_train_im = train_labels
        loss_im = F.mse_loss(pred_train_im, label_train_im)


        model.eval()
        impute_model.eval()

        with torch.no_grad():
            #test_input_edge_index = all_train_edge_index = data.train_edge_index.clone().detach().to(device)
            #test_input_edge_attr = all_train_edge_attr = data.train_edge_attr.clone().detach().to(device)
            x_embd = model(x, test_input_edge_attr, test_input_edge_index)
            pred_im = impute_model([x_embd[test_edge_index[0], :], x_embd[test_edge_index[1], :]])
            pred_test_im = pred_im[:int(test_edge_attr.shape[0] / 2),0]
            no_zero = []
            # print(pred_test_im.tolist())
            list_pred = pred_test_im.tolist()
            for z in range(num):
                if list_pred[z] >= 0 :
                    no_zero.append(list_pred[z])
                else :
                    no_zero.append(0.0)
            #print(no_zero)
            pred_test_im = torch.tensor(no_zero)
            label_test_im = test_labels
            # pred_test_im.size() : 967

            # perfomance of 967 test data
            mse = F.mse_loss(pred_test_im, label_test_im)
            test_rmse = np.sqrt(mse.item())
            l1 = F.l1_loss(pred_test_im, label_test_im)
            test_l1 = l1.item()

        # 合併 1. known(training)使用原本資料集的數值 2.unkown(test)的預測值
        known_0 = data.train_edge_mask.numpy()

        mask_unknown_0 = np.where(known_0==0)[0]
        unmask_known_0 = np.where(known_0==1)[0]
    
        com1 = dict(zip(mask_unknown_0, pred_test_im.numpy().tolist()))
        com2 = dict(zip(unmask_known_0, label_train_im.numpy().tolist()))
        com = {**com1, **com2}
        pred_train_test = []
        for s in range(9552):
            pred_train_test.append(com[s])
        pred_train_test = torch.tensor(pred_train_test)
        # print(pred_train_test[0:30])
        # pred_train_test : 合併後資料


        # 差補部分結束
        #------------------------------------------------------------------------------
        # 將原始training與預測test合併資料進行分類（pred_train_test）

        predict_model.train()
        # args.known = 1 - edge dropout rate
        known_mask_pr = get_known_mask(args.known, int(edge_attr.shape[0] / 2)).to(device)
        # edge_attr存放未被遮罩的特徵值（因為已經差補完，全部資料為已知）
        # int(edge_attr.shape[0] / 2) : 9552個
        # known_mask.size() : 9552
        double_known_mask = torch.cat((known_mask_pr, known_mask_pr), dim=0)
        
        known_edge_index_pr = edge_index
        known_edge_attr_pr = edge_attr

        # 因為要放入x_embd的東西是經過處理的x
        # 而此處的x為 pred_train_test
        pred_train_test = np.array(pred_train_test)
        pred_train_test = np.reshape(pred_train_test, (1194,8))
        pred_train_test = pd.DataFrame(pred_train_test, columns = ['0','1','2','3','4','5','6','7'])
        node_init_pr = create_node(pred_train_test, args.node_mode)
        xx_pr = torch.tensor(node_init_pr, dtype=torch.float)

        x_embd_pr = model(xx_pr, known_edge_attr, known_edge_index)
        X_pr = impute_model([x_embd_pr[edge_index[0, :int(n_row * n_col)]], x_embd_pr[edge_index[1, :int(n_row * n_col)]]])

        # print('X_pr')
        # print(X_pr.size())

        X_pr = torch.reshape(X_pr, [n_row, n_col])
        # print('reshape')
        # print(X_pr.size())

        pred_pr = predict_model(X_pr)[:, 0]
        # print('pred_pr')
        # print(pred_pr.size())

        pred_train_pr = pred_pr[train_y_mask]
        # print('pred_train_pr')
        # print(pred_train_pr.size())

        label_train_pr = y[train_y_mask]

        loss_fn = nn.BCELoss()
        loss_pr = loss_fn(pred_train_pr,label_train_pr)

        with torch.no_grad():
            x_embd_pr = model(xx_pr, edge_attr, edge_index)

            X_pr = impute_model([x_embd_pr[edge_index[0, :int(n_row * n_col)]], x_embd[edge_index[1, :int(n_row * n_col)]]])
    
            X_pr = torch.reshape(X_pr, [n_row, n_col])
            pred_pr = predict_model(X_pr)[:, 0]
            pred_test_pr = pred_pr[test_y_mask]
            label_test_pr = y[test_y_mask]

            pred_train_pr = pred_train_pr.detach().cpu().numpy()
            label_train_pr = label_train_pr.detach().cpu().numpy()
            pred_test_pr = pred_test_pr.detach().cpu().numpy()
            label_test_pr = label_test_pr.detach().cpu().numpy()

            tr_acc = (pred_train_pr.round() == label_train_pr).mean()
            test_acc = (pred_test_pr.round() == label_test_pr).mean()
            #print('test acc: ',test_acc)

        ### 問問問！！！！
        loss = loss_im / loss_pr
        loss.backward()
        opt.step()
        
        train_loss = loss.item()
        if scheduler is not None:
            scheduler.step(epoch)
        for param_group in opt.param_groups:
            Lr.append(param_group['lr'])


        Train_loss.append(train_loss)
        Train_loss_im.append(loss_im.item())
        Train_loss_pr.append(loss_pr.item())
        Test_rmse.append(test_rmse)
        Test_acc.append(test_acc)

        
        if epoch%1000 == 999:
            print()
            print('epoch: ', epoch)
            print('loss: ', train_loss)
            print('loss_imutation: ' , loss_im.item())
            print('loss_prediction: ' , loss_pr.item())
            print('test_rmse: ',test_rmse)
            print('test_acc: ',test_acc)

    # per = pd.DataFrame({'train loss':Train_loss,'Train loss im':Train_loss_im,'Train loss pr':Train_loss_pr,'test rmse':Test_rmse,'test_acc':Test_acc})
    # per.to_csv('perfomance_div.csv')



    pred_train_test_np = pred_train_test.to_numpy()
    ori_data = [pred_train_test_np[i]*(data.max_df-data.min_df)+data.min_df for i in range(1194)]
    ori_data_round = []
    for r in range(1194):
        ori_data_round.append([round(ori_data[r][e]) for e in range(8)])
    adv_sample = pd.DataFrame(ori_data_round)
    adv_sample['label'] = data.df_y
    pd.DataFrame(adv_sample).to_csv(out)

    # mask_unknown_0 = np.where(known_0==0)[0]
    # print('unknown: ', mask_unknown_0)
    #     test_acc = (pred_test_pr.round() == label_test_pr).mean()

            # Train_loss.append(train_loss)
            # Test_rmse.append(test_rmse)
            # Test_l1.append(test_l1)


            # if epoch%10==9:
            #     print('epoch: ', epoch)
            #     print('loss: ', train_loss)
                
            #     print('test rmse: ', test_rmse)
            #     print('test l1: ', test_l1)

                # print('test acc: ',test_acc)

    
    ###### 新加的code 為了求最後插補的真實值 #######################################
    # real_dfX_0 = data.df_np[:, :-1].tolist()
    # real_dfX_0 = np.array([item for sublist in real_dfX_0 for item in sublist])
    # known_0 = data.train_edge_mask.numpy()
    # mask_unknown_0 = np.where(known_0==0)[0]
    # unknown_real_data_0 = real_dfX_0[mask_unknown_0.tolist()]
    # # print(label_test)
    # # print(unknown_real_data_0)

    # label = mask_unknown_0%8 #會回傳是哪一個特徵值 為了要算回正規化之前的值
    # # pred_test_np = np.array(pred_test)
    # # pred_test_original = [pred_test[i]*(data.max_df[label[i]]-data.min_df[label[i]])+data.min_df[label[i]] for i in range(len(label))]
    # # pred_test_original = [round((pred_test[i]-data.min_df[label[i]])/(data.max_df[label[i]]-data.min_df[label[i]])) for i in range(len(label))]
    # pred_test_original = []
    # for i in range(len(label)):
    #     pred_test_original.append(round(pred_test[i]*(data.max_df[label[i]]-data.min_df[label[i]])+data.min_df[label[i]]))
    # # print(pred_test_original)

    # for i in range(len(label)):
    #     real_dfX_0[mask_unknown_0[i]] = pred_test_original[i]
    # # print(real_dfX_0)
    # real_dfX_1 = []
    # real_dfX_0 = np.array_split(real_dfX_0, 1194)
    # for i in range(len(real_dfX_0)):
    #     real_dfX_1.append(real_dfX_0[i].tolist())
    # real_dfX_1 = np.array(real_dfX_1)
    # # print(real_dfX_1)
    # adv_sample = pd.DataFrame(real_dfX_1)
    # adv_sample['label'] = data.df_y
    # pd.DataFrame(adv_sample).to_csv('adv_sample_impute.csv')
    # 已將修改過的值放回原本的資料集上，但是是一整個list，要變成df匯出
    ############################################################################















    # obj['curves'] = dict()
    # obj['curves']['train_loss'] = Train_loss
    # if args.valid > 0.:
    #     obj['curves']['valid_rmse'] = Valid_rmse
    #     obj['curves']['valid_l1'] = Valid_l1
    # obj['curves']['test_rmse'] = Test_rmse
    # obj['curves']['test_l1'] = Test_l1
    # obj['lr'] = Lr

    # obj['outputs']['final_pred_train'] = pred_train
    # obj['outputs']['label_train'] = label_train
    # obj['outputs']['final_pred_test'] = pred_test
    # obj['outputs']['label_test'] = label_test
    # pickle.dump(obj, open(log_path + 'result.pkl', "wb"))

    # if args.save_model:
    #     torch.save(model, log_path + 'model.pt')
    #     torch.save(impute_model, log_path + 'impute_model.pt')

    # # obj = objectview(obj)
    # plot_curve(obj['curves'], log_path+'curves.png',keys=None, 
    #             clip=True, label_min=True, label_end=True)
    # plot_curve(obj, log_path+'lr.png',keys=['lr'], 
    #             clip=False, label_min=False, label_end=False)
    # plot_sample(obj['outputs'], log_path+'outputs.png', 
    #             groups=[['final_pred_train','label_train'],
    #                     ['final_pred_test','label_test']
    #                     ], 
    #             num_points=20)
    # if args.save_prediction and args.valid > 0.:
    #     plot_sample(obj['outputs'], log_path+'outputs_best_valid.png', 
    #                 groups=[['best_valid_rmse_pred_test','label_test'],
    #                         ['best_valid_l1_pred_test','label_test']
    #                         ], 
    #                 num_points=20)
    # if args.valid > 0.:
    #     print("best valid rmse is {:.3g} at epoch {}".format(best_valid_rmse,best_valid_rmse_epoch))
    #     print("best valid l1 is {:.3g} at epoch {}".format(best_valid_l1,best_valid_l1_epoch))
