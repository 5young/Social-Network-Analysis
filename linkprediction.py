def caculate_linkPrediction(T1_matrix, val_ones, val_zeros):
    #----------------------
    # 同時參考上下三角的資訊
    #----------------------
    T1_matrix = (T1_matrix + T1_matrix.T) / 2
    thrs = 0.0000001
    numOfNodes = T1_matrix.shape[0]
    # draw graph
    T1_G = nx.Graph()
    for i in range(numOfNodes):
        T1_G.add_node(i)

    for i in range(T1_matrix.shape[0]):
        for j in range(i+1, T1_matrix.shape[1]):
            if T1_matrix[i, j] > thrs:
                T1_G.add_edge(i, j, weight=T1_matrix[i, j])

    #------------------------------------
    # measure positive and negative edge
    #------------------------------------
    pos_node1 = []
    pos_node2 = []
    neg_node1 = []
    neg_node2 = []

    for i in range(T1_matrix.shape[0]):
        for j in range(i+1, T1_matrix.shape[1]):
            if T1_matrix[i, j] > thrs:
                pos_node1.append(i)
                pos_node2.append(j)
            else: 
                neg_node1.append(i)
                neg_node2.append(j)

    pos_dict = {"node1": pos_node1, "node2": pos_node2}
    neg_dict = {"node1": neg_node1, "node2": neg_node2}
    pos_df = pd.DataFrame(pos_dict)
    neg_df = pd.DataFrame(neg_dict)
    # print("total pos edge len: ", pos_df.shape[0])
    # print("total neg edge len: ", neg_df.shape[0])
    # input()

    # train_pos_df, test_pos_df = train_test_split(pos_df, test_size=0.0)
    # train_neg_df, test_neg_df = train_test_split(neg_df, test_size=0.0)

    train_pos_df = pos_df
    train_neg_df = neg_df

    '''看正負樣本各自數量決定要取多少筆數, 要選小的,
        不然會indexer out of bounds
    '''
    numOfSamples = min(train_pos_df.shape[0], train_neg_df.shape[0])
    # print(train_pos_df.shape[0], "\t", train_neg_df.shape[0])
    # numOfSamples = train_pos_df.shape[0]
    # numOfSamples = 200

    lp_X_train = np.zeros((2*numOfSamples, 1), float)
    lp_y_train = np.zeros((2*numOfSamples, 1), float)

    pos_cn_list = []
    neg_cn_list = []

    '''正的筆副得多時 會抱錯, 父的會超過比數因為沒這麼多筆ˋ
    '''
    for i in range(numOfSamples):
        # positive sample common neighbors
        pos_cn = len(list(
            nx.common_neighbors(T1_G, train_pos_df.iloc[i, 0], train_pos_df.iloc[i, 1])))
        pos_cn_list.append(pos_cn)

        # negative sample common neighbors
        neg_cn = len(list(
            nx.common_neighbors(T1_G, train_neg_df.iloc[i, 0], train_neg_df.iloc[i, 1])))
        neg_cn_list.append(neg_cn)


    lp_X_train[0:numOfSamples, 0] = pos_cn_list
    lp_X_train[numOfSamples:, 0] = neg_cn_list
    # print("pos cn feature: ", pos_cn_list[0:10])
    # print("neg cn feature: ", neg_cn_list[0:10])

    lp_y_train[0:numOfSamples] = 1
    lp_y_train[numOfSamples:] = 0

    # random permutation
    indices = np.random.permutation(lp_X_train.shape[0])
    lp_X_train = lp_X_train[indices]
    lp_y_train = lp_y_train[indices]

    #-------------------------------------- 
    #       link prediction training
    #-------------------------------------- 
    xgbc = XGBClassifier()
    xgbc.fit(lp_X_train.reshape((lp_X_train.shape[0], 1)), 
                lp_y_train.reshape((lp_y_train.shape[0])))

    #-------------------------------------- 
    #              TESTING
    #-------------------------------------- 
    lp_X_test = np.zeros((len(val_ones)+len(val_zeros), 1), float)
    pos_cn_list = []
    neg_cn_list = []
    pos_y_true = []
    neg_y_true = []

    for i in range(len(val_ones)):
        pos_cn = len(list(nx.common_neighbors(T1_G, val_ones[i, 0], val_ones[i, 1])))
        pos_cn_list.append(pos_cn)

        neg_cn = len(list(nx.common_neighbors(T1_G, val_zeros[i, 0], val_zeros[i, 1])))
        neg_cn_list.append(neg_cn)

        pos_y_true.append(1.0)
        neg_y_true.append(0.0)

    lp_X_test[0:len(val_ones), 0] = pos_cn_list
    lp_X_test[len(val_ones):, 0] = neg_cn_list

    y_predict = xgbc.predict(lp_X_test.reshape((lp_X_test.shape[0], 1)))
    y_true = np.concatenate((np.array(pos_y_true), np.array(neg_y_true)), axis=0)

    # print("y_predict ==> ", y_predict)
    # print("y_true ==> ", y_true)
    # input()

    f1_score = metrics.f1_score(y_true, y_predict)

    return f1_score
