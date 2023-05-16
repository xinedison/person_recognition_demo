
import sys
import os
import numpy as np
import logging

class numpy_searcher(object):
    def __init__(self, reference_cnt, feature_dim):
        self.max_length = reference_cnt
        self.feature_dim = feature_dim

        self.data = np.zeros((self.max_length, self.feature_dim))
        self.k2idx = {}
        self.idx2k = {}
        self.free_idx = [i for i in range(reference_cnt, -1, -1)]

    def update(self, k, v):
        assert isinstance(v, list) or isinstance(v, tuple)
        try:
            if len(v) != self.feature_dim:
                logging.error('dim invalid: len(vale):{} , self.dims:{}'.format(len(v), self.feature_dim))
                raise Exception('update invalid feature dim {} for key {}'.format(len(v), k))
            if k not in self.k2idx:
                if len(self.free_idx) == 0:
                    logging.error('out of max length, no free position valid')
                    raise Exception('out of maxlength. max_length={}'.format(self.max_length))
                ref_idx = self.free_idx.pop()
                self.k2idx[k] = ref_idx
                self.idx2k[ref_idx] = k
            else:
                ref_idx = self.k2idx[k]
            self.data[ref_idx, :] = np.array(v)
            return ref_idx
        except Exception:
            logging.error('[update_error]', exc_info=True)
        return -1



    def topk(self, test_feats, topk=3):
        test_batch = np.array(test_feats)
        return self.numpy_topk(test_batch, self.data, topk) 
        

    def numpy_topk(self, test_feats, ref_datas, topk=3):
        epsilon = 1e-10
        feat_dot = np.dot(test_feats, np.transpose(ref_datas))
        #print("== feat dot shape ", feat_dot.shape)
        norm_test_feats = np.linalg.norm(test_feats, ord=2, axis=1, keepdims=True)
        norm_ref_datas = np.linalg.norm(ref_datas, ord=2, axis=1, keepdims=True)
        norm_dot = np.dot(norm_test_feats, np.transpose(norm_ref_datas))
        cos_distances = np.divide(feat_dot, norm_dot+epsilon)
        sorted_topk_idx = np.argsort(cos_distances, axis=1)[:, -topk:]
        sorted_topk_idx = sorted_topk_idx[:, ::-1] # reverse by decrease order
        ##sort_dis = np.sort(cos_distances, axis=1)[:, -topk:]
        topk_results = []
        for i in range(len(test_feats)):
            topk = []
            topk_dis = cos_distances[i, sorted_topk_idx[i]]
            for idx, distance in zip(sorted_topk_idx[i].tolist(), topk_dis.tolist()):
                #print("== topk distance ", distance)
                if distance > 0:
                    key = self.idx2k[idx]
                    topk.append((key, distance))
            topk_results.append(topk)
        return topk_results

if __name__ == '__main__':
    ref_cnt = 100000
    feat_dim = 512
    test_size = 8
    gpu_num = 2
    batch_size = 32
    topk = 3

    np_searcher = numpy_searcher(ref_cnt, feat_dim)

    data = np.random.random((ref_cnt, feat_dim))
    #print('data:\n', data.tolist())
    test_batch = np.random.random((test_size, feat_dim))
    #print('test_batch:\n', test_batch.tolist())

    # init numpy searcher to check correct
    for i, ref_feat in enumerate(data.tolist()):
        np_searcher.update('nk_{}'.format(i), ref_feat)
    

    test_feats = test_batch.tolist()
    np_topk = np_searcher.topk(test_feats, topk=topk)
    print('topk:\n', np_topk)





    


