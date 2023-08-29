import os
import re
import time
import random
import copy
import numpy as np
import multiprocessing as mp
import matplotlib
import matplotlib.pyplot as plt
import pickle
from sklearn import neighbors
from math import log
from tqdm import tqdm
from sklearn.mixture import GaussianMixture


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type='average'):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type == 'none':
            fmtstr = ''
        elif self.summary_type == 'average':
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type == 'sum':
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type == 'count':
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class AdaNS_sampler(object):
    def __init__(self, boundaries, minimum_num_good_samples, random_loguniform=False):
        # minimum number of good samples (b) used to find the value of \alpha for each iteration
        self.minimum_num_good_samples = minimum_num_good_samples
        assert self.minimum_num_good_samples>0, "minimum number of good samples must be greater than zero"
        
        # shape of boundaries: <d, 2>. Specifies the minimum and maximum allowed value of the hyperparameter per dimension.
        self.boundaries = boundaries
        self.dimensions = len(boundaries)
        
        self.all_samples = np.zeros((0, self.dimensions))
        self.all_scores = np.zeros(0)
        self.all_subsamples = []
        
        self.good_samples = np.zeros(0)
        
        # maximum score through all iterations seen so far
        self.max_score = 0
        self.alpha_t = 0

        self.random_sampler = self.sample_loguniform if random_loguniform else self.sample_uniform
        print('---- random sampler is ', self.random_sampler)

    
    def sample_uniform(self, num_samples=1):
        '''
        function to sample unifromly from all the search-space
            - num_samples: number of samples to take
        '''
        if num_samples>0:
            sample_vectors = np.random.uniform(self.boundaries[:,0], self.boundaries[:,1], size=(num_samples, self.dimensions))
            sample_vectors = np.unique(sample_vectors, axis=0)
            while len(sample_vectors) < num_samples:
                count = num_samples - len(sample_vectors)
                sample_vectors = np.concatenate((sample_vectors, np.random.uniform(self.boundaries[:,0], self.boundaries[:,1], size=(count, self.dimensions))))
                sample_vectors = np.unique(sample_vectors, axis=0)
        else:
            sample_vectors = np.zeros((0, self.dimensions))

        return sample_vectors


    def sample_loguniform(self, num_samples=1):
        '''
        function to sample unifromly in the log domain from all the search-space
            - num_samples: number of samples to take
        '''
        if num_samples>0:
            sample_vectors = np.random.uniform([0]*self.dimensions, [log(x,10) for x in self.boundaries[:,1]], size=(num_samples, self.dimensions))
            sample_vectors = np.unique(sample_vectors, axis=0)
            while len(sample_vectors) < num_samples:
                count = num_samples - len(sample_vectors)
                sample_vectors = np.concatenate((sample_vectors, np.random.uniform([0]*self.dimensions, [log(x,10) for x in self.boundaries[:,1]], size=(count, self.dimensions))))
                sample_vectors = np.unique(sample_vectors, axis=0)
        else:
            sample_vectors = np.zeros((0, self.dimensions))
            
        return np.power(10, sample_vectors)
    

    def update_good_samples(self, alpha_t=None):
        '''
        function to update the list of good samples after evaluating a new batch of samples
            - alpha_max: \alpha_max parameter
        '''
        self.max_score = np.max(self.all_scores)
        
        if alpha_t is not None:
            score_thr = alpha_t * self.max_score
        else:
            score_thr = self.alpha_t * self.max_score
        
        self.good_samples = self.all_scores>=score_thr
    

    def configure_alpha(self, alpha_max=1.0, verbose=False):
        '''
        function to determine \alpha based on current good samples
            - alpha_max: \alpha_max
        '''
        if np.sum(self.good_samples)<self.minimum_num_good_samples:
            self.max_score = np.max(self.all_scores)
            alpha_t = alpha_max

            if self.max_score==0:
                sorted_args = np.argsort(self.all_scores)[::-1]
                indices = sorted_args[:self.minimum_num_good_samples]
                self.good_samples[indices] = True

            else:
                itr = 0
                while np.sum(self.good_samples)<self.minimum_num_good_samples and itr<1000:
                    if self.max_score < 0:
                        alpha_t = alpha_t + 0.05
                    else:
                        alpha_t = alpha_t - 0.05

                    self.update_good_samples(alpha_t)
                    itr += 1

                if np.sum(self.good_samples)<self.minimum_num_good_samples:
                    sorted_args = np.argsort(self.all_scores)[::-1]
                    alpha_t = self.all_scores[sorted_args[self.minimum_num_good_samples-1]] / self.all_scores[sorted_args[0]]
                    self.update_good_samples(alpha_t)
                    # indices = sorted_args[:self.minimum_num_good_samples]
                    # self.good_samples[indices] = True
                    # alpha_t = alpha_max
            
            assert np.sum(self.good_samples)>=self.minimum_num_good_samples, print(np.sum(self.good_samples), self.minimum_num_good_samples)
            if verbose:
                print('changing alpha_t to %0.2f' % (alpha_t))
            self.alpha_t = alpha_t

        return self.alpha_t
    

    def update(self, samples, scores, alpha_max, subsamples=None, save_path=None, **kwargs):    
        '''
        function to add newly evaluated samples to the history
            - samples: new samples
            - scores: evaluation score of new samples
            - alpha_max: current \alpha_max
        '''   
        self.all_samples = np.concatenate((self.all_samples, samples), axis=0)
        orig_count = self.all_samples.shape[0]
        self.all_samples, indices = np.unique(self.all_samples, axis=0, return_index=True)
        if self.all_samples.shape[0] < orig_count:
            print(f'==== Removing {orig_count-self.all_samples.shape[0]} duplicate samples')
        
        self.all_scores = np.concatenate((self.all_scores, scores), axis=0)[indices]
        assert len(self.all_samples)==len(self.all_scores)

        if subsamples is not None:
            self.all_subsamples += subsamples
            self.all_subsamples = (np.asarray(self.all_subsamples)[indices]).tolist()

            path_to_subsamples = [f.path for f in os.scandir(save_path) if f.is_dir()]
            for f in os.listdir(path_to_subsamples[0]):
                exp_idx = int(re.search('history_info_([0-9]+)', f).group(1))
                if exp_idx not in indices:
                    print('deleting', f)
                    os.remove(os.path.join(path_to_subsamples[0], f))
            for i, idx in enumerate(indices):        
                os.rename(os.path.join(path_to_subsamples[0], f'history_info_{idx}.pkl'), os.path.join(path_to_subsamples[0], f'history_info_{i}_.pkl')) 
            for i in range(len(indices)):
                os.rename(os.path.join(path_to_subsamples[0], f'history_info_{i}_.pkl'), os.path.join(path_to_subsamples[0], f'history_info_{i}.pkl'))         

        self.update_good_samples(alpha_max)


    def run_sampling(self, evaluator, num_samples, n_iter, minimize=False, alpha_max=1.0, early_stopping=np.Infinity,
        save_path='./sampling', n_parallel=1, executor=mp.Pool, param_names=None, verbose=False):
        '''
        Function to maximize given black-box function and save results to ./sampling/
            - evaluator : the objective function to be minimized
            - num_samples: number of samples to take at each iteration
            - n_iter: total number of sampling rounds
            - minimize: if set to True, the objective function will be minimized, otherwise maximized
            - alpha_max: \alpha_max parameter
            - early_stopping: the sampling loop will terminate after this many iterations without improvmenet
            - save_path: path to save the sampling history and other artifcats
            - n_parallel: number of parallel evaluations
            - executor: function to handle parallel evaluations
        returns: optimal hyperparameters
        '''
        coeff = -1 if minimize else 1

        # set up logging directory
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # adjusting the per-iteration sampling budget to the parallelism level
        if num_samples % n_parallel != 0:
            num_samples = num_samples - (num_samples % n_parallel) + n_parallel
            print('=> Sampling budget was adjusted to be ' + str(num_samples))
            self.minimum_num_good_samples = num_samples

        # apply the sampling algorithm
        best_samples = []
        best_scores = []
        best_subsamples = []
        alpha_vals = []
        num_not_improve = 0

        runtime_total = AverageMeter('Time', ':6.3f')
        runtime_simulation = AverageMeter('Time', ':6.3f')
        for iteration in range(n_iter):
            t0 = time.time()
            if iteration==0:
                samples = self.random_sampler(num_samples)
                origins = ['U']*len(samples)
                prev_max_score = self.max_score
            else:
                max_score_improv = self.max_score - prev_max_score
                prev_max_score = self.max_score
                samples, origins = self.sample(num_samples, verbose=verbose)

                # if the percentage improvement in the maximum score is smaller than 0.1%, activate early stopping
                if max_score_improv==0: #(max_score_improv/prev_max_score) < 0.001:
                    num_not_improve += 1 
                else:
                    num_not_improve = 0

            if num_not_improve > early_stopping:
                print('=> Activating early stopping')
                break

            if origins is not None:
                indices_to_keep = np.nonzero(np.asarray(origins) != 'P')
                samples = samples[indices_to_keep]

            # evaluate current batch of samples
            scores = np.zeros(len(samples))
            subsamples = []
            n_batches = len(samples)//n_parallel if len(samples)%n_parallel==0 else (len(samples)//n_parallel)+1
            t1 = time.time()
            # with tqdm(total=n_batches) as pbar:
            for i in range(n_batches):
                if n_parallel > 1:
                    batch_samples = samples[i*n_parallel:(i+1)*n_parallel]
                    with executor() as e:
                        batch_output = list(e.starmap(evaluator, zip(batch_samples, range(n_parallel))))
                    if isinstance(batch_output[0], tuple):
                        scores[i*n_parallel:(i+1)*n_parallel] = [batch_output[i][0] for i in range(len(batch_output))]
                        subsamples += [batch_output[i][1] for i in range(len(batch_output))]
                    else:
                        scores[i*n_parallel:(i+1)*n_parallel] = batch_output
                else:
                    output = evaluator(samples[i], port_id=0)
                    if isinstance(output, tuple):
                        scores[i] = output[0]
                        subsamples += [output[1]]
                    else:
                        scores[i] = output
                
                scores[i*n_parallel:(i+1)*n_parallel] *= coeff
                
                # pbar.update(1)
                # pbar.set_description('batch %s/%s (samples %s..%s/%s)'%(i+1, num_samples//n_parallel, i*n_parallel, \
                #                                 (i+1)*n_parallel, num_samples))    
            runtime_simulation.update(time.time()-t1)

            # change None MEVs to 0 and report the number
            count = 0
            for i, s in enumerate(scores):
                if np.isnan(s):
                    scores[i] = 0
                    count += 1
            if count > 0:
                print(f'============ converted {count}/{len(scores)} Nan score values to 0')
            assert np.sum(np.isnan(np.asarray(scores, dtype=float)))==0, np.sum(np.isnan(np.asarray(scores, dtype=float)))
            # check whether the MEV is None for the entire batch
            if np.sum(np.asarray(scores)==0) == len(scores) and len(subsamples)>0:
                return None, None, None

            subsamples = None if len(subsamples) == 0 else subsamples 
            self.update(samples=samples, scores=scores, origins=origins, alpha_max=alpha_max, subsamples=subsamples, save_path=save_path)

            # modify \alpha if necessary, to make sure there are enough "good" samples
            alpha = self.configure_alpha(alpha_max, verbose=verbose)
            alpha_vals.append(alpha)
            assert np.sum(self.good_samples) > 0, 'no good samples were found'

            # book-keeping
            best_scores.append(np.max(self.all_scores))
            id_best = np.argmax(self.all_scores)
            best_samples.append(self.all_samples[id_best])
            if subsamples is not None:
                best_subsamples.append(self.all_subsamples[id_best])
                id_to_keep = id_best
            
            runtime_total.update(time.time()-t0)
            if verbose:
                print('=> iter: %d, %d samples, average score: %.3f, best score: %0.3f' %(iteration, len(samples), np.mean(scores), best_scores[-1]))
                print('=> average score on %d good samples: %.3f' %(np.sum(self.good_samples), np.mean(self.all_scores[self.good_samples])))
                # print(self.all_samples[self.good_samples][:10])
                print('best sample:', self.all_samples[id_best])
                print('=> average simulation time per iteration: %.3f' % runtime_simulation.avg)
                print('=> average total time per iteration: %.3f' % runtime_total.avg)

        info = {'best_samples': np.asarray(best_samples),
                'best_scores': np.asarray(best_scores),
                'alpha_vals': alpha_vals,
                'all_samples': self.all_samples,
                'all_scores': self.all_scores,
                'good_samples':self.good_samples}
        
        id_best_overall = np.argmax(best_scores)
        best_sample_overall = best_samples[id_best_overall]

        if len(self.all_subsamples)>0:
            info['all_subsamples'] = self.all_subsamples
            info['best_subsamples'] = np.asarray(best_subsamples)

            with open(os.path.join(save_path, f'history_info.pkl'), 'wb') as f:
                pickle.dump(info, f)

            # input("Press Enter to continue...")
            path_to_subsamples = [f.path for f in os.scandir(save_path) if f.is_dir()]
            for f in os.listdir(path_to_subsamples[0]):
                exp_idx = int(re.search('history_info_([0-9]+)', f).group(1))
                if exp_idx == id_to_keep:
                    os.rename(os.path.join(path_to_subsamples[0], f), os.path.join(path_to_subsamples[0], 'history_info.pkl'))
                else:
                    os.remove(os.path.join(path_to_subsamples[0], f))
        else:
            exp_idx = 0
            while os.path.exists(os.path.join(save_path, f'history_info_{exp_idx}.pkl')):
                exp_idx += 1
            with open(os.path.join(save_path, f'history_info_{exp_idx}.pkl'), 'wb') as f:
                pickle.dump(info, f)

        if len(self.all_subsamples)>0:
            return best_sample_overall, best_scores[id_best_overall], best_subsamples[id_best_overall]
        
        else:
            return best_sample_overall, best_scores[id_best_overall]


class Gaussian_sampler(AdaNS_sampler):
    def __init__(self, boundaries, minimum_num_good_samples, random_loguniform=False,
                    u_random_portion=0.2, local_portion=0.4, cross_portion=0.4, pair_selection_method='random'):
        '''
            - u_random_portion: ratio of samples taken uniformly from the entire space
            - local_portion: ratio of samples taken from gaussian distributions using the "local" method
            - cross_portion: ratio of samples taken from gaussian distributions using the "cross" method
            
                (u_random + local_portion + cross_portion) = 1
            
            - pair_selection_method: how to select pairs for cross samples. Options: ['random','top_scores','top_and_nearest','top_and_furthest','top_and_random']
        '''

        super(Gaussian_sampler, self).__init__(boundaries, minimum_num_good_samples, random_loguniform=random_loguniform)

        # for each sample, specifies how it was created: 'U':uniformly 'L':gaussian local, 'C':gaussian cross
        self.origins = []

        self.u_random_portion = u_random_portion
        self.local_portion = local_portion
        self.cross_portion = cross_portion
        assert (u_random_portion + local_portion + cross_portion) == 1., 'sum of sampling portions must be 1 %f'%(u_random_portion + local_portion + cross_portion)

        self.pair_selection_method = pair_selection_method
        assert pair_selection_method in ['random','top_scores','top_and_nearest','top_and_furthest','top_and_random'], \
                        "pair selection should be one of ['random','top_scores','top_and_nearest','top_and_furthest','top_and_random']"


    def set_params(self, u_random_portion=None, local_portion=None, cross_portion=None, pair_selection_method=None):
        if u_random_portion is not None:
            self.u_random_portion = u_random_portion

        if local_portion is not None:
            self.local_portion = local_portion

        if cross_portion is not None:
            self.cross_portion = cross_portion

        if pair_selection_method is not None:
            self.pair_selection_method = pair_selection_method

        assert (self.u_random_portion + self.local_portion + self.cross_portion) == 1., 'sum of sampling portions must be 1'
        assert pair_selection_method in ['random','top_scores','top_and_nearest','top_and_furthest','top_and_random'], \
                                "pair selection should be one of ['random','top_scores','top_and_nearest','top_and_furthest','top_and_random']"


    def sample(self, num_samples, verbose=True, **kwargs):
        '''
        function to sample from the search-space
            - num_samples: number of samples to take
        '''
        if num_samples==0:
            return np.zeros((0, self.dimensions)).astype(np.int32), []

        data = self.all_samples[self.good_samples]
        assert len(np.unique(data, axis=0))==data.shape[0], (len(np.unique(data, axis=0)), data.shape[0])

        scores = self.all_scores[self.good_samples] - np.min(self.all_scores[self.good_samples])
        # scores = self.all_scores[self.good_samples] - np.min(self.all_scores)
        avg_good_scores = np.mean(scores)
        scores = scores + avg_good_scores
        assert np.sum(scores>=0)==len(scores)

        # "Local" samples created with gaussians
        local_sampling = int(num_samples*self.local_portion+0.001)
        
        max_all_dims = np.max(data, axis=0)
        min_all_dims = np.min(data, axis=0)
        
        gaussian_means = data
        gaussian_covs = np.asarray([((max_all_dims-min_all_dims)/4.0)**2 for _ in range(len(data))])
        gaussian_mix = GaussianMixture(n_components=data.shape[0], covariance_type='diag',
                                      weights_init=np.ones(data.shape[0])/data.shape[0], means_init=data)
        try:
            gaussian_mix.fit(X=data)
            gaussian_mix.means_ = gaussian_means
            gaussian_mix.covariances_ = gaussian_covs
            if np.sum(scores)==0:
                print('====== sum of scores was zero')
                gaussian_mix.weights_ = [1./len(scores)] * len(scores)
            else:
                gaussian_mix.weights_ = scores/np.sum(scores)
        
            if local_sampling>0:
                local_samples  = gaussian_mix.sample(n_samples=local_sampling)[0]
                local_samples  = np.clip(local_samples, self.boundaries[:,0], self.boundaries[:,1])
            else:
                local_samples = np.zeros((0, self.dimensions))
        except:
            local_samples  = self.random_sampler(num_samples=local_sampling)
        
        # "Cross" samples created with gaussians    
        cross_sampling = int(num_samples*self.cross_portion+0.001)
        cross_sampling = cross_sampling + np.mod(cross_sampling, 2)
    
        cross_samples = np.zeros((0, self.dimensions))
        if cross_sampling>0:
            pairs = self.get_pairs(num_pairs=cross_sampling)
            for pair in pairs:
                father = self.all_samples[pair[0]]
                mother = self.all_samples[pair[1]]
                gauss_mean = (father + mother)/2.0
                gauss_cov = (np.absolute(father-mother)/2.0)**2
                gauss_cov = np.diag(gauss_cov)
                sample = np.random.multivariate_normal(gauss_mean, gauss_cov)
                sample = np.clip(sample, self.boundaries[:,0], self.boundaries[:,1])
                sample = np.expand_dims(sample, axis=0)
                cross_samples = np.append(cross_samples, sample, axis=0)                            

        # "Uniform" samples chosen uniformly random
        random_sampling = int(num_samples*self.u_random_portion+0.001)   
        random_samples = self.random_sampler(num_samples=random_sampling)
               
        if verbose:
            print('sampled %d uniformly, %d with local gaussians, %d with cross gaussians'%(len(random_samples), len(local_samples), len(cross_samples)))
        
        origins_random = ['U'] * len(random_samples)
        origins_local = ['L'] * len(local_samples)
        origins_cross = ['C'] * len(cross_samples)
        origins = origins_random + origins_local + origins_cross
                
        sample_vectors = random_samples
        if local_sampling>0:
            sample_vectors = np.concatenate((sample_vectors, local_samples))
            
        if cross_sampling>0:
            sample_vectors = np.concatenate((sample_vectors, cross_samples))

        sample_vectors, indices = np.unique(sample_vectors, axis=0, return_index=True)
        origins = [origins[i] for i in indices]
        while len(sample_vectors) < num_samples:
            count = num_samples - len(sample_vectors)
            # print(f'adding {count} more random samples')
            sample_vectors = np.concatenate((sample_vectors, self.random_sampler(num_samples=count)))
            origins += ['U'] * count
            sample_vectors, indices = np.unique(sample_vectors, axis=0, return_index=True)
            origins = [origins[i] for i in indices]
        
        return sample_vectors, origins


    def update(self, samples, scores, origins, alpha_max, subsamples=None, save_path=None):    
        '''
        function to add newly evaluated samples to the history
            - samples: new samples
            - scores: evaluation score of new samples
            - origins: origin of new samples (zoom, genetic, gaussian-local, gaussian-cross, uniform-random)
            - alpha_max: current \alpha_max
        ''' 
        super(Gaussian_sampler, self).update(samples, scores, alpha_max, subsamples, save_path=save_path)
        self.origins += origins  
    

    def get_pairs(self, num_pairs):
        '''
        function to find pairs of vectors for Gaussian cross sampling
            - num_pairs: number of vector pairs to create
        '''
        pairs = []
        inds = np.where(self.good_samples)[0]
        if self.pair_selection_method == 'random':
            while(len(pairs)<num_pairs):
                choices = np.random.choice(inds, size=2, replace=False)
                pairs.append((choices[0], choices[1]))
        
        elif self.pair_selection_method == 'top_scores':
            scores = self.all_scores[self.good_samples]
            sum_score_mat = np.zeros((len(scores), len(scores)))
            for i, s1 in enumerate(scores[:-1]):
                for j in range(i+1, len(scores)):
                    s2 = scores[j]
                    sum_score_mat[i][j] = s1 + s2
            indices = np.argsort(sum_score_mat, axis=None)[::-1][:num_pairs]
            pair_inds = np.unravel_index(indices, dims=sum_score_mat.shape)
            for p0, p1 in zip(pair_inds[0], pair_inds[1]):
                pairs.append((inds[p0], inds[p1]))
        
        elif self.pair_selection_method == 'top_and_nearest':
            scores = self.all_scores[self.good_samples]
            samples = self.all_samples[self.good_samples]
            sorted_sample_ids = np.argsort(scores)[::-1] 
            distance_mat = np.zeros((len(scores), len(scores)))
            for i, s1 in enumerate(scores[:-1]):
                for j in range(i, len(scores)):
                    s2 = scores[j]
                    distance_mat[i][j] = np.sum((samples[i]-samples[j])**2)
                    distance_mat[j][i] = np.sum((samples[i]-samples[j])**2)
            for i in range(len(scores)):
                distance_mat[i,i] = np.Infinity
            pair_each_point = np.zeros(len(scores)).astype(np.int32)
            id0 = 0
            while(len(pairs)<num_pairs):
                candidates = distance_mat[sorted_sample_ids[id0]]
                closest = np.argsort(candidates)[pair_each_point[id0]]
                pairs.append((inds[sorted_sample_ids[id0]], inds[closest]))
                pair_each_point[id0] += 1
                id0 += 1
                id0 = np.mod(id0, len(scores))
        
        elif self.pair_selection_method == 'top_and_furthest':
            scores = self.all_scores[self.good_samples]
            samples = self.all_samples[self.good_samples]
            sorted_sample_ids = np.argsort(scores)[::-1] 
            distance_mat = np.zeros((len(scores),len(scores)))
            for i, s1 in enumerate(scores[:-1]):
                for j in range(i,len(scores)):
                    s2 = scores[j]
                    distance_mat[i][j] = np.sum((samples[i]-samples[j])**2)
                    distance_mat[j][i] = np.sum((samples[i]-samples[j])**2)        
            for i in range(len(scores)):
                distance_mat[i,i] = 0
            pair_each_point = np.zeros(len(scores)).astype(np.int32)
            id0 = 0
            while(len(pairs)<num_pairs):
                candidates = distance_mat[sorted_sample_ids[id0]]
                farest = np.argsort(candidates)[::-1][pair_each_point[id0]]
                pairs.append((inds[sorted_sample_ids[id0]], inds[farest]))
                pair_each_point[id0] += 1
                id0 += 1
                id0 = np.mod(id0, len(scores))
        
        elif self.pair_selection_method == 'top_and_random':
            scores = self.all_scores[self.good_samples]
            samples = self.all_samples[self.good_samples]
            sorted_sample_ids = np.argsort(scores)[::-1] 
            id0 = 0
            while len(pairs)<num_pairs:
                id1 = id0
                while(id1==id0):
                    id1 = np.random.randint(len(samples))
                pairs.append((inds[sorted_sample_ids[id0]], inds[sorted_sample_ids[id1]]))
                id0 += 1
                id0 = np.mod(id0, len(scores))
        
        return pairs


class RandomOrder_sampler(AdaNS_sampler):
    def __init__(self, length, minimum_num_good_samples, p_swap_min=0.0, p_swap_max=0.5, u_random_portion=0., parents_portion=0., 
                    swap_method='adjacent', groundtruth_order=None):
        '''
            - length: length of sequence to be reordered
            - u_random_portion: portion of samples taken uniformly at random 
        '''
        boundaries = [[0, length]] * length
        super(RandomOrder_sampler, self).__init__(boundaries, minimum_num_good_samples=minimum_num_good_samples)

        self.length = length 
        self.u_random_portion = u_random_portion
        self.parents_portion = parents_portion
        self.p_swap_min = p_swap_min
        self.p_swap_max = p_swap_max
        self.swap_method = swap_method
    
        assert u_random_portion + parents_portion <= 1., 'sum of portions must be <=1'
        
        self.groundtruth_order = groundtruth_order
        self.create_order_respresentation()
    
    def create_order_respresentation(self):
        self.order_repr = []
        for k, v in self.groundtruth_order.items():
            self.order_repr += [k] * len(v)
        # print('order representation:', self.order_repr)
    
    
    def check_constraints(self, sample):
        def check_order(array1, array2):
            if not isinstance(array2, list):
                array2 = array2.tolist()
            
            indices = np.asarray([])
            for element in array1:
                idx = array2.index(element)
                if not np.all(indices <= idx):
                    return False
                indices = np.append(indices, idx)
            return True
    
        for user_order in self.groundtruth_order.values():
            if len(user_order)==1:
                continue
            flag = check_order(user_order, sample)
            if not flag:
                return False

        return True
    
    
    def get_neighbors(self, sample):
        neighbors = []
        for idx in range(len(sample)):
            if idx+1 < len(sample):
                if sample[idx+1] == sample[idx]:
                    continue
                new_sample = copy.deepcopy(sample)
                new_sample[idx], new_sample[idx+1] = new_sample[idx+1], new_sample[idx]
                neighbors.append(new_sample)
        # print('original sample:', sample)
        # print('neighbors:', neighbors)
        return neighbors
    
    
    def sample_uniform(self, num_samples=1):
        '''
        function to sample unifromly from all the search-space
            - num_samples: number of samples to take
        '''
        def perm_generator(seq):
            seen = set()
            length = len(seq)
            while True:
                perm = tuple(random.sample(seq, length))
                if perm not in seen:
                    seen.add(perm)
                    yield perm

        if num_samples>0:
            rand_perms = perm_generator(self.order_repr)
            sample_vectors = np.asarray([next(rand_perms) for _ in range(num_samples)])
            assert len(np.unique(sample_vectors, axis=0)) == num_samples
        else:
            sample_vectors = np.zeros((0, self.dimensions))

        return sample_vectors
    
    
    def swap_adjacent_neighbors(self, sample, p_swap):
        '''
        function to swap the order in the sample
        '''
        n_swaps = max(1, int(p_swap * len(sample)))
        #--------------- do swap
        new_sample = copy.deepcopy(sample)
        for _ in range(n_swaps):
            neighbors = self.get_neighbors(new_sample)
            new_sample = random.choice(neighbors)
            # print('new_sample:', new_sample)

        return new_sample
    
    
    def swap_adjacent(self, sample, p_swap):
        '''
        function to swap the order in the sample
        '''
        #--------------- do swap
        for idx in range(len(sample)):
            p = np.random.rand()
            if p <= p_swap:
                #----------- swap with previous or next index
                # idx_swap = (idx + np.random.choice([-1, 1])) % self.length
                if idx==self.length-1:
                    idx_swap = idx-1
                elif idx==0:
                    idx_swap = idx+1
                else:
                    idx_swap = (idx + np.random.choice([-1, 1]))
                assert 0 <= idx_swap < self.length
                sample[idx], sample[idx_swap] = sample[idx_swap], sample[idx]

        return sample


    def swap_adjacent_subset(self, sample, p_swap):
        n_swaps = int(p_swap * self.length + 0.001)
        indices_to_swap = np.random.choice(self.length, size=n_swaps, replace=False)
        
        for idx in indices_to_swap:
            #----------- swap with previous or next index
            # idx_swap = (idx + np.random.choice([-1, 1])) % self.length
            if idx==self.length-1:
                idx_swap = idx-1
            elif idx==0:
                idx_swap = idx+1
            else:
                idx_swap = (idx + np.random.choice([-1, 1]))
            assert 0 <= idx_swap < self.length
            sample[idx], sample[idx_swap] = sample[idx_swap], sample[idx]

        return sample
    
    
    def sample(self, num_samples, verbose=True, **kwargs):
        '''
        function to sample from the search-space
            - num_samples: number of samples to take
            - portion_parents: optionally can keep a portion of samples for the next round
            - p_swap_max: upper bound on the per-element swapping
        '''
        num_samples_orig = num_samples
        if num_samples==0:
            return np.zeros((0, self.dimensions)).astype(np.int32), None

        # samples taken uniformly at random
        n_random_samples = int(self.u_random_portion * num_samples)
        if n_random_samples > 0.:
            random_samples = self.sample_uniform(num_samples=n_random_samples)
        else:
            random_samples = np.zeros((0, self.dimensions)).astype(np.int32)
        
        num_parents = int(self.parents_portion * num_samples)
        if num_parents > 0:
            indices_to_keep = np.argsort(self.all_scores[self.good_samples])[::-1][:num_parents]
            samples_to_keep = self.all_samples[self.good_samples][indices_to_keep]
            
        num_samples -= (num_parents + n_random_samples)
        residual_samples = 0
        if num_samples ==0:
            randorder_samples = np.zeros((0, self.dimensions)).astype(np.int32)
        else:
            if self.swap_method=='adjacent':
                swap_func = self.swap_adjacent
            elif self.swap_method=='adjacent_neighbor':
                swap_func = self.swap_adjacent_neighbors
            elif self.swap_method=='adjacent_subset':
                swap_func = self.swap_adjacent_subset
            else:
                raise NotImplementedError

            if num_samples >= int(np.sum(self.good_samples)+0.001):
                residual_samples = num_samples - int(np.sum(self.good_samples)+0.001)
                num_samples = int(np.sum(self.good_samples)+0.001)
                assert num_samples > 0
                randorder_samples = self.all_samples[self.good_samples][:num_samples]
                randorder_scores = self.all_scores[self.good_samples][:num_samples]
            else:
                inds = np.where(self.good_samples)[0]
                probs = (self.all_scores[self.good_samples] - np.min(self.all_scores[self.good_samples])) + 0.001
                # probs = (self.all_scores[self.good_samples] - np.min(self.all_scores)) + 0.001
                if np.sum(probs)==0:
                    probs = np.ones_like(probs)
                choices = np.random.choice(inds, size=num_samples, replace=False, p=probs/np.sum(probs))
                assert len(choices)==num_samples
                randorder_samples = np.asarray([self.all_samples[c] for c in choices])
                randorder_scores = np.asarray([self.all_scores[c] for c in choices])
                
            randorder_scores *= -1.
            randorder_scores = randorder_scores - np.min(randorder_scores)
            if np.sum(randorder_scores)==0:
                prob_swap = np.random.uniform(0., self.p_swap_max, size=num_samples)
            else:
                prob_swap = (randorder_scores / np.max(randorder_scores)) * (self.p_swap_max - self.p_swap_min)
                prob_swap += self.p_swap_min
            idx = 0
            while idx < num_samples:
                print('Swapping...')
                #----------------- swapping
                new_sample = swap_func(randorder_samples[idx], p_swap=prob_swap[idx])
                randorder_samples[idx] = new_sample
                idx += 1
                # if self.check_constraints(new_sample):
                #     randorder_samples[idx] = new_sample
                #     idx += 1

        if residual_samples > 0:
            random_samples = np.concatenate((random_samples, self.sample_uniform(num_samples=residual_samples)), axis=0)
            n_random_samples = random_samples.shape[0]
        
        origins = ['R'] * len(randorder_samples)
        if n_random_samples > 0.:
            randorder_samples = np.concatenate((random_samples, randorder_samples), axis=0)
            origins = ['U'] * len(random_samples) + origins
        if num_parents > 0.:
            randorder_samples = np.concatenate((samples_to_keep, randorder_samples), axis=0)
            origins = ['P'] * len(samples_to_keep) + origins

        if verbose:
            print('kept %d from before, sampled %d uniformly, %d with swapping'%(num_parents, n_random_samples, num_samples))

        assert len(origins)==randorder_samples.shape[0]

        assert randorder_samples.shape[0] == num_samples_orig, f'took {randorder_samples.shape[0]} samples but should be {num_samples_orig}'

        return randorder_samples, origins