import math
import os
import json
import pickle
import multiprocessing as mp
import nltk
from tqdm import tqdm
import re
import time
import numpy as np
import numba as nb
from numba.typed import List
from numba.types import bool_
from numba import njit, prange, objmode, cuda
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.spatial.distance import squareform
from sklearn.preprocessing import normalize
from collections import Counter
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def save_obj(obj, name):
    pickle.dump(obj,open(name + '.pkl', 'wb'), protocol=4)
    
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def save_progress(ProcessID, V):
    f = open('Proc'+str(ProcessID)+'.txt', "w")
    f.write(str(V))
    f.close()
    
#Процедура "распределения" n работ между nProcesses обработчиками
def chunkify(n,nProcesses):
    m = n // nProcesses
    for i in range(nProcesses):
        if i == nProcesses-1:
            yield i*m, n-1
        else:    
            yield i*m, (i+1)*m-1

# Из текста выделяются слова 
def text_to_tokens(text):
    return re.findall('[A-Za-z]+[-]?[A-Za-z]+', text)

def tokenlist_to_dict(token_list):
    token_dict = dict()
    for i in range(len(token_list)):
        token_dict[token_list[i]] = i
    return token_dict

def tokenlist_to_targetdict(token_list, target_token_mask):
    target_token_dict = dict()
    j = 0
    for i in range(len(token_list)):
        if target_token_mask[i]:
            target_token_dict[token_list[i]] = j
            j += 1
        else:
            target_token_dict[token_list[i]] = -1
    return target_token_dict

def tokens_to_inds(tokens, full_dict):
    n = len(tokens)
    inds = np.full(n, -1)
    for i in range(n):
        inds[i] = full_dict[tokens[i]]
    return inds

def tokens_to_target_inds(tokens, target_token_dict):
    n = len(tokens)
    inds = np.full(n, -1)
    for i in range(n):
        inds[i] = target_token_dict[tokens[i]]
    return inds

# Z-score Normalization (row wise for nonzero elements)
@nb.njit(parallel=True)
def zscore(X):
    n = X.shape[0]
    for i in nb.prange(n):
        mask = X[i] > 0.0
        X[i][np.logical_not(mask)] = -np.inf
        vals = X[i][mask]
        std = np.std(vals)        
        if std != 0.0:
            mean = np.mean(vals)
            X[i][mask] = (X[i][mask] - mean)/std
            X[i][i] = np.max(X[i][mask])


def generate_token_list(work_dir):
    token_list = list()
    for filename in tqdm(os.listdir(work_dir)):
        if filename.endswith('.json'):
            with open(os.path.join(work_dir, filename), 'r') as f:
                data = json.load(f)
                for d in data:
                    text = d['text']
                    raw_tokens = text_to_tokens(text)
                    lower_tokens = [w.lower() for w in raw_tokens]
                    unique_tokens = list(set(lower_tokens))
                    token_list.extend(unique_tokens)
    return list(set(token_list))



def parallel_tokenize(work_dir):
    
    nProcesses = mp.cpu_count() # Use all available processors
    
    def worker(StartInd, EndInd, files, procnum, return_dict):
        err = []
        R = list()
        for i in range(StartInd, EndInd+1):
            try:
                filename = files[i]
                if filename.endswith('.json'):
                    with open(os.path.join(work_dir, filename), 'r') as f:
                        data = json.load(f)
                        for d in data:
                            text = d['text']
                            raw_tokens = text_to_tokens(text)
                            lower_tokens = [w.lower() for w in raw_tokens]
                            unique_tokens = list(set(lower_tokens))
                            R.extend(unique_tokens)
                save_progress(procnum, (i-StartInd)/(EndInd-StartInd))
            except:
                err.append(i)
        return_dict[procnum] = R

    files = os.listdir(work_dir)

    manager = mp.Manager()
    return_dict = manager.dict()

    pool = mp.Pool(nProcesses)
    jobs = []

    # Create jobs
    n = len(files)
    t_start = time.time()
    i = 0
    for StartInd, EndInd in chunkify(n, nProcesses):
        jobs.append(pool.apply_async(worker, (StartInd, EndInd, files, i, return_dict)))
        i += 1

    # Wait for all jobs to finish
    for job in jobs:
        job.get()

    # Clean up
    pool.close()

    token_set = set()
    for i in range(len(return_dict)):
        token_set = token_set.union(return_dict[i])

    token_list = list(token_set)
    save_obj(token_list, 'token_list')

    t_end = time.time()

    print("Time:", t_end - t_start)

    return token_list




def parallel_count_docs(work_dir, token_list):
    
    nProcesses = mp.cpu_count() # Use all available processors
    token_dict = tokenlist_to_dict(token_list)
    
    def worker(StartInd, EndInd, files, token_dict, procnum, return_dict):
        err = []
        n_tokens = len(token_dict)
        document_number = np.zeros(n_tokens, dtype=int)
        for i in range(StartInd, EndInd+1):
            try:
                filename = files[i]
                if filename.endswith('.json'):
                    with open(os.path.join(work_dir, filename), 'r') as f:
                        data = json.load(f)
                        for d in data:
                            text = d['text']
                            raw_tokens = text_to_tokens(text)
                            lower_tokens = [w.lower() for w in raw_tokens]
                            unique_tokens = list(set(lower_tokens))
                            token_inds = tokens_to_inds(unique_tokens, token_dict)
                            document_number[token_inds] += 1
                save_progress(procnum, (i-StartInd)/(EndInd-StartInd))
            except:
                err.append(i)
        return_dict[procnum] = document_number

    files = os.listdir(work_dir)

    manager = mp.Manager()
    return_dict = manager.dict()

    pool = mp.Pool(nProcesses)
    jobs = []

    # Create jobs
    n = len(files)
    t_start = time.time()
    i = 0
    for StartInd, EndInd in chunkify(n, nProcesses):
        jobs.append(pool.apply_async(worker, (StartInd, EndInd, files, token_dict, i, return_dict)))
        i += 1

    # Wait for all jobs to finish
    for job in jobs:
        job.get()

    # Clean up
    pool.close()

    n_tokens = len(token_dict)
    document_number = np.zeros(n_tokens, dtype=int)
    for i in range(len(return_dict)):
        document_number += return_dict[i]

    save_obj(document_number, 'document_number')

    t_end = time.time()

    print("Time:", t_end - t_start)

    return document_number



def filter_target_tokens(token_dict, token_list, document_number, threshold=723):
    n_all_tokens = len(token_dict)
    all_token_inds = np.arange(n_all_tokens)
    all_tokens = np.array(token_list)
    target_token_mask = document_number > threshold

    n_target_tokens = np.sum(target_token_mask)
    target_tokens = all_tokens[target_token_mask]
    target_token_inds = np.arange(n_target_tokens)
    target_token_dict = tokenlist_to_targetdict(token_list, target_token_mask)

    save_obj(target_tokens, 'target_tokens')
    
    return target_tokens, target_token_dict


def collocations(inds, size):
    row = []
    col = []
    dists = []
    n = len(inds)
    for i in range(n):
        A = inds[i]
        if A > -1:
            for j in range(i+1,min(i+size+1,n)):
                B = inds[j]
                if B > -1:
                    if A > B:
                        row.append(A)
                        col.append(B)
                        dists.append(j-i)
                    else:
                        row.append(B)
                        col.append(A)
                        dists.append(j-i)
    return row, col, dists



def compute_collocation_stats(work_dir, n_target_tokens, target_token_dict, win_size=7):
    
    displacements = np.zeros((n_target_tokens, n_target_tokens), dtype=np.int32)
    counts = np.zeros((n_target_tokens, n_target_tokens), dtype=np.int32)
    
    for filename in tqdm(os.listdir(work_dir)):
        if filename.endswith('.json'):
            with open(os.path.join(work_dir, filename), 'r') as f:
                data = json.load(f)
                for d in data:
                    text = d['text']
                    raw_tokens = text_to_tokens(text)
                    lower_tokens = [w.lower() for w in raw_tokens]
                    inds = tokens_to_target_inds(lower_tokens, target_token_dict)
                    row, col, dists = collocations(inds, win_size)
                    for j in range(len(row)):
                        displacements[row[j], col[j]] += dists[j]
                        counts[row[j], col[j]] += 1
                        
    return displacements, counts




def parallel_collocation_matrix(work_dir, n_target_tokens, target_token_dict, win_size=7, nProcesses=8):
    
    def worker(StartInd, EndInd, files, n_target_tokens, target_token_dict, win_size, procnum, return_dict):
        err = []

        displacements = np.zeros((n_target_tokens, n_target_tokens), dtype=np.int32)
        counts = np.zeros((n_target_tokens, n_target_tokens), dtype=np.int32)

        for i in range(StartInd,EndInd+1):
            try:
                filename = files[i]
                if filename.endswith('.json'):
                    with open(os.path.join(work_dir, filename), 'r') as f:
                        data = json.load(f)
                        for d in data:
                            text = d['text']
                            raw_tokens = text_to_tokens(text)
                            lower_tokens = [w.lower() for w in raw_tokens]
                            inds = tokens_to_target_inds(lower_tokens, target_token_dict)
                            row, col, dists = collocations(inds, win_size)
                            for j in range(len(row)):
                                displacements[row[j], col[j]] += dists[j]
                                counts[row[j], col[j]] += 1

                save_progress(procnum, (i-StartInd)/(EndInd-StartInd))
            except:
                err.append(i)
        final = [displacements, counts]
        save_obj(final, 'chunk'+str(procnum))

        return_dict[procnum] = procnum
    
    files = os.listdir(work_dir)

    manager = mp.Manager()
    return_dict = manager.dict()

    pool = mp.Pool(nProcesses)
    jobs = []

    # Create jobs
    n = len(files)
    t_start = time.time()
    i = 0
    for StartInd, EndInd in chunkify(n, nProcesses):
        jobs.append(pool.apply_async(worker, (StartInd, EndInd, files, n_target_tokens, target_token_dict, win_size, i, return_dict)))
        i += 1

    # Wait for all jobs to finish
    for job in jobs:
        job.get()

    # Clean up
    pool.close()

    t_end = time.time()

    print("Time:", t_end - t_start)

    return return_dict


def aggregate_collocation_data(nProcesses=8, prefix='chunk'):

    for i in range(nProcesses):
        chunk = load_obj(f'{prefix}{i}')
        if i == 0:
            displacement_sums = chunk[0]
            counts = chunk[1]
        else:
            displacement_sums += chunk[0]
            counts += chunk[1]
        print(i)

    save_obj(displacement_sums, 'displacement_sums')
    save_obj(counts, 'counts')

    return displacement_sums, counts


def compute_similarities(n_target_tokens, counts, displacement_sums):

    mask = counts > 0
    displacements = np.zeros((n_target_tokens, n_target_tokens),  dtype=np.float32)
    displacements[mask] = displacement_sums[mask] / counts[mask]
    displacements = displacements + displacements.T - np.diag(np.diag(displacements)) # Copy lower triangle to upper triangle
    displacements[displacements == 0.0] = np.inf
    amounts = counts + counts.T - np.diag(np.diag(counts)) # Copy lower triangle to upper triangle
    similarities = amounts / displacements
    zscore(similarities)  # Assuming zscore is already defined

    save_obj(displacements, 'displacements')
    save_obj(amounts, 'amounts')
    save_obj(similarities, 'similarities')

    return displacements, amounts, similarities


def compute_proximity():

    proximity = load_obj('amounts')
    proximity_col_sums = np.sum(proximity, axis=0)
    proximity = proximity / proximity_col_sums
    proximity_row_sums = np.sum(proximity, axis=1, keepdims=True)
    proximity = proximity / proximity_row_sums
    np.fill_diagonal(proximity, np.max(proximity, axis=1))
    save_obj(proximity, 'proximity')
    
    return proximity


@nb.njit
def matrix_size(condensed_matrix):
    n = math.ceil((condensed_matrix.shape[0] * 2)**.5)
    if (condensed_matrix.ndim != 1) or (n * (n - 1) / 2 != condensed_matrix.shape[0]):
        raise ValueError('Incompatible vector size.')
    return n

@njit(inline='always')
def condensed_size(matrix_size):
    return int((matrix_size*(matrix_size-1))/2)

@njit(inline='always')
def condensed_idx(i,j,n):
    return int(i*n + j - i*(i+1)/2 - i - 1)

@njit(parallel=True)
def condensed_to_square(condensed):
    n = matrix_size(condensed)
    out = np.empty((n,n), dtype = condensed.dtype)
    for i in nb.prange(n):
        for j in range(i+1,n):
            out[i,j] = condensed[condensed_idx(i,j,n)]
            out[j,i] = out[i,j]
    return out
    
def pack(list_of_arrays):
    m = len(list_of_arrays)
    n_packed = np.sum([len(l) for l in list_of_arrays]).item()
    start_inds = np.full(m, -1)
    if m > 0:
        packed = np.full(n_packed, -1, dtype = list_of_arrays[0].dtype)
    else:
        packed = np.full(n_packed, -1)
    k = 0
    for i in range(m):
        L = list_of_arrays[i]
        l = len(L)
        start_inds[i] = k
        packed[k:k+l] = L
        k += l
    return start_inds, packed

def jaccard_pairwise_similarities(start_inds, sets_packed, gpu_device_id = -1, threads_per_block = (8, 16)):    
       
    @njit(parallel=True)
    def jaccard_pairwise_similarities_cpu(start_inds, sets_packed, out):

        def array_extract(ind, starts, packed):
            n = len(packed)
            a, b = -1, -1
            if (ind >= 0) and (ind < n):
                a = starts[ind]
                if ind < len(starts)-1:
                    b = starts[ind+1]
                else:
                    b = n
            return packed[a:b]
        
        if sets_packed.ndim != 1:
            raise ValueError('Incompatible vector size.')
        n = len(start_inds)
        for i in nb.prange(n):
            A = set(array_extract(i, start_inds, sets_packed))
            for j in range(i+1,n):
                B = set(array_extract(j, start_inds, sets_packed))
                den = min(len(A),len(B))
                if den > 0:
                    sim = len(A.intersection(B)) / den
                else:
                    sim = 0.0
                out[condensed_idx(i,j,n)] = sim

    @cuda.jit
    def jaccard_pairwise_similarities_gpu(start_inds, sets_packed, out):
        if sets_packed.ndim != 1:
            raise ValueError('Incompatible vector size.')
        n = len(start_inds)

        def array_extract(ind, starts, packed):
            n = len(packed)
            a, b = -1, -1
            if (ind >= 0) and (ind < n):
                a = starts[ind]
                if ind < len(starts)-1:
                    b = starts[ind+1]
                else:
                    b = n
            return packed[a:b]
        
        def jaccard_similarity(A, B):
            na = len(A)
            nb = len(B)
            k = 0
            for i in range(na):
                for j in range(nb):
                    if A[i] == B[j]:
                        k += 1
                        #break #Если убрать, то работает в два раза быстрее
            den = min(na,nb)
            if den > 0:
                sim = k / den
            else:
                sim = 0.0            
            return sim

        i,j = cuda.grid(2)
        if (i < n) and (j > i) and (j < n):
            A = array_extract(i, start_inds, sets_packed)
            B = array_extract(j, start_inds, sets_packed)
            out[condensed_idx(i,j,n)] = jaccard_similarity(A,B)
       
    n = len(start_inds)
    condensed_len = condensed_size(n)
    if gpu_device_id > -1:
        gpu = nb.cuda.select_device(gpu_device_id)
        grid_dim = (int(n/threads_per_block[0] + 1), int(n/threads_per_block[1] + 1))       
        stream = cuda.stream()
        out_gpu = cuda.device_array(shape=condensed_len, dtype = np.float32, stream = stream)
        start_inds_gpu = cuda.to_device(start_inds, stream=stream)
        sets_packed_gpu = cuda.to_device(sets_packed, stream=stream)        
        jaccard_pairwise_similarities_gpu[grid_dim, threads_per_block](start_inds_gpu, sets_packed_gpu, out_gpu)
        out = out_gpu.copy_to_host(stream = stream)
    else:
        out = np.full(condensed_len, 0.0, dtype = np.float32)
        jaccard_pairwise_similarities_cpu(start_inds, sets_packed, out)
       
    out2 = condensed_to_square(out)
    np.fill_diagonal(out2, 1.0)
    
    return out2




def pairwise_similarity(X):
    n = len(X)
    out = np.empty((n,n))
    np.fill_diagonal(out,1)
    for i in range(n):
        for j in range(i+1,n):
            out[i,j] = len(np.intersect1d(X[i],X[j])) / min(len(X[i]),len(X[j])) #Jaccard similarity
            out[j,i] = out[i,j]
    return out
    
@njit(inline='always')
def single_basis(sim, s, start):
    n = sim.shape[0]    
    out = np.array([start])
    C = np.delete(np.arange(n),start)
    while len(C) > 0:
        C = C[np.sum(sim[out][:,C] >= s, axis=0) == len(out)]
        if len(C) > 0:
            best = np.argsort(np.sum(sim[out][:,C], axis=0))[::-1][0]
            out = np.append(out,C[best])
            C = np.delete(C,best)
    return out
    
@njit(parallel=True)
def all_bases(S, s):    
   
    
    n = S.shape[0]
    inds = np.arange(n)
    out = [np.array([0])] * n
    
    
    for i in nb.prange(n):
        print(i)

        loc = inds[S[i] >= s]
        if len(loc) > 0:            
            basis = single_basis(S[loc][:,loc], s, np.where(loc == i)[0][0])
            if len(basis) > 0:
                out[i] = loc[basis]
            else:
                out[i] = np.empty(0,dtype=nb.int64)
        else:
            out[i] = np.empty(0,dtype=nb.int64)


    return out

def unique_bases(bases, min_size = 1):
    u_bases = Counter([frozenset(basis) for basis in bases if (len(basis) > 0) and (len(basis) >= min_size)])
    unique = np.empty(len(u_bases), object)
    unique[:] = [np.array(list(basis)) for basis in list(u_bases.keys())]
    return unique

def merging(bases, pairwise, q):
    n = pairwise.shape[0]
    tril = np.tril(np.full((n,n),True),-1) # Lower triangle of matrix
    mask = pairwise >= q
    tril_and_mask = tril & mask
    
    graph = csr_matrix(tril_and_mask)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
        
    uniq_labels = np.unique(labels)
    n_uniq_labels = len(uniq_labels)
    bound_bases = np.empty(n_uniq_labels, object)
    bound_bases[:] = [np.array(list(set().union(*bases[labels == i])), dtype = int) for i in uniq_labels]    
    return labels, bound_bases

def flexi_clust(S, s, q, min_size = 1, gpu_device_id = -1):
    bases = all_bases(S, s)   
    unique = unique_bases(bases, min_size)
    n_unique = len(unique)
    if n_unique > 0:        
        start_inds, packed = pack(unique)
        pairwise = jaccard_pairwise_similarities(start_inds, packed, gpu_device_id)
        structure, clusters = merging(unique, pairwise, q)
    else:
        clusters = np.array([], dtype = int)
        pairwise = np.array([])
        structure = np.array([], dtype = int)
    return unique, structure, clusters, pairwise

def single_basis3(sim, s, start):
    n = sim.shape[0]    
    out = np.array([start])
    C = np.delete(np.arange(n),start)
    while len(C) > 0:
        C = C[np.sum(sim[out][:,C] >= s, axis=0) == len(out)]
        if len(C) > 0:
            best = np.argsort(np.sum(sim[out][:,C], axis=0))[::-1][0]
            out = np.append(out,C[best])
            C = np.delete(C,best)
    return out


@njit(parallel=True)
def generate_bunches(similarities, s, mode = 0):
    n = similarities.shape[0]
    inds = np.arange(n)
    out = [np.empty(0,dtype=np.int64)] * n 
    for i in nb.prange(n):
        if mode == 0:
            loc = inds[similarities[i] >= s]
            if loc.shape[0] > 0:
                start = np.where(loc == i)[0][0]
                sim = similarities[loc][:,loc]
                bunch = np.array([start])
                C = np.delete(np.arange(loc.shape[0]),start)
                while C.shape[0] > 0:
                    C = C[np.sum(sim[bunch][:,C] >= s, axis=0) == bunch.shape[0]]
                    if C.shape[0] > 0:
                        best = np.argsort(np.sum(sim[bunch][:,C], axis=0))[::-1][0]
                        bunch = np.append(bunch,C[best])
                        C = np.delete(C,best)            
                if len(bunch) > 0:
                    out[i] = loc[bunch]
        else:
            out[i] = inds[similarities[i] >= s]
    return out

@njit(parallel=False)
def generate_bunch1(similarities, start, n_neighbors = 1000, n_objects = 10, kmax = 3, max_iters = 1000):
    n = similarities.shape[0]
    bunch = np.empty(0,dtype=np.int64)
    neighborhood = np.argsort(similarities[start])[::-1][:n_neighbors]    
    if n_neighbors >= n_objects:
        bunch = neighborhood[:n_objects]
        objective = np.sum(similarities[bunch][:,bunch])        
        neighborhood_set = set(neighborhood)
        residual = np.array(list(neighborhood_set - set(bunch)))
        k = 1
        print(objective)
        for niter in range(max_iters):
            candidates = np.random.permutation(residual)[:k]
            new_places = np.random.choice(n_objects, k, replace=False)
            new_bunch = np.copy(bunch)
            new_bunch[new_places] = candidates
            new_objective = np.sum(similarities[new_bunch][:,new_bunch])
            if new_objective > objective:
                bunch = new_bunch
                objective = new_objective
                print(niter,k,objective)
                residual = np.array(list(neighborhood_set - set(bunch)))
            k += 1
            if k > kmax: k = 1
    else:
        bunch = neighborhood
    return bunch

@njit(parallel=False)
def generate_bunch2(similarities, start, n_objects = 10, max_iters = 1000):
    n = similarities.shape[0]
    bunch = np.argsort(similarities[start])[::-1][:n_objects]
    objective = np.sum(similarities[bunch][:,bunch])
    print(objective)
    for niter in range(max_iters):
        new_bunch = np.copy(bunch)
        new_bunch[randint(0,n_objects-1)] = randint(0,n-1)
        new_objective = np.sum(similarities[new_bunch][:,new_bunch])
        if new_objective > objective:
            bunch = new_bunch
            objective = new_objective
            print(niter, objective)
    return bunch

@njit(parallel=False)
def generate_bunch3(similarities, start, n_objects = 10):
    n = similarities.shape[0]
    bunch = np.array([start])
    for i in range(n_objects-1):
        candidates = np.argsort(np.sum(similarities[bunch,:], axis=0))[::-1]
        for cand in candidates:
            if not cand in bunch:
                bunch = np.append(bunch,cand)
                break
    for i in range(n_objects):
        sub = np.delete(bunch,i)
        candidates = np.argsort(np.sum(similarities[sub,:], axis=0))[::-1]
        for cand in candidates:
            if not cand in sub:
                bunch[i] = cand
                break
    return bunch

@njit(parallel=False)
def generate_bunch4(similarities, start, n_objects = 10):
    n = similarities.shape[0]
    bunch = np.argsort(similarities[start])[::-1][:n_objects]
    return bunch



@njit(parallel=True)
def generate_bunches(similarities, n_objects = 100):
    n = similarities.shape[0]
    out = np.empty((n, n_objects), dtype=np.int64)    
    for i in nb.prange(n):
        out[i,:] = np.argsort(similarities[i])[::-1][:n_objects]
    return out


@njit(parallel=True)
def graph_components(X, threshold = 0.8):
    n = X.shape[0]
    ref = np.arange(n)
    for i in range(n):
        mask = np.full(n, False)
        for j in nb.prange(n):
            mask[j] = X[i,j] >= threshold
        inds = ref[mask]
        minv = np.min(inds)
        for j in nb.prange(n):
            for k in range(inds.shape[0]):
                if inds[k] == ref[j]:
                    ref[j] = minv
                    break
    uniq = np.unique(ref)
    out = np.empty_like(ref)
    for i in range(uniq.shape[0]):
        out[ref == uniq[i]] = i
    return out