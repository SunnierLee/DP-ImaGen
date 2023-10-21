from gensim.models import KeyedVectors
import numpy as np
import torch

from category_name import *

def cal_semantic_distribution_distance(model, d1, d2):
    sd = []
    for cls in d1:
        similarity = []
        s1_list = d1[cls]
        s2_list = d2[cls]
        for s1 in s1_list:
            for s2 in s2_list:
                if s1 is None or s2 is None:
                    continue
                simi = model.similarity(s1, s2)
                similarity.append(simi)
        print(similarity)
        if len(similarity) != 0:
            sd.append(np.mean(similarity))
    return np.mean(sd)

def cal_semantic_distribution_distance2(model, d1, d2):
    sd = []
    for cls in d1:
        similarity = []
        s1_list = d1[cls]
        s2_list = d2[cls]
        for s1 in s1_list:
            for s2 in s2_list:
                if s1 is None or s2 is None:
                    continue
                simi = model.similarity(s1, s2)
                similarity.append(simi)
        print(similarity)
        if len(similarity) != 0:
            sd.append(np.mean(similarity))
    return np.mean(sd)




def refine_imagenet(model):
    for cls in imagenet1k:
        semantic = imagenet1k[cls]
        semantic = semantic.replace("_", " ").replace(",", " ").replace("-", " ")
        semantics = semantic.split(" ")
        semantics.reverse()
        for s in semantics:
            try:
                _ = model[s]
                break
            except:
                s = None
                continue
        print(s)
        imagenet1k[cls] = s


def refine_imagenet2():
    new_specific = {}
    new_semantics = list(specific_class.values())
    for cls in target:
        new_specific[cls] = new_semantics
    return new_specific

def refine_celeba():
    for cls in celeba:
        for i in range(len(celeba[cls])):
            s = celeba[cls][i].split('_')[-1].lower()
            celeba[cls][i] = s


def refine_cifar10():
    for cls in cifar10:
        cifar10[cls] = [cifar10[cls]]

word2vec_output_file = 'glove.twitter.27B.25d.word2vec.txt'
glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

refine_imagenet(glove_model)
refine_cifar10()
refine_celeba()

specific_class_ = "src/QueryResults/chosen_top50_classes_for_cifar10_epsilon0.01.pth"
# specific_class_ = None
if specific_class_ is not None:
    specific_class = torch.load(specific_class_)
    for cls in specific_class:
        for i in range(len(specific_class[cls])):
            specific_class[cls][i] = imagenet1k[specific_class[cls][i]]
else:
    specific_class = imagenet1k
print(specific_class)
# replace index with semantic

target = cifar10

if specific_class_ is None:
    specific_class = refine_imagenet2()

dist = cal_semantic_distribution_distance(glove_model, specific_class, target)
print(dist)

