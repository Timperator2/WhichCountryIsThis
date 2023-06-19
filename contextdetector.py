#parts of this code and the pretrained models from Ahsen Khaliq (https://huggingface.co/akhaliq)

from huggingface_hub import hf_hub_download
import clip
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
from typing import Tuple, List, Union, Optional
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm import trange
import io
import PIL.Image
import datamanager

conceptual_weight = None
coco_weight = None

T = None
D = None
CPU = None
CUDA = None
is_gpu = None
clip_model = None
preprocess = None
tokenizer = None
device = None
prefix_length = None

#load the captioning models
def loadModels(conceptual=True,coco=True):

    global coco_weight
    global conceptual_weight

    coco_path,conceptual_path = loadModelsPath(conceptual,coco) #get the path

    coco_weight = loadModelTorch(coco_path) #load weights
    conceptual_weight = loadModelTorch(conceptual_path)

#get model path and download it if necessary
def loadModelsPath(conceptual=True,coco=True):

    coco_path = None
    conceptual_path = None

    if conceptual:
        print("loading model conceptual")
        conceptual_path = hf_hub_download(repo_id="akhaliq/CLIP-prefix-captioning-conceptual-weights",
                                                filename="conceptual_weights.pt") #load from file or download
    if coco:
        print("loading model coco")
        coco_path = hf_hub_download(repo_id="akhaliq/CLIP-prefix-captioning-COCO-weights",
                                          filename="coco_weights.pt")

    return coco_path,conceptual_path


#load weights
def loadModelTorch(model_path):

    if model_path:

        model = ClipCaptionModel(prefix_length)

        model.load_state_dict(torch.load(model_path, map_location=CPU))
        model = model.eval()
        model = model.to(device)

        return model



#init pytorch
def initTorch():

    print ("init Torch")

    global T,D,CPU,CUDA,is_gpu,clip_model,preprocess,tokenizer,device,prefix_length

    T = torch.Tensor
    D = torch.device
    CPU = torch.device('cpu')
    CUDA = get_device
    is_gpu = True
    device = CUDA(0) if is_gpu else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    device = CUDA(0) if is_gpu else "cpu"
    prefix_length = 10

#get graphics card when available
def get_device(device_id: int) -> D:
    if not torch.cuda.is_available():
        return CPU
    device_id = min(torch.cuda.device_count() - 1, device_id)
    return torch.device(f'cuda:{device_id}')

#class for multilayer perceptron
class MLP(nn.Module):

    def forward(self, x: T) -> T:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


#model class
class ClipCaptionModel(nn.Module):

    # @functools.lru_cache #FIXME
    def get_dummy_token(self, batch_size: int, device: D) -> T:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: T, prefix: T, mask: Optional[T] = None, labels: Optional[T] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        # print(embedding_text.size()) #torch.Size([5, 67, 768])
        # print(prefix_projections.size()) #torch.Size([5, 1, 768])
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, prefix_size: int = 512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if prefix_length > 10:  # not enough memory
            self.clip_project = nn.Linear(prefix_size, self.gpt_embedding_size * prefix_length)
        else:
            self.clip_project = MLP(
                (prefix_size, (self.gpt_embedding_size * prefix_length) // 2, self.gpt_embedding_size * prefix_length))


#prefix class
class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


#beam search
def generate_beam(model, tokenizer, beam_size: int = 5, prompt=None, embed=None,
                  entry_length=67, temperature=1., stop_token: str = '.'):
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(beam_size, -1)
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(generated.shape[0], 1, -1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [tokenizer.decode(output[:int(length)]) for output, length in zip(output_list, seq_lengths)]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]
    return output_texts

#parameters for generation
def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.,
        stop_token: str = '.',
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in trange(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]

#generate caption
def inference(image, model):

    use_beam_search = False

    pil_image = PIL.Image.fromarray(image)
    image = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        prefix = clip_model.encode_image(image).to(device, dtype=torch.float32)
        prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)
    if use_beam_search:
        generated_text_prefix = generate_beam(model, tokenizer, embed=prefix_embed)[0]
    else:
        generated_text_prefix = generate2(model, tokenizer, embed=prefix_embed)

    return generated_text_prefix

#generate caption with selected model
def getCaption(image,model="conceptual"):

    img = np.asarray(PIL.Image.open(io.BytesIO(image["file"])))

    if model == "coco":
        model = coco_weight
    elif model == "conceptual":
        model = conceptual_weight
    else:
        model = 0

    return inference(img,model) #generate and return caption

#check if word is actually describing the image content
def isWordRelevant(word):
    filterList = ("a","of","it","with","the","and","to","that","are","is","by","picture","photo")
    if word in filterList:
        return False
    else:
        return True

#get a list with all occuring words and the percentage of associated image captions
def getAverageWordList(images,model="conceptual",filter=True):

    all_words = {}

    for image in images:
        words = getWordList(image,model,filter) #generate a word list with absolute occurences
        for word in words:
            if word not in all_words: #if word not in list of all occuring words
                all_words[word] = 1
            else:
                all_words[word] = all_words[word] + 1

    for word in all_words:
        all_words[word] = all_words[word] / len(images) #calculate percentage

    return all_words


#get all relevant words for a single image caption
def getWordList(image,model="conceptual",filter=True):

    filtered_words = []

    caption = getCaption(image,model) #get the caption
    words = caption.split(" ") #split into single words
    for word in words:
        word = word.lower() #make all lower case
        word = word.replace(".","") #remove all dots (sentence endings)

        if (filter and isWordRelevant(word)) or not filter: #filter irrelevant words
            filtered_words.append(word) #append word to word list

    return filtered_words


#get the relative occurences of all words in a word list compared to a larger set of occuring words
def getRelativeWordList(wordlist,total_wordcounts):

    if wordlist:
        for word in wordlist:
            wordlist[word] = wordlist[word] / total_wordcounts[word] #calculate relative occurence

    return wordlist

#get all words with number of occurence over several wordlists
def getTotalWordCounts(wordlists):

    all_words = {}

    for wordlist in wordlists:
        if wordlist:
            for word in wordlist:
                if word not in all_words: #if word not yet in list of all occuring words
                    all_words[word] = wordlist[word]
                else:
                    all_words[word] = all_words[word] + wordlist[word]

    return all_words

#compare a panorama word list and a country word list for similarity
def compareWordLists(wordlist,test_wordlist):
    score = 0
    for word in wordlist:
        if word in test_wordlist: #if word is in both word lists
            score = score + test_wordlist[word] #the number of occurences in the second word list is added to the score

    return score / len(wordlist) #normalize score (as number of occurences is a percentage value)


#get list with average and relative word lists for all countries
def getCompleteList(number=100,fov=90,model="conceptual"):

    countries = datamanager.getAllCountriesWithExternalData()

    all_wordlists = []

    for country in countries: #load average word list for all countries
        all_wordlists.append([country,datamanager.loadAverageWordList(country,number,fov,model)])

    only_wordlists = [item[1] for item in all_wordlists] #only wordlists without country information

    total_list= getTotalWordCounts(only_wordlists) #get all occuring words

    for country in all_wordlists:
        if country[1]: #if country has a word list
            country.append(getRelativeWordList(country[1].copy(), total_list))  #get relative word list
        else:
            country.append(False)

    return all_wordlists



def analyseWordLists(images,number=100,fov=90,model="conceptual",filter=True,mode=1):

    print ("analysing captions")

    complete_list = getCompleteList(number,fov,model) #get relative word lists for all countries

    similarity_total = []

    for image in images:

        wordlist = getWordList(image,model,filter) #get word list for image

        print ("words for image: ", wordlist)

        similarity = []

        for country in complete_list:

            average_wordlist = country[1] #get the average word list of a country

            if average_wordlist:

                if mode == 0: #compare with average word lists (emphasis on generally frequent words)
                    similarity.append([country[0], compareWordLists(wordlist,average_wordlist)])
                elif mode == 1: #compare with relative word lists (emphasis on rare words)
                    relative_wordlist = country[2]
                    similarity.append([country[0], compareWordLists(wordlist, relative_wordlist)])

        for index,country in enumerate(similarity):
            if len (similarity_total) < len(similarity): #for first image
                similarity_total.append([country[0],country[1]]) #append country and similarity score
            else: #for further images
                similarity_total[index][1] = similarity_total[index][1] + country[1] #add score

    for index,value in enumerate(similarity_total):
        similarity_total[index][1] = (value[1] / len(images)) #normalize

    similarity_total.sort(key=lambda x: x[1],reverse=True)


    return similarity_total


initTorch() #init pytorch
loadModels(True,False) #load models








