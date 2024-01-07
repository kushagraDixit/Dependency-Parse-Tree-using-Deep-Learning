import sys
sys.path.append('scripts')
import argparse
import torch
import numpy as np
from scripts import state
from scripts import evaluate
import torchtext as text
from torchtext.data import Field
import torch.nn.functional as F
from tqdm import tqdm 
import time
from sklearn.metrics import f1_score

class ActionPredictor(torch.nn.Module):
    def __init__(self,embedding_dim,pos_dim,hidden_dim,out_dim,pos_dict_len):
        super().__init__()
        self.pos_embeddings = torch.nn.Embedding(pos_dict_len, pos_dim)
        self.W_word = torch.nn.Linear(embedding_dim, hidden_dim, bias=True)
        self.W_pos = torch.nn.Linear(pos_dim, hidden_dim, bias=True)
        self.acti_fn = torch.nn.ReLU()
        self.W_out = torch.nn.Linear(hidden_dim, out_dim, bias=True)

    def forward(self, word_vec,pos_vec):
        pos_vec = self.pos_embeddings(pos_vec).mean(1).squeeze(1)
        h = self.W_word(word_vec) + self.W_pos(pos_vec)
        y = self.acti_fn(h)
        out = self.W_out(y)

        return out
    

class ActionPredictorWithDependencyLabel(torch.nn.Module):
    def __init__(self,embedding_dim,pos_dim,lab_dim,hidden_dim,out_dim,pos_dict_len,lab_dict_len):
        super().__init__()
        print(embedding_dim)
        self.pos_embeddings = torch.nn.Embedding(pos_dict_len, pos_dim)
        self.lab_embeddings = torch.nn.Embedding(lab_dict_len, lab_dim)
        self.W_word = torch.nn.Linear(embedding_dim, hidden_dim, bias=True)
        self.W_pos = torch.nn.Linear(pos_dim, hidden_dim, bias=True)
        self.W_lab = torch.nn.Linear(lab_dim, hidden_dim, bias=True)
        self.acti_fn = torch.nn.ReLU()
        self.W_out = torch.nn.Linear(hidden_dim, out_dim, bias=True)

    def forward(self, word_vec,pos_vec,lab_vec):
        pos_vec = self.pos_embeddings(pos_vec).mean(1).squeeze(1)
        lab_vec = self.lab_embeddings(lab_vec).mean(1).squeeze(1)
        h = self.W_word(word_vec) + self.W_pos(pos_vec) + self.W_lab(lab_vec)
        y = self.acti_fn(h)
        out = self.W_out(y)

        return out

class ParseTree:
    def __init__(self, buffer, cwindow) -> None:
        initial_stack = []
        buffer_end = []

        for i in range(cwindow):
            pad_token = state.Token(idx=-1,word='[PAD]',pos='NULL')
            buffer_token = state.Token(idx=-1,word='[PAD]',pos='NULL')
            initial_stack.append(pad_token)
            buffer_end.append(buffer_token)
            
        
        new_buffer = buffer + buffer_end
        dependency_list = []

        self.praseState = state.ParseState(stack=initial_stack,parse_buffer=new_buffer, dependencies=dependency_list)
        pass

    def get_next_state(self,action):
        action_list = action.split('_')

        if(action_list[0]=='SHIFT'):
            state.shift(self.praseState)
        elif(action_list[0]=='REDUCE'):
            if(action_list[1]=='L'):
                state.left_arc(self.praseState, action_list[2])
            elif(action_list[1]=='R'):
                state.right_arc(self.praseState,action_list[2])
    
    def is_final_state(self,cwindow):
        return state.is_final_state(self.praseState,cwindow)
    
    def is_state_valid_for_reduce(self,cwindow):
        return state.is_state_valid_for_reduce(self.praseState,cwindow)
    
    def is_state_valid_for_shift(self,cwindow):
        return state.is_state_valid_for_shift(self.praseState,cwindow)
    
    def get_current_state(self):
        return self.praseState


class CreateDataset:
    def __init__(self, train_path, dev_path, test_path, pos_tag_path, action_tag_path,embedding_name,embedding_dim, context_window, isConcat,lab_emb_incl,device):
        
        self.cwindow = context_window
        self.action_dict = self.makeMapping(action_tag_path)
        self.action_rev_dict = self.getReverseDict(self.action_dict)
        self.pos_dict = self.makeMapping(pos_tag_path)
        self.lab_dict = self.createLabelDictionary(self.action_dict)
        self.word_emb = self.getWordEmbeddings(embedding_name,embedding_dim)
        self.concat = isConcat
        self.lab_emb_incl = lab_emb_incl
        self.train_sentences,self.train_pos_tags,self.train_gold_actions = self.processRawFiles(train_path)
        self.dev_sentences,self.dev_pos_tags,self.dev_gold_actions = self.processRawFiles(dev_path)
        self.test_sentences,self.test_pos_tags,self.test_gold_actions = self.processRawFiles(test_path)
        self.train_input_words,self.train_input_pos,self.train_input_lab,self.train_output = self.createFinalData(self.train_sentences,self.train_pos_tags,self.train_gold_actions,device)
        
        

    def createFinalData(self,sentences,speech,actions,device):
            #create tokenlist through those lists
            token_list = self.createTokenList(sentences,speech)

            #from token list for each sentence intiailise a state and take action and save states
            word_vectors = []
            pos_vectors = []
            lab_vectors = []
            output_label = []

            for i,tokens in enumerate(token_list):
                parseTree = ParseTree(tokens,self.cwindow)
                for action in actions[i]:
                    self.saveStateAsInputOutput(parseTree.get_current_state(),word_vectors,pos_vectors,lab_vectors,self.cwindow,self.word_emb,self.pos_dict,self.lab_dict,self.concat,sentences[i],speech[i],self.lab_emb_incl)
                    output_label.append(self.action_dict[action])
                    #output_label.append(self.action_dict.get(action))
                    parseTree.get_next_state(action)

            word_vectors = np.array(word_vectors)
            word_vectors = torch.tensor(word_vectors, dtype=torch.float).to(device)
            pos_vectors = np.array(pos_vectors)
            pos_vectors = torch.tensor(pos_vectors, dtype=torch.long).to(device)
            lab_vectors = np.array(lab_vectors)
            lab_vectors = torch.tensor(lab_vectors, dtype=torch.long).to(device)
            output_label = np.array(output_label)
            output_label = torch.tensor(output_label, dtype=torch.long).to(device)
            return word_vectors,pos_vectors,lab_vectors,output_label

    def getTrainingDataForModel(self, batch_size):
        dataset_train = torch.utils.data.TensorDataset(self.train_input_words,self.train_input_pos,self.train_input_lab, self.train_output)
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size, shuffle=True)
        return train_loader

    def processRawFiles(self,file_path):
        file = open(file_path, 'r')
        Lines = file.readlines()
        sentences, speech, actions = [], [], []
 
        for line in Lines:
            arr = line.split('|||')
            sen, pos, action = self.getAllArrays(arr)
            sentences.append(sen)
            speech.append(pos)
            actions.append(action)

        return sentences,speech,actions
    
    def createTokenList(self,sentences,speech):
        token_list = []
        for i,sentence in enumerate(sentences):
            my_list = []
            for j,word in enumerate(sentence):        
                new_token = state.Token(idx=j,word=word,pos=speech[i][j])
                my_list.append(new_token)
            token_list.append(my_list)
        
        return token_list
    
    def saveStateAsInputOutput(self,parseState,word_vectors,pos_vectors,lab_vectors,cwindow,vec,pos_dict,lab_dict,concat,sentence,speech,lab_emb_incl):
        #print("State : ")
        #showCurrentState(parseState)
        
        token_vector = parseState.stack[-cwindow:] + parseState.parse_buffer[:cwindow]
        words_vector = []
        pos_vector = []
        lab_vector = []

        for token in token_vector:
            words_vector.append(token.word)
            pos_vector.append(pos_dict[token.pos])

        if(lab_emb_incl):
            for token in parseState.stack[-cwindow:]:
                leftmost_word,rightmost_word,leftmost_pos,rightmost_pos,left_lab,right_lab=self.getChilds(parseState,token.word,sentence,speech)
                words_vector.extend([leftmost_word,rightmost_word])
                pos_vector.extend([pos_dict[leftmost_pos],pos_dict[rightmost_pos]])
                lab_vector.extend([lab_dict[left_lab],lab_dict[right_lab]])

        word_list = vec.get_vecs_by_tokens(words_vector, lower_case_backup=True)

        if(concat):
            word_list = np.asarray(word_list.flatten())
        else:
            word_list = np.array(word_list.mean(dim=0))
        word_vectors.append(word_list)
        pos_vectors.append(pos_vector)
        lab_vectors.append(lab_vector)
    
    def getChilds(self,parseState,word,sentence,speech):
        leftmost_word = '[PAD]'
        leftmost_pos = 'NULL'
        rightmost_word = '[PAD]'
        rightmost_pos = 'NULL'
        left_lab = 'NULL'
        right_lab = 'NULL'
        if(word=='[PAD]'):
            #print("Returning for PAD !!!!!")
            return leftmost_word,rightmost_word,leftmost_pos,rightmost_pos,left_lab,right_lab
        
        leftmost_child = sentence.index(word)
        rightmost_child = sentence.index(word)
        #print(f"index word : {rightmost_child}")

        

        for dependency in parseState.dependencies:
            if dependency.source.word == word:
                index_target = sentence.index(dependency.target.word)
                if index_target>rightmost_child:
                    rightmost_child = index_target
                    rightmost_word = dependency.target.word
                    rightmost_pos = speech[index_target]
                    right_lab = dependency.label

                if(index_target<leftmost_child):
                    leftmost_child = index_target
                    leftmost_word = dependency.target.word
                    leftmost_pos = speech[index_target]
                    left_lab = dependency.label

        #print(f"left_most index : {leftmost_child}")
        #print(f"right_most index : {rightmost_child}")

        return leftmost_word,rightmost_word,leftmost_pos,rightmost_pos,left_lab,right_lab
    
    def getAllArrays(self,ar):
        sen = ar[0].split()
        pos = ar[1].split()
        action = ar[2].split()

        return sen,pos,action
    
    def createLabelDictionary(self,action_dict):
        lab_dict = {}
        idx = 0
        for key in action_dict:
            breakKey = key.split('_')
            if(breakKey[0]=='REDUCE'):
                if breakKey[2] not in lab_dict:
                    lab_dict[breakKey[2]] = idx
                    idx = idx+1
        lab_dict['NULL'] = idx
        return lab_dict
    
    def makeMapping(self,file):
        my_dict = {}
        with open(file) as f:
            data = f.readlines()
            i = 0
            for line in data:
                my_dict[line.split()[0].strip()] = i
                i+=1
        return my_dict
    
    def getWordEmbeddings(self,name,emb_dim):
        print(f"loading for dim : {emb_dim} and name : {name}")
        return text.vocab.GloVe(name=name, dim=emb_dim)
    
    def getPosEmbeddings(self,pos_dict):
        return torch.nn.Embedding(len(pos_dict),50)
    
    def getReverseDict(self,my_map):
        return dict((v, k) for k, v in my_map.items())  

 
def findPredictedActions(model,dataset,sentences,pos_tags,cwindow,lab_emb_incl,device):
    pred_actions = []

    for i,sentence in enumerate(sentences):
        pred_action = evaluate_this_sentence(model,dataset,sentence,pos_tags[i],cwindow,lab_emb_incl,device)
        pred_actions.append(pred_action)

    return pred_actions

def trainMyModel(model,optimizer,loss_fn,train_loader,max_epoch,dataset,embedding_name,embedding_dim,lab_emb_incl,device):
    best_perf_dict = {"metric": 0, "epoch": 0, "dev_loss": sys.maxsize}

    best_perf_dict = {"UAS": 0, "LAS": 0, "epoch": 0}
    # Begin the training loop
    for ep in range(1, max_epoch+1):
        print(f"Epoch {ep}")
        train_loss = []  
        for inp_words,inp_pos, inp_lab,res in tqdm(train_loader):
            model.train()
            optimizer.zero_grad()
            if(lab_emb_incl):
                out = model(inp_words.to(device),inp_pos.to(device),inp_lab.to(device))
            else:
                out = model(inp_words.to(device),inp_pos.to(device)) 
            loss = loss_fn(out, res.to(device))
            loss.backward() 
            optimizer.step()  
            train_loss.append(loss.cpu().item())

        print(f"Average training batch loss: {np.mean(train_loss)}")

        #Evaluation of Model
        pred_actions_dev = findPredictedActions(model,dataset,dataset.dev_sentences,dataset.dev_pos_tags,dataset.cwindow,lab_emb_incl,device)
        uas,las = evaluate.compute_metrics(dataset.dev_sentences,dataset.dev_gold_actions,pred_actions_dev,dataset.cwindow)
        
        print(f"UAS : {uas}  LAS : {las}") 

        if las > best_perf_dict["LAS"]:
            best_perf_dict["LAS"] = las
            best_perf_dict["UAS"] = uas
            best_perf_dict["epoch"]  = ep
            torch.save({
                    "model_param": model.state_dict(),
                    "optim_param": optimizer.state_dict(),
                    "dev_las": las,
                    "dev_uas": uas,
                    "epoch": ep
                }, f"./Models/models_{embedding_name}_ed{embedding_dim}_concat{dataset.concat}_ep{ep}")
            best_path = f"./Models/models_{embedding_name}_ed{embedding_dim}_concat{dataset.concat}_ep{ep}"
    

    print(f"""\nBest Dev performance of UAS : {best_perf_dict["UAS"]} and LAS : {best_perf_dict["LAS"]} at epoch {best_perf_dict["epoch"]}""")
    return best_path

def show(token):
    print(f"word index : {token.idx}  word : {token.word}  pos : {token.pos}")

def show_dep(dependencyEdge):
    print(f"Source : {dependencyEdge.source}  Target : {dependencyEdge.target}  Label : {dependencyEdge.label}")

def showCurrentState(parseState):
    print("Stack : ")
    print(len(parseState.stack))
    for elem in parseState.stack:
        show(elem)

    print("Buffer : ")
    print(len(parseState.parse_buffer))
    for elem in parseState.parse_buffer:
        show(elem)

    print("Dependency : ")
    print(len(parseState.dependencies))
    for elem in parseState.dependencies:
        show_dep(elem)

def evaluate_this_sentence(model,dataset,sentence,pos_tags,cwindow,lab_emb_incl,device):
    #convert sentence and pos to arrays and eventually embeddddings
    token_sentence = []

    for j,word in enumerate(sentence):        
        new_token = state.Token(idx=j,word=word,pos=pos_tags[j])
        token_sentence.append(new_token)
    
    #make the initial state and for each state L 
    parseTree = ParseTree(token_sentence,cwindow)
    #showCurrentState(parseTree.get_current_state())
    actions = []
    
    while parseTree.is_final_state(cwindow=cwindow)==False:     
        word_vectors = []
        pos_vectors = []
        lab_vectors = []
        dataset.saveStateAsInputOutput(parseTree.get_current_state(),word_vectors,pos_vectors,lab_vectors,dataset.cwindow,dataset.word_emb,dataset.pos_dict,dataset.lab_dict,dataset.concat,sentence,pos_tags,dataset.lab_emb_incl)
        #showCurrentState(parseState=parseTree.get_current_state())
        word_vectors = np.array(word_vectors)
        word_vectors = torch.tensor(word_vectors, dtype=torch.float).to(device)
        pos_vectors = np.array(pos_vectors)
        pos_vectors = torch.tensor(pos_vectors, dtype=torch.long).to(device)
        lab_vectors = np.array(lab_vectors)
        lab_vectors = torch.tensor(lab_vectors, dtype=torch.long).to(device)

        if(lab_emb_incl):
            out = model(word_vectors.to(device),pos_vectors.to(device),lab_vectors.to(device))
        else:
            out = model(word_vectors.to(device),pos_vectors.to(device))
          
        top_two = out.topk(2)
        action1 = top_two[1].cpu().detach().numpy()[0][0]
        action2 = top_two[1].cpu().detach().numpy()[0][1]
        action1 = dataset.action_rev_dict[action1]
        action2 = dataset.action_rev_dict[action2]
        

        if action1=='SHIFT':
            if parseTree.is_state_valid_for_shift(cwindow):
                actions.append(action1)
                parseTree.get_next_state(action1)
            else:
                actions.append(action2)
                parseTree.get_next_state(action2)
        else:
            if parseTree.is_state_valid_for_reduce(cwindow):
                actions.append(action1)
                parseTree.get_next_state(action1)
            else:
                actions.append('SHIFT')
                parseTree.get_next_state('SHIFT')
    return actions

def computeResultsOnTestFile(model,best_path,dataset,lab_emb_incl,device):
    model.load_state_dict(torch.load(best_path), strict=False)

    model.eval()
    pred_actions_test = findPredictedActions(model,dataset,dataset.test_sentences,dataset.test_pos_tags,dataset.cwindow,lab_emb_incl,device)
    uas,las = evaluate.compute_metrics(dataset.test_sentences,dataset.test_gold_actions,pred_actions_test,dataset.cwindow)
    print(f"Test UAS : {uas} TEST LAS : {las}")

def writeResultsForHiddenFile(hidden_path,dataset):
    file = open(hidden_path, 'r')
    Lines = file.readlines()
    sentences, speech = [], []
 
    for line in Lines:
        arr = line.split('|||')
        sen = arr[0].split()
        pos = arr[1].split()
        sentences.append(sen)
        speech.append(pos)

    out_file = open("results.txt", "w")

    for i,sentence in enumerate(sentences):
        action = evaluate_this_sentence(model,dataset,sentence,speech[i],dataset.cwindow,lab_emb_incl,device)
        action.append('\n')
        out_file.writelines(' '.join(action))

    out_file.close()

def evaluateOnTestSentences(model,dataset,device):
    sentence = "Mary had a little lamb ."
    pos_tags = "PROPN AUX DET ADJ NOUN PUNCT"

    sentence = sentence.split()
    pos_tags = pos_tags.split()

    actions = evaluate_this_sentence(model,dataset,sentence,pos_tags,dataset.cwindow,dataset.lab_emb_incl,device)
    print(f"Sentence : {' '.join(sentence)}")
    print(f"Actions : {' '.join(actions)}")

    sentence = "I ate the fish raw ."
    pos_tags = "PRON VERB DET NOUN ADJ PUNCT"

    sentence = sentence.split()
    pos_tags = pos_tags.split()

    actions = evaluate_this_sentence(model,dataset,sentence,pos_tags,dataset.cwindow,dataset.lab_emb_incl,device)
    print(f"Sentence : {' '.join(sentence)}")
    print(f"Actions : {' '.join(actions)}")

    sentence = "With neural networks , I love solving problems ."
    pos_tags = "ADP ADJ NOUN PUNCT PRON VERB VERB NOUN PUNCT"

    sentence = sentence.split()
    pos_tags = pos_tags.split()

    actions = evaluate_this_sentence(model,dataset,sentence,pos_tags,dataset.cwindow,dataset.lab_emb_incl,device)
    print(f"Sentence : {' '.join(sentence)}")
    print(f"Actions : {' '.join(actions)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', default=0.001, type=float, help='The Learning Rate of the Model')
    parser.add_argument('--epochs', default=20, type=int, help='Total number of epochs for training')
    parser.add_argument('--batch_size', default=64, type=int, help='The batch size for training')
    parser.add_argument('--context_window', default=2, type=int, help='Context Window for Stack and Buffer')
    parser.add_argument('--embedding_dim', default=50, type=int, help='Word Embedding Dimensions')
    parser.add_argument('--embedding_name', default='6B', type=str, help='Dimension Name')
    parser.add_argument('--concatenation', default=1, type=int, help='0: Mean and 1:Concatenation')
    parser.add_argument('--lab_emb_incl', default=0, type=int, help='1: Include Labels for Leftmost and Right Most Child')
    parser.add_argument('--train_path', default='./data/train.txt', type=str, help='Directory where the training data is stored')
    parser.add_argument('--dev_path', default='./data/dev.txt', type=str, help='Directory where the dev data is stored')
    parser.add_argument('--test_path', default='./data/test.txt', type=str, help='Directory where the test data is stored')
    parser.add_argument('--hidden_path', default='./data/hidden.txt', type=str, help='Directory where the hidden data file is stored')
    parser.add_argument('--pos_tag_path', default='./data/pos_set.txt', type=str, help='Directory where the pos_tags are stored')
    parser.add_argument('--action_tag_path', default='./data/tagset.txt', type=str, help='Directory where the actions are stored')
    torch.manual_seed(64)

    args = parser.parse_args()
    learning_rate, max_epoch, BATCH_SIZE, context_window, embedding_dim, embedding_name, isConcatinate, train_path, dev_path, test_path, hidden_path, pos_tag_path, action_tag_path, lab_emb_incl = (
        args.learning_rate, args.epochs, args.batch_size, args.context_window, args.embedding_dim, args.embedding_name, args.concatenation, args.train_path,
        args.dev_path, args.test_path, args.hidden_path, args.pos_tag_path, args.action_tag_path, args.lab_emb_incl)

    if(isConcatinate==1):
        isConcatinate=True
    else:
        isConcatinate=False

    if(lab_emb_incl==1):
        lab_emb_incl = True
    else:
        lab_emb_incl = False
    
    word_vector_size = embedding_dim

    if(isConcatinate):
        print("Concatination will be Done !!!!")
        if(lab_emb_incl):
            word_vector_size = embedding_dim*context_window*4
        else:
            word_vector_size = embedding_dim*context_window*2
    else:
        print("Mean will be taken !!!")
    
    print(embedding_dim)
    print(f"EPOCHS : {max_epoch}")
    print(f"Concatination : {isConcatinate}")
    print(f"Embedding name : {embedding_name}")
    print(f"Embedding name : {embedding_dim}")
    pos_dim = 50
    hidden_dim = 200
    lab_dim = 50
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    dataset = CreateDataset(train_path,dev_path,test_path,pos_tag_path,action_tag_path,embedding_name,embedding_dim,context_window,isConcatinate,lab_emb_incl,device)
    out_dim = len(dataset.action_dict)
    train_loader = dataset.getTrainingDataForModel(BATCH_SIZE)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Create a randomly initialized model and the optimizer.
    if lab_emb_incl==1:
        model = ActionPredictorWithDependencyLabel(embedding_dim=word_vector_size,pos_dim=pos_dim,lab_dim=lab_dim,hidden_dim=hidden_dim,out_dim=out_dim,pos_dict_len=len(dataset.pos_dict),lab_dict_len=len(dataset.lab_dict)).to(device)
    else:
        model = ActionPredictor(embedding_dim=word_vector_size,pos_dim=pos_dim,hidden_dim=hidden_dim,out_dim=out_dim,pos_dict_len=len(dataset.pos_dict)).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_model = trainMyModel(model,optimizer,loss_fn,train_loader,max_epoch,dataset,embedding_name,embedding_dim,lab_emb_incl,device)
    computeResultsOnTestFile(model,best_model,dataset,lab_emb_incl,device)
    writeResultsForHiddenFile(hidden_path,dataset)
    evaluateOnTestSentences(model,dataset,device)
