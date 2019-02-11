import stanfordnlp
import time

root='/root'
lang='en'
text = "You, you love it how I move you\nYou love it how I touch you\nMy one, when all is said and done\nYou'll believe God is a woman\n\nAnd I, I feel it after midnight\nA feeling that you can't fight\nMy one, it lingers when we're done\nYou'll believe God is a woman\n\nI don't wanna waste no time, yuh\nYou ain't got a one-track mind, yuh\nHave it any way you like, yuh\nAnd I can tell that you know\nI know how I want it\n\nAin't nobody else can relate\nBoy I like that you ain't afraid\nBaby lay me down and let's pray\nI'm telling you the way I like it\nHow I want it\n\nYuh\nAnd I can be all of things you tell me not to be, yuh\nWhen you try to come for me I keep on flourishing, yuh\nAnd he see the universe when I'm in company, uh\nIt's all in me\n\nYou, you love it how I move you\nYou love it how I touch you\nMy one, when all is said and done\nYou'll believe God is a woman\n\nAnd I, I feel it after midnight\nA feeling that you can't fight\nMy one, it lingers when we're done\nYou'll believe God is a woman\n\nI tell you all the things you should know\nSo baby take my hands, save your soul\nWe can make it last, take it slow\nAnd I can tell that you know\nI know how I want it\n\nBut you different from the rest\nAnd boy if you confess you might get blessed\nSee if you deserve what comes next\nI'm telling you the way I like it\nHow I want it\n\nYuh\nAnd I can be all of things you tell me not to be, yuh\nWhen you try to come for me I keep on flourishing, yuh\nAnd he see the universe when I'm in company\nIt's all in me\n\nYou, you love it how I move you\nYou love it how I touch you\nMy one, when all is said and done\nYou'll believe God is a woman\n\nAnd I, I feel it after midnight\nA feeling that you can't fight\nMy one, it lingers when we're done\nYou'll believe God is a woman, yeah yeah\n\nGod is a woman, yeah yeah\nGod is a woman\nMy one (one)\nWhen all is said and done\nYou'll believe God is a woman\n\n(You'll believe God)\nGod is a woman (oh, yeah)\nGod is a woman, yeah\n(One) It lingers when we're done\nYou'll believe God is a woman";
processors = "tokenize,mwt,pos,lemma" #tokenize,mwt,pos,lemma,depparse

stanfordnlp.download(lang, resource_dir=root, should_download=True, confirm_if_exists=False)

start_time = time.time()
pipeline = stanfordnlp.Pipeline(lang=lang, models_dir=root, use_gpu=False, processors=processors) # This sets up a default neural pipeline in English
elapsed_time = time.time() - start_time
print("loaded in:%f" % elapsed_time)

start_time = time.time()
doc = pipeline(text)
elapsed_time = time.time() - start_time
print("parsed in:%f" % elapsed_time)

for sentence in doc.sentences:
    sentence.print_dependencies()
    #sentence.print_tokens()
    for token in sentence.tokens:
        word=token.words[0]
        print("Index:%s word:%s lemma:%s" %(token.index,word.text,word.lemma))