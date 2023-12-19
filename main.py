from pre_processing import sbStemmer,pStemmer, lemmatizer
from intent_matching import intentMatching
from small_talk import stResponse
from question_answering import qaResponse


stoplist = ['bye', 'goodbye']
stop = False


print("Hi nice to meet you, I'm S.A.M")
while stop != True:
    query = input(":>>")
    if query not in stoplist:
        
        #Intent Matching here:
        intent = intentMatching(pStemmer(query))
        #intent = intentMatching(query)
        print(intent)
   

        if intent == "Small Talk":
            response = stResponse(query)
            print(response)
            continue
        
        elif intent == "Question":
            
            response = qaResponse(query)
            print(response)
            continue

        else:
            print("S.A.M: GRRRRRRRRR I can't work out what you mean :/")
            print("REBOOT... REBOOT...")


    else:
        print("Chatbot: ByeBye")
        stop = True




