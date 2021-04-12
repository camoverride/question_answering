from answer_question import Answerer


qa = Answerer()

ans = qa.answer_question("What is the biggest city in the USA?")

print(ans["answer"]["answer"])
print("########################################################")
print(ans)
