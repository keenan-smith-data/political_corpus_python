import corpus_creation as create
import nltk_tokenize_corpus as tokenize
import models.lasso_logistic_regression_grid_search as lasso_search
import models.knn_grid_search as knn_search
import models.randomforest_grid_search as forest_search
import models.linearsvc_grid_search as svc_search
import time

def main():
    yes_answers = {"yes", "Yes", "YES", "y", "ye", "yeah", "yea", "affirmative"}
    no_answers = {"no", "No", "NO", "N", "n", "Nope", "Nah", "Negative"}

    print("\nWelcome to my ISE535 Project")
    print("\nFirst, Let me build the Corpus for you")
    
    cont = ""
    while cont not in yes_answers:
        cont = input("\n Would you like me to do that? (y/N) ")
        if cont in yes_answers:
            print("\nPreparing Corpus for you now!")
            create.main()
            print("\nThe corpus is now built and ready for Tokenizing")
            print("\nWould you like to Tokenize (y/N)")
            print("\nI have to let you know that this may take 10 mins or so!")
            tok = input("\nWould you like to Tokenize (y/N)")
            if tok in yes_answers:
                print("\nAlright, let me get that for you")
                print("\nI have to let you know that this may take 10 mins or so!")
                tokenize.main()
                print("\nAlright, I got that tokenized for you")
                print("\nOk, here comes the big tasks, you may want to watch a movie if you run these")
                print("\nI am going to have to run some large grid searches to give you some possible models to examine")
                print("\nSeriously though, this is going to take awhile and there is a chance it wont work")
                print("\nAlso, if you are on Windows, this may not work at all...")
                modeling = input("\nWould you like to continue? (y/N) ")
                if modeling in yes_answers:
                    lasso_search.main()
                    print("\n That is one out of the way")
                    knn_search.main()
                    print("\nThat's two!")
                    svc_search.main()
                    print("\nThree, ah, ah, ah")
                    print("\nYou probably won't even see this")
                    time.sleep(10)
                    print("\nHere is the really long one unfortunately")
                    forest_search.main()
                    print("\nPhew, we are done, that was the easy part")
                    print("\nNow you have to make sense of all this mess")
                else:
                    print("\n ok, maybe next time")

            else:
                print("\nTake it easy, see you soon")

        else:
            print("\nIt's ok, I wasn't that interesting anyway!")
            break


if __name__ == "__main__":
    main()    


