# questionnaire_config.py

QUESTIONNAIRE_CONFIG = {
    "neuroticism": {
        "standardize": (
            "Please rate the accuracy of each statement according to your description using a score between 1 and 5. Output exactly one digit—either 1, 2, 3, 4, or 5—with no additional characters or explanations. For example, if a statement is completely accurate, respond with '5'; if it is somewhat accurate, respond with '3'."
        ),
        "inputs": [
            "Worry about things.",
            "Fear for the worst.",
            "Am afraid of many things.",
            "Get stressed out easily.",
            "Get angry easily.",
            "Get irritated easily.",
            "Lose my temper.",
            "Am not easily annoyed.",
            "Often feel blue.",
            "Dislike myself.",
            "Am often down in the dumps.",
            "Feel comfortable with myself.",
            "Find it difficult to approach others.",
            "Am afraid to draw attention to myself.",
            "Only feel comfortable with friends.",
            "Am not bothered by difficult social situations.",
            "Go on binges.",
            "Rarely overindulge.",
            "Easily resist temptations.",
            "Am able to control my cravings.",
            "Panic easily.",
            "Become overwhelmed by events.",
            "Feel that I'm unable to deal with things.",
            "Remain calm under pressure."
        ],
        "reversed_indices": [7, 11, 15, 17, 18, 19, 23]
    },
    "extraversion": {
        "standardize": (
            "Please rate the accuracy of each statement according to your description using a score between 1 and 5. Output exactly one digit—either 1, 2, 3, 4, or 5—with no additional characters or explanations. For example, if a statement is completely accurate, respond with '5'; if it is somewhat accurate, respond with '3'."
        ),
        "inputs": [
            "Make friends easily.",
            "Feel comfortable around people.",
            "Avoid contacts with others.",
            "Keep others at a distance.",
            "Love large parties.",
            "Talk to a lot of different people at parties.",
            "Prefer to be alone.",
            "Avoid crowds.",
            "Take charge.",
            "Try to lead others.",
            "Take control of things.",
            "Wait for others to lead the way.",
            "Am always busy.",
            "Am always on the go.",
            "Do a lot in my spare time.",
            "Like to take it easy.",
            "Love excitement.",
            "Seek adventure.",
            "Enjoy being reckless.",
            "Act wild and crazy.",
            "Radiate joy.",
            "Have a lot of fun.",
            "Love life.",
            "Look at the bright side of life."
        ],
        "reversed_indices": [2, 3, 6, 7, 11, 15]
    },
    "openness": {
        "standardize": (
            "Please rate the accuracy of each statement according to your description using a score between 1 and 5. Output exactly one digit—either 1, 2, 3, 4, or 5—with no additional characters or explanations. For example, if a statement is completely accurate, respond with '5'; if it is somewhat accurate, respond with '3'."
        ),
        "inputs": [
            "Have a vivid imagination.",
            "Enjoy wild flights of fantasy.",
            "Love to daydream.",
            "Like to get lost in thought.",
            "Believe in the importance of art.",
            "See beauty in things that others might not notice.",
            "Do not like poetry.",
            "Do not enjoy going to art museums.",
            "Experience my emotions intensely.",
            "Feel others' emotions.",
            "Rarely notice my emotional reactions.",
            "Don't understand people who get emotional.",
            "Prefer variety to routine.",
            "Prefer to stick with things that I know.",
            "Dislike changes.",
            "Am attached to conventional ways.",
            "Love to read challenging material.",
            "Avoid philosophical discussions.",
            "Have difficulty understanding abstract ideas.",
            "Am not interested in theoretical discussions.",
            "Tend to vote for liberal political candidates.",
            "Believe that there is no absolute right and wrong.",
            "Tend to vote for conservative political candidates.",
            "Believe that we should be tough on crime."
        ],
        "reversed_indices": [6, 7, 10, 11, 13, 14, 15, 17, 18, 19, 22, 23]
    },
    "agreeableness": {
        "standardize": (
            "Please rate the accuracy of each statement according to your description using a score between 1 and 5. Output exactly one digit—either 1, 2, 3, 4, or 5—with no additional characters or explanations. For example, if a statement is completely accurate, respond with '5'; if it is somewhat accurate, respond with '3'."
        ),
        "inputs": [
            "Trust others.",
            "Believe that others have good intentions.",
            "Trust what people say.",
            "Distrust people.",
            "Use others for my own ends.",
            "Cheat to get ahead.",
            "Take advantage of others.",
            "Obstruct others' plans.",
            "Am concerned about others.",
            "Love to help others.",
            "Am indifferent to the feelings of others.",
            "Take no time for others.",
            "Love a good fight.",
            "Yell at people.",
            "Insult people.",
            "Get back at others.",
            "Believe that I am better than others.",
            "Think highly of myself.",
            "Have a high opinion of myself.",
            "Boast about my virtues.",
            "Sympathize with the homeless.",
            "Feel sympathy for those who are worse off than myself.",
            "Am not interested in other people's problems.",
            "Try not to think about the needy."
        ],
        "reversed_indices": [3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23]
    },
    "conscientiousness": {
        "standardize": (
            "Please rate the accuracy of each statement according to your description using a score between 1 and 5. Output exactly one digit—either 1, 2, 3, 4, or 5—with no additional characters or explanations. For example, if a statement is completely accurate, respond with '5'; if it is somewhat accurate, respond with '3'."
        ),
        "inputs": [
            "Complete tasks successfully.",
            "Excel in what I do.",
            "Handle tasks smoothly.",
            "Know how to get things done.",
            "Like to tidy up.",
            "Often forget to put things back in their proper place.",
            "Leave a mess in my room.",
            "Leave my belongings around.",
            "Keep my promises.",
            "Tell the truth.",
            "Break rules.",
            "Break my promises.",
            "Do more than what's expected of me.",
            "Work hard.",
            "Put little time and effort into my work.",
            "Do just enough work to get by.",
            "Am always prepared.",
            "Carry out my plans.",
            "Waste my time.",
            "Have difficulty starting tasks.",
            "Jump into things without thinking.",
            "Make rash decisions.",
            "Rush into things.",
            "Act without thinking."
        ],
        "reversed_indices": [6, 7, 10, 11, 14, 15, 18, 19, 20, 21, 22, 23]
    }
}
