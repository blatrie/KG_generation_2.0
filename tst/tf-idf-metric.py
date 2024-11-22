from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import sys
sys.path.insert(0, '../src/pipeline/')
from KB_generation import get_kb, KB

nltk.download('wordnet')

# Nouveaux textes sur l'économie
documents = [
    """ Predicting how White House policy is going to affect the American economy is always fraught with uncertainty. Donald J. Trump’s return to the White House has taken the doubt up a notch.
Mr. Trump has proposed or hinted at a range of policies — including drastically higher tariffs, mass deportations, deregulation and a fraught relationship with the Federal Reserve as it sets interest rates — that could shape the economy in complex ways.
There are two multiplicative sources of uncertainty: One, of course, is what they’re going to do, said Michael Feroli, the chief U.S. economist at J.P. Morgan. The other is: Even if you know what they’re going to do, what is it going to mean for the economy?
What forecasters do know is that America’s economy is solid heading into 2025, with low unemployment, solid wage gains, gradually declining Federal Reserve interest rates, and inflation that has been slowly returning to a normal pace after years of rapid price increases. Factory construction took off under the Biden administration, and those facilities will be slowly opening their doors in the coming years.
But what comes next for growth and for inflation is unclear — especially because when it comes to huge issues like whether or not to assert more control over the Federal Reserve, Mr. Trump is getting different advice from different people in his orbit. Here are some of the crucial wild cards. Tariffs: Likely Inflationary. How Much Is Unclear.
If economists agree about one thing regarding Mr. Trump’s policies, it is that his tariff proposals could boost prices for consumers and lift inflation. But the range of estimates over how much is wide.
When Mr. Trump imposed tariffs during his first term, they pushed up prices for consumers, but only slightly.
But what he is promising now could be more sweeping. Mr. Trump has floated a variety of plans this time, but they have often included across-the-board tariffs and levies of 60 percent or more on goods from China. It’s not at all clear that this is going to be anything like it was the last time around, said Omair Sharif, founder of Inflation Insights. Fed staff suggested back in 2018 that the central bank could hold steady in the face of price increases coming from tariffs, assuming that consumers and investors expected inflation to remain fairly steady over time. But Jerome H. Powell, the Fed chair, acknowledged last week that this time, we’re in a different situation.
Six years ago, inflation had been slow for more than a decade, so a small bump to prices barely registered. This time, inflation has recently been rapid, which could change how price increases filter through the economy. Deportations: Could Slow Growth, but Details Matter.
Tariffs are not the only thing that economists are struggling to figure out. It is also unclear what immigration policy might look like under a Trump administration, making it difficult to model the possible economic impact.
Mr. Trump has repeatedly promised the biggest deportation in American history while on the campaign trail, and he has at times hinted at high-skill immigration reform. During an interview on the All In podcast, he said what I will do is, you graduate from a college, I think you should get, automatically, as part of your diploma, a green card to be able to stay in this country.
But reforming the legal immigration system for highly educated workers would require Congress’s participation, and the campaign barely talked about such plans.
And when it comes to lower-skill immigration, while there are things the administration can do unilaterally to start deportations, there’s a huge range of estimates around how many people might be successfully removed. It’s hard to round people up, cases might get caught up in the court system and newcomers may replace recent deportees.
Economists at Goldman Sachs have estimated that a Trump administration might expel anywhere from 300,000 to 2.1 million people in 2025. The low end is based on removal trends from Mr. Trump’s earlier term in office, and the higher end is based on deportation trends from the Eisenhower administration in the 1950s, which Mr. Trump has suggested he would like to emulate.
Kent Smetters, the faculty director of the Penn Wharton Budget Model, which measures the fiscal impact of public policies, said he was assuming that the administration managed to deport a few hundred thousand people in its first year in office — which he said would have a relatively small effect on either growth or inflation in an economy the size of America’s.
It’s not as big of an effect as you might think, he said. It’s not the same as if you were getting rid of all undocumented workers, and they’re going to fall far short of that, is my guess.”The tariffs Mr. Trump put in effect in 2018 do not offer a good economic precedent for how such a large tariff on goods coming from China in particular might play out, Mr. Sharif said. The earlier rounds heavily affected imports like aluminum, steel and other economic inputs, rather than final products.
These are not things you go out and buy at Home Depot on the weekend, he said. The new ones, by contrast, would hit things like T-shirts and tennis shoes, so they could feed much more directly into consumer price inflation."""
]

# Initialiser le vectorizer avec suppression des stop words en anglais
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1, 2))
# vectorizer = TfidfVectorizer(stop_words='english', min_df=0.5, ngram_range=(1, 2)) # min_df = 0.1 : les mots qui apparaissent dans moins de 10% des documents seront ignorés. 
# vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(documents)

# Obtenir les mots du vocabulaire
feature_names = vectorizer.get_feature_names_out()

# Calculer la moyenne des scores TF-IDF pour chaque mot sur tous les documents
tfidf_scores = np.mean(tfidf_matrix.toarray(), axis=0)

# Obtenir les indices des 6 scores les plus élevés
top_indices = np.argsort(tfidf_scores)[-20:][::-1]

# Afficher les 6 mots les plus importants et leurs scores
important_words = []
print("Les 20 mots les plus importants :")
for index in top_indices:
    important_words.append(feature_names[index])
    print(f"{feature_names[index]}: {tfidf_scores[index]:.4f}")

# text = "The ongoing disruptions in the global supply chain have led to increased costs for businesses worldwide. Shipping delays, rising raw material prices, and labor shortages have all contributed to inflationary pressures. Companies are now facing difficult decisions regarding price adjustments to maintain profitability. Governments are considering interventions to ease these pressures, but the long-term outlook remains uncertain as geopolitical tensions and regulatory changes further complicate the recovery process. The rapid advancement of financial technology, or fintech, is transforming the landscape of global markets. Digital payments, blockchain, and decentralized finance are disrupting traditional banking systems, enabling faster and more efficient transactions. However, regulators are struggling to keep up with the pace of innovation, raising concerns about data privacy and financial security. As fintech companies continue to attract investment, there is growing debate about whether these innovations will lead to greater financial inclusion or exacerbate existing inequalities. Climate change poses significant risks to the global economy, forcing policymakers to rethink their strategies. The shift towards a green economy involves significant investments in renewable energy, electric vehicles, and sustainable agriculture. However, the transition is not without challenges, as fossil fuel-dependent industries resist regulatory changes. Economists argue that without proper incentives and international cooperation, the costs of climate inaction could far outweigh the investments needed for a sustainable future. Governments are exploring carbon pricing and green bonds as potential solutions to finance this transition."

text = """ Predicting how White House policy is going to affect the American economy is always fraught with uncertainty. Donald J. Trump’s return to the White House has taken the doubt up a notch.
Mr. Trump has proposed or hinted at a range of policies — including drastically higher tariffs, mass deportations, deregulation and a fraught relationship with the Federal Reserve as it sets interest rates — that could shape the economy in complex ways.
There are two multiplicative sources of uncertainty: One, of course, is what they’re going to do,” said Michael Feroli, the chief U.S. economist at J.P. Morgan. The other is: Even if you know what they’re going to do, what is it going to mean for the economy?
What forecasters do know is that America’s economy is solid heading into 2025, with low unemployment, solid wage gains, gradually declining Federal Reserve interest rates, and inflation that has been slowly returning to a normal pace after years of rapid price increases. Factory construction took off under the Biden administration, and those facilities will be slowly opening their doors in the coming years.
But what comes next for growth and for inflation is unclear — especially because when it comes to huge issues like whether or not to assert more control over the Federal Reserve, Mr. Trump is getting different advice from different people in his orbit. Here are some of the crucial wild cards. Tariffs: Likely Inflationary. How Much Is Unclear.
If economists agree about one thing regarding Mr. Trump’s policies, it is that his tariff proposals could boost prices for consumers and lift inflation. But the range of estimates over how much is wide.
When Mr. Trump imposed tariffs during his first term, they pushed up prices for consumers, but only slightly.
But what he is promising now could be more sweeping. Mr. Trump has floated a variety of plans this time, but they have often included across-the-board tariffs and levies of 60 percent or more on goods from China. It’s not at all clear that this is going to be anything like it was the last time around,” said Omair Sharif, founder of Inflation Insights. Fed staff suggested back in 2018 that the central bank could hold steady in the face of price increases coming from tariffs, assuming that consumers and investors expected inflation to remain fairly steady over time. But Jerome H. Powell, the Fed chair, acknowledged last week that this time, we’re in a different situation.”
Six years ago, inflation had been slow for more than a decade, so a small bump to prices barely registered. This time, inflation has recently been rapid, which could change how price increases filter through the economy. Deportations: Could Slow Growth, but Details Matter.
Tariffs are not the only thing that economists are struggling to figure out. It is also unclear what immigration policy might look like under a Trump administration, making it difficult to model the possible economic impact.
Mr. Trump has repeatedly promised the biggest deportation in American history while on the campaign trail, and he has at times hinted at high-skill immigration reform. During an interview on the All In” podcast, he said “what I will do is, you graduate from a college, I think you should get, automatically, as part of your diploma, a green card to be able to stay in this country.”
But reforming the legal immigration system for highly educated workers would require Congress’s participation, and the campaign barely talked about such plans.
And when it comes to lower-skill immigration, while there are things the administration can do unilaterally to start deportations, there’s a huge range of estimates around how many people might be successfully removed. It’s hard to round people up, cases might get caught up in the court system and newcomers may replace recent deportees.
Economists at Goldman Sachs have estimated that a Trump administration might expel anywhere from 300,000 to 2.1 million people in 2025. The low end is based on removal trends from Mr. Trump’s earlier term in office, and the higher end is based on deportation trends from the Eisenhower administration in the 1950s, which Mr. Trump has suggested he would like to emulate.
Kent Smetters, the faculty director of the Penn Wharton Budget Model, which measures the fiscal impact of public policies, said he was assuming that the administration managed to deport a few hundred thousand people in its first year in office — which he said would have a relatively small effect on either growth or inflation in an economy the size of America’s.
It’s not as big of an effect as you might think, he said. It’s not the same as if you were getting rid of all undocumented workers, and they’re going to fall far short of that, is my guess.The tariffs Mr. Trump put in effect in 2018 do not offer a good economic precedent for how such a large tariff on goods coming from China in particular might play out, Mr. Sharif said. The earlier rounds heavily affected imports like aluminum, steel and other economic inputs, rather than final products.
These are not things you go out and buy at Home Depot on the weekend,” he said. The new ones, by contrast, would hit things like T-shirts and tennis shoes, so they could feed much more directly into consumer price inflation."""

print(important_words)
lemmatizer = WordNetLemmatizer()
for i in range(len(important_words)):
    word = lemmatizer.lemmatize(important_words[i])
    important_words[i] = word

print(important_words)
print("")
print("")
myKB = KB()
myKB = get_kb(text)[0]
# print(myKB.relations)

nodes = []
for rel in myKB.relations:
    for node in [rel['head'], rel['tail']]:
        word = lemmatizer.lemmatize(node)
        if word not in nodes:
            nodes.append(word)

print(nodes)

def compute_tf_idf_metric(importants, graph_nodes):
    nb = 0
    max = len(graph_nodes)
    for node in graph_nodes:
        if node in importants:
            nb += 1
    return nb/max

print(f"Accuracy of tf-idf metric : {100*compute_tf_idf_metric(important_words, nodes):.2f}.%")