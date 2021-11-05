# make sure to pip install gensim and nltk if you haven't already 
import gensim
import nltk
import re
import string
import matplotlib.pyplot as plt
import numpy as np

doc = [] 

#Open lyrics document
with open('/Users/lindegaard/Desktop/Programming/fall_2_hw_team_6/Text Analytics/Lyrics_2020_Clean') as f:
    contents=f.read()
    doc.append(contents)

# Remove punctuation, then tokenize documents

punc = re.compile( '[%s]' % re.escape( string.punctuation ) )
term_vec = [ ]

for d in doc:
    d = d.lower()
    d = punc.sub( '', d )
    term_vec.append( nltk.word_tokenize( d ) )

# Print resulting term vectors

#for vec in term_vec:
    #print(vec)

# removing stop words 

stop_words = nltk.corpus.stopwords.words( 'english' )

for i in range( 0, len( term_vec ) ):
    term_list = [ ]

    for term in term_vec[ i ]:
        if term not in stop_words:
            term_list.append( term )

    term_vec[ i ] = term_list
print(term_vec)
# Print term vectors with stop words removed

for vec in term_vec:
    print(vec)

####################################################  Print individual character counts #################################################### 

with open('/Users/lindegaard/Desktop/Programming/fall_2_hw_team_6/Text Analytics/Lyrics_2016_Clean') as txt:
    txt = txt.read()
#print(t)
# Convert to lower case, use regex to create a version with no punctuation

t = txt.lower()
#print(t)
t_no_punc = re.sub( r'[^\w\s]', '', t )
#print(t_no_punc)

def print_dict( d ):
    """Print frequency dictionary. Key is 'representation', v
    frequency of representation.

    Args:
    &nbsp; d (dict): Dictionary of (rep,freq) pairs
    """

    keys = list( d.keys() )
    keys.sort()

    for k in keys:
        print( f'{k}: {d[ k ]}; ', end='' )
    print( '' )

# Create punc, no punc dictionaries to hold character frequencies

char_dict = { }
char_dict_no_punc = { }

# Count characters in text w/punctuation

for c in txt:
    char_dict[ c ] = ( 1 if c not in char_dict else char_dict[ c ] + 1 )

# Print results

print( 'Character frequency' )
print_dict( char_dict )

for c in t_no_punc:
    char_dict_no_punc[ c ] = ( 1 if c not in char_dict_no_punc else char_dict_no_punc[ c ] + 1 )

print( 'Character frequency w/o punctuation' )
print_dict( char_dict_no_punc )

# Plot as bar graph

char = list( char_dict.keys() )
char.sort()

freq = [ ]
for c in char:
    freq = freq + [ char_dict[ c ] ]

    # Add any character in punctuation dict but not in no punctuation dict w/freq of zero
    
    if c not in char_dict_no_punc:
        char_dict_no_punc[ c ] = 0
    
char_no_punc = list( char_dict_no_punc.keys() )
char_no_punc.sort()

freq_no_punc = [ ]
for c in char_no_punc:
    freq_no_punc = freq_no_punc + [ char_dict_no_punc[ c ] ]

X = np.arange( len( freq ) )
w = 0.35

fig = plt.figure( figsize=(10,5) )
ax = fig.add_axes( [ 0, 0, 1, 1 ] )
ax.bar( X + 0.00, freq, color='b', width=w, label='w/punc' )
ax.bar( X + 0.33, freq_no_punc, color='orange', width=w, label='w/o punc' )

plt.ylabel( 'Frequency' )
plt.xlabel( 'Character' )
plt.xticks( X + w / 2, char )
plt.legend( loc='best' )
plt.show()




####################################################  Print tern counts!! #################################################### 

# Term frequencies

# Convert text to lower-case term tokens

t = re.sub( r'[^\w\s]', '', txt )
tok = t.lower().split()

# Count term frequencies

d = { }
for term in tok:
    d[ term ] = ( 1 if term not in d else d[ term ] + 1 )

# Print results

print( 'Term frequencies' )
print_dict( d )

# Plot as bar graph

term = list( d.keys() )
term.sort()

freq = [ ]
for t in term:
    freq = freq + [ d[ t ] ]

x_pos = range( len( term ) )
fig = plt.figure( figsize=(15,40) )
ax = fig.add_axes( [ 0, 0, 1, 1 ] )
ax.barh( x_pos, freq, color='b' )

plt.ylabel( 'Term' )
plt.xlabel( 'Frequency' )
plt.yticks( x_pos, term )
plt.show()


####################################################  Print bigram counts!! #################################################### 


# Bigram frequencies

# Convert text to lower-case term tokens

t = re.sub( r'[^\w\s]', '', txt )
tok = t.lower().split()

# Build bigrams, count frequencies

d = { }
for i in range( 1, len( tok ) ):
    bigram = (tok[ i - 1 ],tok[ i ] )
    d[ bigram ] = ( 1 if bigram not in d else d[ bigram ] + 1 )

# Print results

print( 'Bigram frequencies' )
print_dict( d )


####################################################  Print part of speech frequencies!! #################################################### 

def POS_expand( term_POS ):
    """Convert second elements of tuple, shortened POS tag, to expanded POS description

    Args:
     term_POS (tuple): Tuple of (term,POS-tag)

    Returns:
    (tuple): Tuple of (term,POS-tag-plus-description)
    """

    tag = term_POS[ 1 ]
    if tag == 'CC':
        exp = 'coordinating conjunction'
    elif tag == 'CD':
        exp = 'cardinal number'
    elif tag == 'DT':
        exp = 'determiner'
    elif tag == 'EX':
        exp = 'existential there'
    elif tag == 'FW':
        exp = 'foreign word'
    elif tag == 'IN':
        exp = 'preposition'
    elif tag == 'JJ':
        exp = 'adjective'
    elif tag == 'JJR':
        exp = 'comparative adjective'
    elif tag == 'JJS':
        exp = 'superlative adjective'
    elif tag == 'LS':
        exp = 'list item marker'
    elif tag == 'MD':
        exp = 'modal'
    elif tag == 'NN':
        exp = 'noun'
    elif tag == 'NNS':
        exp = 'plural noun'
    elif tag == 'NNP':
        exp = 'proper noun'
    elif tag == 'NNPS':
        exp = 'plural proper noun'
    elif tag == 'PDT':
        exp = 'predeterminer'
    elif tag == 'POS':
        exp = 'possessive ending'
    elif tag == 'PRP':
        exp = 'personal pronoun'
    elif tag == 'PRP$':
        exp = 'possessive pronoun'
    elif tag == 'RB':
        exp = 'adverb'
    elif tag == 'RBR':
        exp = 'comparative adverb'
    elif tag == 'RBS':
        exp = 'superlative adverb'
    elif tag == 'RP':
        exp = 'particle'
    elif tag == 'SYM':
        exp = 'symbol'
    elif tag == 'TO':
        exp = 'to'
    elif tag == 'UH':
        exp = 'interjection'
    elif tag == 'VB':
        exp = 'verb'
    elif tag == 'VBD':
        exp = 'past tense verb'
    elif tag == 'VBG':
        exp = 'gerund verb'
    elif tag == 'VBN':
        exp = 'past participle verb'
    elif tag == 'VBP':
        exp = 'non-3rd person verb'
    elif tag == 'VBZ':
        exp = '3rd person verb'
    elif tag == 'WDT':
        exp = 'wh-determiner'
    elif tag == 'WP':
        exp = 'wh-pronoun'
    elif tag == 'WP$':
        exp = 'possessive wh-pronoun'
    elif tag == 'WRB':
        exp = 'wh-adverb'
    else:
        exp = 'default'

    return (term_POS[ 0 ],tag + ' ' + exp)

# (term,part-of-speech) frequencies

# Convert text to lower-case term tokens, POS tag them

t = re.sub( r'[^\w\s]', '', txt )
tok = t.lower().split()
tok = nltk.pos_tag( tok )

# Build (term,POS) and POS dictionaries, count frequencies

d = { }
d_POS = { }

for term_POS in tok:
    term_POS = POS_expand( term_POS )
    d[ term_POS ] = ( 1 if term_POS not in d else d[ term_POS ] + 1 )

    POS = term_POS[ 1 ]
    d_POS[ POS ] = ( 1 if POS not in d_POS else d_POS[ POS ] + 1 )

# Print results

print( '(term,POS) frequencies' )
print_dict( d )
print( 'POS frequencies' )
print_dict( d_POS )

# Plot POS frequencies as bar graph

POS = list( d_POS.keys() )
POS.sort()

freq = [ ]
for p in POS:
    freq = freq + [ d_POS[ p ] ]

x_pos = range( len( POS ) )
fig = plt.figure( figsize=(10,10) )
ax = fig.add_axes( [ 0, 0, 1, 1 ] )
ax.barh( x_pos, freq, color='b' )

plt.ylabel( 'POS' )
plt.xlabel( 'Frequency' )
plt.yticks( x_pos, POS )
plt.show()

#################################################### Porter stemming #################################################### 



# Porter stem remaining terms

porter = nltk.stem.porter.PorterStemmer()

for i in range( 0, len( term_vec ) ):
    for j in range( 0, len( term_vec[ i ] ) ):
        term_vec[ i ][ j ] = porter.stem( term_vec[ i ][ j ] )

# Print term vectors with stop words removed

for vec in term_vec:
    print(vec)
    
#################################################### Practice problem 2 to create term frequency matrix #################################################### 

def porter_stem( txt ):
    """Porter stem terms in text block

    Args:
    &nbsp; txt (list of string): Text block as list of individual terms

    Returns:
    &nbsp; (list of string): Text block with terms Porter stemmed
    """

    porter = nltk.stem.porter.PorterStemmer()

    for i in range( 0, len( txt ) ):
        txt[ i ] = porter.stem( txt[ i ] )

    return txt


def remove_stop_word( txt ):
    """Remove all stop words from text blo
    Args:
    &nbsp; txt (list of string): Text block as list of individual terms

    Returns:
    &nbsp; (list of string): Text block with stop words removed
    """

    term_list = [ ]
    stop_word = nltk.corpus.stopwords.words( 'english' )

    for term in txt:
        term_list += ( [ ] if term in stop_word else [ term ] )

    return term_list


# Mainline

#Re create the doc document (a list of the lyric text)

doc = [] 

#Open lyrics document
with open('/Users/lindegaard/Desktop/Programming/fall_2_hw_team_6/Text Analytics/Lyrics_2016_Clean') as f:
    contents=f.read()
    doc.append(contents)
    
# Remove punctuation except hyphen

punc = string.punctuation.replace( '-', '' )
for i in range( 0, len( doc ) ):
    doc[ i ] = re.sub( '[' + punc + ']+', '', doc[ i ] )

# Lower-case and tokenize text

for i in range( 0, len( doc ) ):
    doc[ i ] = doc[ i ].lower().split()

# Stop word remove w/nltk stop word list, then Porter stem

for i in range( 0, len( doc ) ):
    doc[ i ] = remove_stop_word( doc[ i ] )
    doc[ i ] = porter_stem( doc[ i ] )

# Create list of all (unique) stemmed terms

term_list = set( doc[ 0 ] )
for i in range( 1, len( doc ) ):
    term_list = term_list.union( doc[ i ] )
term_list = sorted( term_list )

# Count occurrences of unique terms in each document

n = len( term_list )
freq = [ ]
for i in range( 0, len( doc ) ):
    freq.append( [ 0 ] * n )
    for term in doc[ i ]:
        pos = term_list.index( term )
        freq[ -1 ][ pos ] += 1

# Print transposed term-frequency list for easier viewing #UNABLE TO GET RUNNING - F STRING FORMAT ERROR
print( '....................mice..lord..1984' )
for i in range( 0, len( term_list ) ):
    print( f"{term_list[ i ]: &lt;{20}}", end='' )
    for j in range( 0, len( doc ) ):
        print( f"{freq[ j ][ i ]:4d}", end='' )
    print( '' )


#################################################### Practice problem 3 to do clustering #################################################### 


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

#Re create the doc document (a list of the lyric text)

doc = [] 

#Open lyrics document
with open('/Users/lindegaard/Desktop/Programming/fall_2_hw_team_6/Text Analytics/Lyrics_2020_Clean') as f:
    contents=f.read()
    doc.append(contents)
print(doc)

full_sent = [ ]
for i in range( 0, len( doc ) ):
    sent = re.sub( r'[\.!\?\n]"', '\'', doc[ i ] )
    full_sent += re.split( '[\.!\?\n]', sent )
full_sent = [sent.strip() for sent in full_sent]
print(full_sent)


# Remove empty sentences

i = 0
while i < len( full_sent ):
    if len( full_sent[ i ] ) == 0:
        del full_sent[ i ]
    else:
        i += 1

# Remove punctuation

sent = [ ]

punc = string.punctuation.replace( '-', '' )
for i in range( 0, len( full_sent ) ):
    sent.append( re.sub( '[' + punc + ']+', '', full_sent[ i ] ) )

# Porter stem

porter = nltk.stem.porter.PorterStemmer()
stems = { }

for i in range( 0, len( sent ) ):
    tok = sent[ i ].split()
    for j in range( 0, len( tok ) ):
        if tok[ j ] not in stems:
            stems[ tok[ j ] ] = porter.stem( tok[ j ] )
        tok[ j ] = stems[ tok[ j ] ]

    sent[ i ] = ' '.join( tok )
print(sent)
# Remove empty sentences after stop word removal

i = 0
while i < len( sent ):
    if len( sent[ i ] ) == 0:
        del sent[ i ]
    else:
        i += 1

# Convert frequencies to TF-IDF values, get cosine similarity
# of all pairs of documents

tfidf = TfidfVectorizer( stop_words='english', max_df=0.8, max_features=1000 )
term_vec = tfidf.fit_transform( sent )
X = cosine_similarity( term_vec )

# Fit vectors to clusters

clust = KMeans( n_clusters=12, random_state=1 ).fit( X )
print( clust.labels_ )

for i in range( 0, len( set( clust.labels_ ) ) ):
    print( f'Cluster {i}:' )

    for j in range( 0, len( clust.labels_ ) ):
        if clust.labels_[ j ] == i:
            print( full_sent[ j ].replace( '"', '' ).strip() )

    print()


#HEALEY EXAMPLE #################################################### #################################################### #################################################### #################################################### 
txt = [
    'I was thirty-seven then. Test sentence. Sentence. Hello.',
    'Test sentences all day in part 2. This is document 2.',
    'Test sentences for part 3. This is doc 3.',
    'Test sentence for part 4. This is document 4. How does this show up. Apple. Bananas.'
]
print(txt)
# Split text blocks into sentences

full_sent = [ ]
for i in range( 0, len( txt ) ):
    sent = re.sub( r'[\.!\?]"', '"', txt[ i ] )
    print(sent)
    full_sent += re.split( '[\.!\?]', sent )
    print(full_sent)
full_sent = [sent.strip() for sent in full_sent]
print(full_sent)
# Remove empty sentences

i = 0
while i < len( full_sent ):
    if len( full_sent[ i ] ) == 0:
        del full_sent[ i ]
    else:
        i += 1

# Remove punctuation

sent = [ ]

punc = string.punctuation.replace( '-', '' )
for i in range( 0, len( full_sent ) ):
    sent.append( re.sub( '[' + punc + ']+', '', full_sent[ i ] ) )

# Porter stem

porter = nltk.stem.porter.PorterStemmer()
stems = { }

for i in range( 0, len( sent ) ):
    tok = sent[ i ].split()
    for j in range( 0, len( tok ) ):
        if tok[ j ] not in stems:
            stems[ tok[ j ] ] = porter.stem( tok[ j ] )
        tok[ j ] = stems[ tok[ j ] ]

    sent[ i ] = ' '.join( tok )
print(sent)
# Remove empty sentences after stop word removal

i = 0
while i < len( sent ):
    if len( sent[ i ] ) == 0:
        del sent[ i ]
    else:
        i += 1

# Convert frequencies to TF-IDF values, get cosine similarity
# of all pairs of documents

tfidf = TfidfVectorizer( stop_words='english', max_df=0.8, max_features=1000 )
term_vec = tfidf.fit_transform( sent )
X = cosine_similarity( term_vec )
print(tfidf)
print(term_vec)
print(X)
# Fit vectors to clusters

clust = KMeans( n_clusters=5, random_state=1 ).fit( X )
print( clust.labels_ )

for i in range( 0, len( set( clust.labels_ ) ) ):
    print( f'Cluster {i}:' )

    for j in range( 0, len( clust.labels_ ) ):
        if clust.labels_[ j ] == i:
            print( full_sent[ j ].replace( '"', '' ).strip() )

    print()


#################################################### Practice problem 3 to determine HOW MANY CLUSTERS #################################################### 


from scipy.spatial.distance import cdist

def elbow( X, max_clust=25 ):
    distort = [ ]
    inertia = [ ]

    map_distort = { }
    map_inertia = { }

    elbow_distort = 1
    elbow_inertia = 1

    K = range( 1, max_clust )
    for k in K:
        kmean_model = KMeans( n_clusters=k )
        kmean_model.fit( X )

        distort.append( sum( np.min( cdist( X, kmean_model.cluster_centers_, 'euclidean' ), axis=1 ) ) / X.shape[ 0 ] )
        inertia.append( kmean_model.inertia_ )

        map_distort[ k ] = sum( np.min( cdist( X, kmean_model.cluster_centers_, 'euclidean' ), axis=1 ) ) / X.shape[ 0 ]
        map_inertia[ k ] = kmean_model.inertia_

    prev_k = ''
    prev_v = 0
    prev_pct = 0
    for i,(k,v) in enumerate( map_distort.items() ):
        if prev_k == '':
            print( f'{k}: {v:.4f}' )
            prev_k = str( k )
            prev_v = v
            continue

        print( f'{k}: {v:.4f} ', end='' )

        diff_v = prev_v - v
        diff_v_pct = diff_v / prev_v * 100.0
        print( f'{diff_v:.4f}, {diff_v_pct:.2f}%' )

        if i > 2 and prev_pct - diff_v_pct < 0.5:
            elbow_distort = i + 1
            break

        prev_k = str( k )
        prev_v = v
        prev_pct = diff_v_pct

    print()

    prev_k = ''
    prev_v = 0
    prev_pct = 0
    for i,(k,v) in enumerate( map_inertia.items() ):
        if prev_k == '':
            print( f'{k}: {v:.4f}' )
            prev_k = str( k )
            prev_v = v
            continue

        print( f'{k}: {v:.4f} ', end='' )

        diff_v = prev_v - v
        diff_v_pct = diff_v / prev_v * 100.0
        print( f'{diff_v:.4f}, {diff_v_pct:.2f}%' )

        if i > 2 and prev_pct - diff_v_pct < 0.5:
            elbow_inertia = i + 1
            break

        prev_k = str( k )
        prev_v = v
        prev_pct = diff_v_pct

    return max( elbow_distort, elbow_inertia )


#################################################### Practice problem 3 to print topic clusters #################################################### 


from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer

# Count raw term frequencies

count = CountVectorizer( stop_words='english' )
term_vec = count.fit_transform( sent )

n_topic = 5

# Build a string list of [ 'Topic 1', 'Topic 2', ..., 'Topic n' ]

col_nm = [ ]
for i in range( 1, n_topic + 1 ):
    col_nm += [ f'Topic {i}' ]

# Fit an LDA model to the term vectors, get cosine similarities

lda_model = LDA( n_components=n_topic )
concept = lda_model.fit_transform( term_vec )
X = cosine_similarity( concept )

# Determine ideal number of clusters using kmeans with elbow method
#elbow( X, 10)

# Print top 20 terms for each topic

feat = count.get_feature_names()
topic_list = [ ]
for i,topic in enumerate( lda_model.components_ ):
    top_n = [ feat[ i ] for i in topic.argsort()[ -20: ] ]
    top_feat = ' '.join( top_n )
    topic_list.append( f"topic_{'_'.join(top_n[ :3 ] ) }" )

    print( f'Topic {i}: {top_feat}' )
print()

# Cluster sentences and print clusters

clust = KMeans( n_clusters=5 ).fit( concept )

for i in range( 0, len( set( clust.labels_ ) ) ):
    print( f'Cluster {i}:' )
    for j in range( 0, len( clust.labels_ ) ):
        if clust.labels_[ j ] != i:
            continue
        print( full_sent[ j ] )

    print()


#################################################### Practice problem  4 for SENTIMENT ANALYSIS #################################################### 

#Re create the doc document (a list of the lyric text)

doc = [] 

#Open lyrics document
with open('/Users/lindegaard/Desktop/Programming/fall_2_hw_team_6/Text Analytics/Lyrics_2016_Clean') as f:
    contents=f.read()
    doc.append(contents)
print(doc)

full_sent = [ ]
for i in range( 0, len( doc ) ):
    sent = re.sub( r'[\.!\?\n]"', '\'', doc[ i ] )
    full_sent += re.split( '[\.!\?\n]', sent )
full_sent = [sent.strip() for sent in full_sent]
print(full_sent)


# Remove empty sentences

i = 0
while i < len( full_sent ):
    if len( full_sent[ i ] ) == 0:
        del full_sent[ i ]
    else:
        i += 1

# Remove punctuation

sent = [ ]

punc = string.punctuation.replace( '-', '' )
for i in range( 0, len( full_sent ) ):
    sent.append( re.sub( '[' + punc + ']+', '', full_sent[ i ] ) )

# Porter stem

porter = nltk.stem.porter.PorterStemmer()
stems = { }

for i in range( 0, len( sent ) ):
    tok = sent[ i ].split()
    for j in range( 0, len( tok ) ):
        if tok[ j ] not in stems:
            stems[ tok[ j ] ] = porter.stem( tok[ j ] )
        tok[ j ] = stems[ tok[ j ] ]

    sent[ i ] = ' '.join( tok )
print(sent)
# Remove empty sentences after stop word removal

i = 0
while i < len( sent ):
    if len( sent[ i ] ) == 0:
        del sent[ i ]
    else:
        i += 1


nltk.download( 'vader_lexicon' )
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#txt = 'Two men, dressed in denim jackets and trousers and wearing "black, shapeless hats," walk single-file down a path near the pool. Both men carry blanket rolls  called bindles  on their shoulders. The smaller, wiry man is George Milton. Behind him is Lennie Small, a huge man with large eyes and sloping shoulders, walking at a gait that makes him resemble a huge bear. When Lennie drops near the pool\'s edge and begins to drink like a hungry animal, George cautions him that the water may not be good. This advice is necessary because Lennie is retarded and doesn\'t realize the possible dangers. The two are on their way to a ranch where they can get temporary work, and George warns Lennie not to say anything when they arrive. Because Lennie forgets things very quickly, George must make him repeat even the simplest instructions. Lennie also likes to pet soft things. In his pocket, he has a dead mouse which George confiscates and throws into the weeds beyond the pond. Lennie retrieves the dead mouse, and George once again catches him and gives Lennie a lecture about the trouble he causes when he wants to pet soft things (they were run out of the last town because Lennie touched a girl\'s soft dress, and she screamed). Lennie offers to leave and go live in a cave, causing George to soften his complaint and tell Lennie perhaps they can get him a puppy that can withstand Lennie\'s petting. As they get ready to eat and sleep for the night, Lennie asks George to repeat their dream of having their own ranch where Lennie will be able to tend rabbits. George does so and then warns Lennie that, if anything bad happens, Lennie is to come back to this spot and hide in the brush. Before George falls asleep, Lennie tells him they must have many rabbits of various colors.'

# Convert to sentences, create VADER sentiment analyzer

#sentence = txt.split( '.' )
sentiment = SentimentIntensityAnalyzer()

for i in range( 0, len( sent ) ):

    # Print sentence's compound sentiment score

    score = sentiment.polarity_scores( sent[ i ] )
    print( sent[ i ] )
    print( 'Sentiment:', score[ 'compound' ] )



